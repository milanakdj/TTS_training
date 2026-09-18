"""Seed-VC over the oov_core clips, round-robin across long-enough references.

Gates, both earned rather than assumed:
  * speaker  -- cos(out, its reference) via CAMPPlus. MFCC-mean cosine saturates
                near 1.0 on this material (all 10 pilot clips landed 0.96-0.998)
                and cannot discriminate; CAMPPlus spread them 0.567-0.904.
  * reference length -- only refs with >=MIN_REF s of *post-VAD* speech are used.
                In the pilot, corr(ref duration, speaker score) = +0.38 and both
                failures used the two shortest refs, while every PASS used 10 s+.
                One ref was trimmed 72% yet still scored 0.890, so what matters
                is what survives the trim, not the raw length.
"""
import os, sys, json, random, time, io, argparse
os.environ.setdefault("D", "/root/tts/TTS_training/synthetic_pipeline")
D = os.environ["D"]
os.chdir(f"{D}/vc_service/seed-vc"); sys.path.insert(0, os.getcwd()); sys.path.insert(0, f"{D}/vc_service")
import numpy as np, soundfile as sf, librosa, torch, torchaudio, warnings
warnings.filterwarnings("ignore")
import server

STAGE = "/workspace/oov_distill/stage"
REFS  = os.environ.get("REF_POOL", "/workspace/oov_distill/out/ref_pool")
SRCDIR = "/workspace/oov_distill/out/vc_src"

def spk_fn(camp, dev):
    def f(x, sr):
        if x.ndim > 1: x = x.mean(1)
        if sr != 16000: x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        w = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).to(dev)
        fb = torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
        fb = fb - fb.mean(0, keepdim=True)
        with torch.no_grad(): e = camp(fb.unsqueeze(0)).squeeze(0).cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)
    return f

def usable_refs(min_ref):
    """Keep only references with enough speech left AFTER the same VAD trim the
    converter applies -- otherwise we select on a length the model never sees."""
    keep = []
    for fn in sorted(f for f in os.listdir(REFS) if f.endswith(".wav")):
        p = os.path.join(REFS, fn)
        try:
            y, sr = sf.read(p, dtype="float32")
            if y.ndim > 1: y = y.mean(1)
            yt, _ = server.vad_trim_reference(y, sr)
            if len(yt) / sr >= min_ref: keep.append(fn)
        except Exception: pass
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude-core", action="store_true",
                    help="convert everything EXCEPT the already-converted oov_core")
    ap.add_argument("--all-clips", action="store_true",
                    help="convert every clean clip, not just the oov_core subset")
    ap.add_argument("--out", default="/workspace/oov_distill/out/vc")
    ap.add_argument("--min-ref", type=float, default=8.0)
    ap.add_argument("--min-spk-cos", type=float, default=0.85)
    ap.add_argument("--diffusion-steps", type=int, default=25)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = [json.loads(l) for l in open("/workspace/oov_distill/out/selected.jsonl", encoding="utf-8")
            ][:]
    if a.exclude_core:
        rows = [r for r in rows if not r.get("oov_core")]   # the core is already converted
    elif not a.all_clips:
        rows = [r for r in rows if r.get("oov_core")]
    if a.limit: rows = rows[:a.limit]
    print(f"sources: {len(rows)} clips / {sum(r['dur'] for r in rows)/3600:.2f} h", flush=True)

    server.load_models()
    camp = server.STATE["campplus_model"]; dev = next(camp.parameters()).device
    spk = spk_fn(camp, dev)

    refs = usable_refs(a.min_ref)
    print(f"refs usable at >={a.min_ref}s post-VAD: {len(refs)} of "
          f"{len([f for f in os.listdir(REFS) if f.endswith('.wav')])}", flush=True)
    rng = random.Random(0); rng.shuffle(refs)

    # sources are pre-extracted to wav: the VC venv has no pyarrow, and this
    # also avoids re-reading a 126 MB row group per clip.
    #
    # The manifest is appended and flushed per row. The first full-corpus run
    # buffered it in memory and wrote at the end, so when it was killed at
    # 4008/25338 every score went with it -- the wavs survived, the numbers did
    # not. Rows already in the manifest are skipped; wavs on disk without a
    # manifest row are scored in place ("backfill") rather than reconverted.
    man_path = f"{a.out}/vc_manifest.jsonl"
    done_man = set()
    if os.path.exists(man_path):
        for line in open(man_path, encoding="utf-8"):
            try: done_man.add(json.loads(line)["id"])
            except Exception: pass
    print(f"manifest already holds {len(done_man)} rows", flush=True)
    mf = open(man_path, "a", encoding="utf-8")

    man = []; t_audio = t_wall = 0.0; npass = 0; nconv = nback = 0
    for k, r in enumerate(rows):
        outp = f"{a.out}/{r['id']}_vc.wav"
        on_disk = os.path.exists(outp)
        if on_disk and r["id"] in done_man: continue
        src = f"{SRCDIR}/{r['id']}.wav"
        if not os.path.exists(src):
            print(f"  {r['id']} missing source wav, skipped", flush=True); continue
        # refs[k % len(refs)] over a fixed row order and a seed-0 shuffle, so a
        # backfill reproduces the exact pair that produced the file on disk.
        ref = refs[k % len(refs)]; rp = os.path.join(REFS, ref)
        x, sr = sf.read(src, dtype="float32")
        dt = None
        if on_disk:
            try: wav, osr = sf.read(outp, dtype="float32")
            except Exception as e:
                print(f"  {r['id']} unreadable output ({type(e).__name__}), reconverting", flush=True)
                on_disk = False
        if not on_disk:
            t0 = time.time()
            try:
                wav, osr = server.run_voice_conversion(src, rp, diffusion_steps=a.diffusion_steps)
            except Exception as e:
                print(f"  {r['id']} FAILED {type(e).__name__}: {str(e)[:70]}", flush=True); continue
            dt = time.time() - t0
        wav = np.asarray(wav, dtype="float32")
        yr, rsr = sf.read(rp, dtype="float32")
        cos_ref = float(spk(wav, osr) @ spk(yr, rsr))
        cos_src = float(spk(wav, osr) @ spk(x, sr))
        ok = cos_ref >= a.min_spk_cos
        npass += ok
        if on_disk:
            nback += 1
        else:
            sf.write(outp, wav, osr)
            t_audio += len(wav) / osr; t_wall += dt; nconv += 1
        m = dict(id=r["id"], source_id=r["id"], text=r["text"], ref=ref,
                 cos_out_ref=round(cos_ref, 3), cos_out_src=round(cos_src, 3),
                 passed=bool(ok), dur=round(len(wav)/osr, 2),
                 vc_s=(None if dt is None else round(dt, 2)), backfilled=bool(on_disk),
                 n_oov=r.get("n_oov"), n_oov_learnable=r.get("n_oov_learnable"))
        man.append(m); mf.write(json.dumps(m, ensure_ascii=False) + "\n"); mf.flush()
        if (k + 1) % 10 == 0 or k < 5:
            rt = f"{t_audio/t_wall:.1f}x RT" if t_wall > 0 else "backfill only"
            took = "--" if dt is None else f"{dt:.1f}s"
            print(f"  [{k+1}/{len(rows)}] {r['id']} cos_ref={cos_ref:.3f} "
                  f"({'PASS' if ok else 'below gate'})  {took}  running {rt}  "
                  f"pass-rate {npass/len(man):.0%}", flush=True)
    mf.close()
    if man:
        cr = np.array([m["cos_out_ref"] for m in man]); cs = np.array([m["cos_out_src"] for m in man])
        print(f"\nconverted {nconv} | backfilled {nback} | {t_audio/3600:.2f} h in {t_wall/60:.1f} min = "
              f"{t_audio/max(t_wall,1e-9):.1f}x realtime (GPU shared)")
        print(f"cos(out,ref)  mean {cr.mean():.3f}  p10 {np.percentile(cr,10):.3f}  p90 {np.percentile(cr,90):.3f}")
        print(f"cos(out,src)  mean {cs.mean():.3f}")
        print(f"passing >={a.min_spk_cos}: {npass}/{len(man)} = {npass/len(man):.0%}")


if __name__ == "__main__":
    main()
