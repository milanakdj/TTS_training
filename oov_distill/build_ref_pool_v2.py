"""Reference pool v2 -- selected on POST-VAD speech, across every usable speaker.

v1 selected on raw duration and only 169 of its 772 clips survived the converter's
own VAD trim at >=8 s. Across 25,338 clips that would reuse each reference ~150
times, which defeats the point of converting at all. So here the length test is
the same trim the converter applies, and the speaker cap is lifted from 400 to
everyone who qualifies.
"""
import os, sys, json, glob, random, hashlib, collections
D = os.environ.get("D", "/root/tts/TTS_training/synthetic_pipeline")
os.chdir(f"{D}/vc_service/seed-vc"); sys.path.insert(0, os.getcwd()); sys.path.insert(0, f"{D}/vc_service")
import numpy as np, soundfile as sf, soxr, warnings; warnings.filterwarnings("ignore")
import server   # module import only -- load_models() is never called, no GPU touched

NE = {"ne", "nepali", "ne-np", "ne_np"}
SRC = "/workspace/proc_data_new/indicvoices-r-long"
REMAP = ("/projects/data/ttsteam/proc_data_new", "/workspace/proc_data_new")
OUT = "/workspace/oov_distill/out/ref_pool_v2"
MIN_POST_VAD = 8.0
PER_SPEAKER = 2
RAW_MIN = 9.0          # below this, post-VAD can never reach 8 s
RAW_MAX = 40.0

def main():
    os.makedirs(OUT, exist_ok=True)
    by_spk = collections.defaultdict(list)
    for f in glob.glob(f"{SRC}/**/*.jsonl", recursive=True):
        if "rejection" in f or "backup" in f or ".trash" in f: continue
        for l in open(f, errors="ignore"):
            try: r = json.loads(l)
            except Exception: continue
            if str(r.get("language", "")).strip().lower() not in NE: continue
            d = float(r.get("duration") or 0)
            if not (RAW_MIN <= d <= RAW_MAX): continue
            s = r.get("speaker_id")
            if s: by_spk[s].append(r)
    print(f"speakers with a candidate clip >={RAW_MIN}s: {len(by_spk)}", flush=True)

    rng = random.Random(0)
    seen, man, per = set(), [], collections.Counter()
    tried = kept = 0
    for si, s in enumerate(sorted(by_spk)):
        cands = by_spk[s][:]
        rng.shuffle(cands)
        for r in cands:
            if per[s] >= PER_SPEAKER: break
            p = r["audio_filepath"].replace(*REMAP)
            if not os.path.exists(p): continue
            tried += 1
            try:
                y, sr = sf.read(p, dtype="float32")
            except Exception: continue
            if y.ndim > 1: y = y.mean(1)
            yt, _ = server.vad_trim_reference(y, sr)     # the converter's own trim
            if len(yt) / sr < MIN_POST_VAD: continue
            h = hashlib.md5(y.tobytes()).hexdigest()
            if h in seen: continue
            seen.add(h)
            if sr != 22050: y = soxr.resample(y, sr, 22050).astype("float32"); sr = 22050
            m = float(np.abs(y).max())
            if m < 1e-4: continue
            y = y / m * 0.95
            fn = f"{s}_{per[s]}.wav"
            sf.write(os.path.join(OUT, fn), y, sr)
            per[s] += 1; kept += 1
            man.append(dict(file=fn, speaker_id=s, gender=r.get("gender"),
                            age_group=r.get("age_group"),
                            raw_s=round(len(y)/sr, 2), post_vad_s=round(len(yt)/ (sr if sr==22050 else sr), 2)))
        if (si + 1) % 200 == 0:
            print(f"  {si+1}/{len(by_spk)} speakers scanned, {kept} refs kept "
                  f"({len(per)} distinct speakers)", flush=True)
    with open(os.path.join(OUT, "manifest.jsonl"), "w", encoding="utf-8") as fh:
        for m in man: fh.write(json.dumps(m, ensure_ascii=False) + "\n")
    g = collections.Counter(m["gender"] for m in man)
    big = per.most_common(1)[0][1] / max(len(man), 1) if man else 0
    print(f"\nv2 pool: {len(man)} clips / {len(per)} speakers  (tried {tried} candidates)")
    print(f"  biggest-speaker share: {big:.2%}   gender: {dict(g)}")
    print(f"  every clip has >={MIN_POST_VAD}s of speech after the converter's VAD trim")
    print(f"  v1 for contrast: 772 clips / 400 speakers, only 169 usable at this threshold")

if __name__ == "__main__":
    main()
