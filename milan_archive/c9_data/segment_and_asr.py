"""Segment the c9nepali Edge-TTS shards into Parler-sized clips and re-transcribe.

The source rows are ~39 s of read-aloud news with ARTICLE-level text, so the text
does not align to any sub-span of the audio. We therefore cut on silence and get
per-segment text back from the ASR service rather than trying to split the article.

Writes 16 kHz FLAC segments + one JSONL manifest row per segment.
"""
import argparse, io, json, os, sys, hashlib
import numpy as np, soundfile as sf, requests
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--shards", required=True, help="e.g. 0-49")
ap.add_argument("--out", default="/root/tts/TTS_training/c9_data")
ap.add_argument("--min-s", type=float, default=4.0)
ap.add_argument("--max-s", type=float, default=20.0)
ap.add_argument("--sil-db", type=float, default=-35.0, help="dB below clip peak = silence")
ap.add_argument("--sil-s", type=float, default=0.30)
ap.add_argument("--workers", type=int, default=24)
args = ap.parse_args()

ASR_PORTS = [8003, 8013, 8023, 8033, 8043, 8053, 8063, 8073]
RAW = os.path.join(args.out, "raw")
SEG = os.path.join(args.out, "segments")
a, b = args.shards.split("-")
SHARDS = list(range(int(a), int(b) + 1))
MAN = os.path.join(args.out, "manifests", f"segments_{a}_{b}.jsonl")
os.makedirs(SEG, exist_ok=True); os.makedirs(os.path.dirname(MAN), exist_ok=True)

DEV = lambda s: sum('ऀ' <= c <= 'ॿ' for c in s)


def cut_points(y, sr):
    """Split indices at silences >= sil_s, packing pieces into [min_s, max_s]."""
    fl = int(0.02 * sr)
    n = len(y) // fl
    if n == 0:
        return []
    rms = np.sqrt(np.add.reduceat(y[:n * fl] ** 2, np.arange(0, n * fl, fl)) / fl + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    sil = db < (db.max() + args.sil_db)
    # candidate boundaries = middle of each silence run >= sil_s
    bounds, c0 = [], None
    need = int(args.sil_s / 0.02)
    for i, s in enumerate(sil):
        if s and c0 is None:
            c0 = i
        elif not s and c0 is not None:
            if i - c0 >= need:
                bounds.append((c0 + i) // 2 * fl)
            c0 = None
    if c0 is not None and n - c0 >= need:
        bounds.append((c0 + n) // 2 * fl)
    # pack boundaries into segments within [min_s, max_s]
    segs, start = [], 0
    for bd in bounds + [len(y)]:
        dur = (bd - start) / sr
        if dur < args.min_s:
            continue
        if dur <= args.max_s:
            segs.append((start, bd)); start = bd
        else:  # forced split of an over-long stretch
            k = int(np.ceil(dur / args.max_s))
            step = (bd - start) // k
            for j in range(k):
                s0 = start + j * step
                s1 = bd if j == k - 1 else start + (j + 1) * step
                if (s1 - s0) / sr >= args.min_s:
                    segs.append((s0, s1))
            start = bd
    return [(s, e) for s, e in segs if args.min_s <= (e - s) / sr <= args.max_s]


sess = requests.Session()


def asr(wav_bytes, port):
    r = sess.post(f"http://127.0.0.1:{port}/transcribe",
                  files={"audio": ("seg.wav", wav_bytes, "audio/wav")}, timeout=180)
    r.raise_for_status()
    return r.json()


def feats(y, sr):
    import librosa
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr, frame_length=1024)
    v = f0[~np.isnan(f0)]
    rms = float(np.sqrt((y ** 2).mean()))
    fl = int(0.02 * sr); n = len(y) // fl
    fr = np.sqrt(np.add.reduceat(y[:n * fl] ** 2, np.arange(0, n * fl, fl)) / fl + 1e-12)
    d = 20 * np.log10(fr + 1e-12)
    speech = float((d > d.max() - 35).mean())
    return (float(np.median(v)) if len(v) else 0.0,
            float(np.std(v)) if len(v) else 0.0, rms, speech)


def do_row(job):
    idx, sid, rid, audio_bytes = job
    port = ASR_PORTS[idx % len(ASR_PORTS)]
    out = []
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        for k, (s, e) in enumerate(cut_points(y, sr)):
            seg = y[s:e]
            buf = io.BytesIO(); sf.write(buf, seg, sr, format="WAV", subtype="PCM_16")
            wav = buf.getvalue()
            try:
                j = asr(wav, port)
            except Exception as ex:
                out.append({"error": f"asr:{ex}", "row_id": rid}); continue
            text = (j.get("text") or "").strip()
            if not text:
                continue
            dv = DEV(text) / max(1, len(text.replace(" ", "")))
            f0m, f0s, rms, speech = feats(seg, sr)
            name = f"{rid}_{k:03d}"
            sub = os.path.join(SEG, f"{int(sid):04d}")
            os.makedirs(sub, exist_ok=True)
            path = os.path.join(sub, name + ".flac")
            sf.write(path, seg, sr, format="FLAC")
            out.append({
                "seg_id": name, "shard": int(sid), "row_id": rid,
                "audio_path": path, "sr": sr, "duration_s": round(len(seg) / sr, 3),
                "text": text, "devanagari_ratio": round(dv, 4),
                "n_chars": len(text), "chars_per_s": round(len(text) / (len(seg) / sr), 2),
                "f0_median": round(f0m, 1), "f0_std": round(f0s, 1),
                "rms": round(rms, 5), "speech_frac": round(speech, 4),
            })
    except Exception as ex:
        out.append({"error": f"row:{ex}", "row_id": rid})
    return out


jobs = []
for sid in SHARDS:
    p = os.path.join(RAW, f"shard_{sid:04d}.parquet")
    if not os.path.exists(p):
        continue
    f = pq.ParquetFile(p)
    for bt in f.iter_batches(batch_size=25):
        for r in bt.to_pylist():
            jobs.append((len(jobs), sid, r["id"], r["audio"]["bytes"]))
print(f"rows to segment: {len(jobs)} from {len(SHARDS)} shards", flush=True)

nseg = nerr = 0
with open(MAN, "w") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
    for i, res in enumerate(ex.map(do_row, jobs)):
        for r in res:
            if "error" in r:
                nerr += 1
            else:
                nseg += 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        if (i + 1) % 200 == 0:
            fh.flush()
            print(f"[{i+1}/{len(jobs)}] segments={nseg} errors={nerr}", flush=True)
print(f"DONE rows={len(jobs)} segments={nseg} errors={nerr} -> {MAN}")
