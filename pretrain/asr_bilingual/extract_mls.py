"""Extract N hours of MLS English from parquet into files + a NeMo manifest.

The parquet already carries `audio_duration`, so nothing has to be probed or
decoded -- we write the embedded bytes straight through. Files are sharded into
1k-file subdirs because /workspace is a network FS and flat mega-directories
stat badly.
"""
import os, io, json, sys, glob
import pyarrow.parquet as pq

OUT = "/root/tts/TTS_training/pretrain/asr_bilingual/data/en_files"
MAN = "/root/tts/TTS_training/pretrain/asr_bilingual/manifests/en_train.jsonl"
TARGET_H = float(os.environ.get("TARGET_H", 300))
os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.dirname(MAN), exist_ok=True)

shards = sorted(glob.glob("/root/tts/TTS_training/pretrain/asr_bilingual/data/mls_eng/data/train-*.parquet"))
tot = 0.0; n = 0
with open(MAN, "w") as mf:
    for sh in shards:
        if tot / 3600 >= TARGET_H: break
        for batch in pq.ParquetFile(sh).iter_batches(batch_size=512):
            for r in batch.to_pylist():
                dur = float(r["audio_duration"])
                txt = (r["transcript"] or "").strip()
                if not txt or dur <= 0.5 or dur > 30: continue
                sub = f"{OUT}/{n//1000:05d}"
                if n % 1000 == 0: os.makedirs(sub, exist_ok=True)
                ext = os.path.splitext(r["audio"]["path"])[1] or ".flac"
                p = f"{sub}/{n:08d}{ext}"
                with open(p, "wb") as f: f.write(r["audio"]["bytes"])
                mf.write(json.dumps({"audio_filepath": p, "duration": dur,
                                     "text": txt, "lang": "en"}) + "\n")
                n += 1; tot += dur
            if tot / 3600 >= TARGET_H: break
        print(f"  {os.path.basename(sh)}: {n:,} utts {tot/3600:.1f} h", flush=True)
print(f"[done] {n:,} utts, {tot/3600:.1f} h -> {MAN}", flush=True)
