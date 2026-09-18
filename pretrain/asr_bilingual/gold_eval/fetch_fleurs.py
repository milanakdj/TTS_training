"""Pull FLEURS ne_np test to local 16 kHz wavs + a raw manifest. Gold, human-read.

datasets 5.x routes Audio decoding through torchcodec, which is not installed and
which we are not adding to this venv (the NeMo install is order-sensitive:
cu126 torch + separate numba-cuda). Take the bytes undecoded and let soundfile
do it -- FLEURS ships wav, so there is nothing exotic to decode.
"""
import io, json, os
import numpy as np, soundfile as sf
from datasets import load_dataset, Audio

OUT = "/root/tts/TTS_training/pretrain/asr_bilingual/gold_eval"
ds = load_dataset("google/fleurs", "ne_np", split="test").cast_column("audio", Audio(decode=False))
print("rows:", len(ds), flush=True)

rows, bad = [], 0
for i, r in enumerate(ds):
    a = r["audio"]
    raw = a.get("bytes")
    if raw is None:
        with open(a["path"], "rb") as fh: raw = fh.read()
    x, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if x.ndim > 1: x = x.mean(1)
    if sr != 16000:
        import librosa; x = librosa.resample(x, orig_sr=sr, target_sr=16000); sr = 16000
    p = f"{OUT}/clips/fleurs_{r['id']:06d}_{i:04d}.wav"
    sf.write(p, x, sr)
    rows.append({"audio_filepath": p, "duration": round(len(x)/sr, 3),
                 "text": r["transcription"], "raw_text": r["raw_transcription"],
                 "fleurs_id": r["id"], "gender": r.get("gender"),
                 "source": "fleurs_ne_np_test", "lang": "ne-NP", "label_kind": "human"})
    if i % 200 == 0: print(i, flush=True)

with open(f"{OUT}/fleurs_ne_test.raw.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("WROTE", len(rows), "clips,", round(sum(r["duration"] for r in rows)/3600, 3), "h", flush=True)
