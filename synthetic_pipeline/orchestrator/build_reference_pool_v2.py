"""
Reference-voice pool v2 -- replaces the Titung-based pool, which turned out to
be essentially one voice with synthetic noise/reverb augmentation layered on
(pitch std ~4Hz across 2668 rows -- confirmed via librosa pitch analysis).

Real, genuinely multi-speaker sources used instead (verified the same way --
pitch std 38-59Hz, multi-modal, spanning male and female ranges):
  - lilgoose777/nepali-tts-massive-combined (231 rows, real recordings)
  - google/fleurs ne_np via himalaya-ai/nep-voice-tts-compilation's finetune
    mirror (real academic multi-speaker read-speech corpus, cc-by-4.0)
"""
import io
import os
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download

OUT_DIR = "/root/tts/TTS_training/synthetic_pipeline/manifests/reference_pool_v2"
INDEX_PATH = "/root/tts/TTS_training/synthetic_pipeline/manifests/reference_pool_v2_index.parquet"
FLEURS_TRAIN_SHARDS = 10  # of 20 available; plenty of volume for our pool size
TARGET_POOL_SIZE = 600

os.makedirs(OUT_DIR, exist_ok=True)


def audio_bytes_from_field(audio_val):
    raw = audio_val["bytes"]
    data, sr = sf.read(io.BytesIO(raw), dtype="int16")
    return data, sr


def duration_ok(raw_bytes, lo=2.5, hi=15.0):
    try:
        info = sf.info(io.BytesIO(raw_bytes))
        return lo <= info.frames / info.samplerate <= hi
    except Exception:
        return False


def main():
    rows = []

    print("[1/2] lilgoose777/nepali-tts-massive-combined (real recordings)...", file=sys.stderr)
    p1 = hf_hub_download("lilgoose777/nepali-tts-massive-combined", "data/train-00000-of-00001.parquet", repo_type="dataset")
    p2 = hf_hub_download("lilgoose777/nepali-tts-massive-combined", "data/test-00000-of-00001.parquet", repo_type="dataset")
    lg_df = pd.concat([pd.read_parquet(p1), pd.read_parquet(p2)], ignore_index=True)
    for _, row in lg_df.iterrows():
        raw = row["audio"]["bytes"]
        if duration_ok(raw):
            rows.append({"raw_bytes": raw, "source": "lilgoose777"})
    print(f"  kept {len(rows)} from lilgoose777", file=sys.stderr)

    print(f"[2/2] google/fleurs ne_np via himalaya-ai/nep-voice-tts-compilation ({FLEURS_TRAIN_SHARDS} shards)...", file=sys.stderr)
    fleurs_count = 0
    for i in range(FLEURS_TRAIN_SHARDS):
        fname = f"data/finetune/google_fleurs_ne_np/ne_np/train/google_fleurs_ne_np-ne_np-train-{i:05d}.parquet"
        p = hf_hub_download("himalaya-ai/nep-voice-tts-compilation", fname, repo_type="dataset")
        df = pd.read_parquet(p, columns=["audio"])
        for _, row in df.iterrows():
            raw = row["audio"]["bytes"]
            if duration_ok(raw):
                rows.append({"raw_bytes": raw, "source": "fleurs_ne_np"})
                fleurs_count += 1
    print(f"  kept {fleurs_count} from fleurs", file=sys.stderr)

    print(f"total candidates: {len(rows)}", file=sys.stderr)
    if len(rows) > TARGET_POOL_SIZE:
        idx = np.random.RandomState(42).choice(len(rows), TARGET_POOL_SIZE, replace=False)
        rows = [rows[i] for i in idx]

    index_rows = []
    pitches = []
    for i, r in enumerate(rows):
        data, sr = audio_bytes_from_field({"bytes": r["raw_bytes"]})
        out_path = os.path.join(OUT_DIR, f"ref_{i:04d}.wav")
        sf.write(out_path, data, sr, subtype="PCM_16")
        index_rows.append({"ref_id": i, "path": out_path, "source": r["source"]})
        try:
            f, _ = librosa.load(io.BytesIO(r["raw_bytes"]), sr=None, mono=True)
            f0, _, _ = librosa.pyin(f, sr=sr, fmin=60, fmax=400, frame_length=2048)
            f0v = f0[~np.isnan(f0)]
            if len(f0v) > 5:
                pitches.append(np.median(f0v))
        except Exception:
            pass

    idx_df = pd.DataFrame(index_rows)
    idx_df.to_parquet(INDEX_PATH)
    print(f"wrote {len(idx_df)} reference clips to {OUT_DIR}", file=sys.stderr)
    print("source breakdown:", idx_df["source"].value_counts().to_dict(), file=sys.stderr)
    pitches = np.array(pitches)
    print(f"pitch diversity check: n={len(pitches)} mean={pitches.mean():.1f} std={pitches.std():.1f} min={pitches.min():.1f} max={pitches.max():.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
