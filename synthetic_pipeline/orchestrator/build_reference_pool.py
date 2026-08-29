"""
Build a pool of real Nepali reference-voice clips for the VC service to rotate
through, so the 500k-row synthesis run gets real speaker-timbre diversity
instead of reusing one or two voices.

Pulls several shards of Titung/nepali-tts-tagged-combined, filters by its
built-in quality columns (snr, pesq, 3-12s duration -- same range the repo's
own Qwen voice-cloning runbook prefers for reference audio), and writes a pool
of WAV files + an index parquet.
"""
import io
import sys

import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download

N_SHARDS = 4
POOL_SIZE = 400
OUT_DIR = "/root/tts/TTS_training/synthetic_pipeline/manifests/reference_pool"
INDEX_PATH = "/root/tts/TTS_training/synthetic_pipeline/manifests/reference_pool_index.parquet"

import os
os.makedirs(OUT_DIR, exist_ok=True)


def audio_dict_to_wav_bytes(audio_val):
    raw = audio_val["bytes"]
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="int16")
    except Exception:
        import librosa
        data, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main():
    frames = []
    for shard in range(N_SHARDS):
        fname = f"data/train-{shard:05d}-of-00015.parquet"
        print(f"downloading {fname}...", file=sys.stderr)
        p = hf_hub_download("Titung/nepali-tts-tagged-combined", fname, repo_type="dataset")
        df = pd.read_parquet(p)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    print(f"total rows across {N_SHARDS} shards: {len(all_df)}", file=sys.stderr)

    good = all_df[
        (all_df["speech_duration"] >= 3.0)
        & (all_df["speech_duration"] <= 12.0)
        & (all_df["snr"] > 25)
        & (all_df["pesq"] > 3.0)
    ].copy()
    print(f"rows passing quality filter: {len(good)}", file=sys.stderr)

    good = good.sample(n=min(POOL_SIZE, len(good)), random_state=42).reset_index(drop=True)

    index_rows = []
    for i, row in good.iterrows():
        wav_bytes = audio_dict_to_wav_bytes(row["audio"])
        out_path = os.path.join(OUT_DIR, f"ref_{i:04d}.wav")
        with open(out_path, "wb") as f:
            f.write(wav_bytes)
        index_rows.append({"ref_id": i, "path": out_path, "source_id": row["id"], "duration": row["speech_duration"]})

    idx_df = pd.DataFrame(index_rows)
    idx_df.to_parquet(INDEX_PATH)
    print(f"wrote {len(idx_df)} reference clips to {OUT_DIR}, index at {INDEX_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
