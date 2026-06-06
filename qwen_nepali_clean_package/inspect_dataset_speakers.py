import argparse
import os
import re
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def has_devanagari(text):
    return bool(DEVANAGARI_RE.search(text or ""))


def detect_columns(sample):
    keys = list(sample.keys())
    audio_col = "audio" if "audio" in keys else next(
        (k for k in keys if "audio" in k.lower()), None
    )
    text_col = next(
        (k for k in ["text", "transcription", "sentence", "transcript"] if k in keys),
        None,
    )
    speaker_col = next(
        (k for k in ["speaker_id", "speaker", "client_id"] if k in keys),
        None,
    )
    return audio_col, text_col, speaker_col


def audio_duration(sample, audio_col):
    if not audio_col:
        return 0.0
    audio = sample.get(audio_col)
    if not isinstance(audio, dict) or "array" not in audio:
        return 0.0
    array = np.asarray(audio["array"])
    sr = audio.get("sampling_rate") or 0
    if sr <= 0:
        return 0.0
    return float(array.shape[0]) / float(sr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--min-sec", type=float, default=1.0)
    parser.add_argument("--max-sec", type=float, default=12.0)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ds = load_dataset(args.dataset, split=args.split, token=token, streaming=True)

    speaker_counts = Counter()
    speaker_seconds = defaultdict(float)
    total = 0
    valid = 0
    no_speaker = 0
    audio_col = text_col = speaker_col = None

    for sample in ds:
        total += 1
        if total > args.max_rows:
            break

        if audio_col is None:
            audio_col, text_col, speaker_col = detect_columns(sample)
            print(f"audio_col={audio_col} text_col={text_col} speaker_col={speaker_col}")

        text = str(sample.get(text_col, "")).strip() if text_col else ""
        if not text or not has_devanagari(text):
            continue

        duration = audio_duration(sample, audio_col)
        if duration < args.min_sec or duration > args.max_sec:
            continue

        speaker = str(sample.get(speaker_col, "")).strip() if speaker_col else ""
        if not speaker:
            no_speaker += 1
            continue

        speaker_counts[speaker] += 1
        speaker_seconds[speaker] += duration
        valid += 1

    print(f"Scanned rows: {min(total, args.max_rows)}")
    print(f"Valid rows with speaker: {valid}")
    print(f"Valid rows without speaker: {no_speaker}")
    print("\nTop speakers:")
    for speaker, count in speaker_counts.most_common(args.top):
        minutes = speaker_seconds[speaker] / 60.0
        print(f"{speaker}\trows={count}\tminutes={minutes:.1f}")


if __name__ == "__main__":
    main()
