import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
DEFAULT_DATASETS = ["Titung/nepali-tts-tagged-combined"]


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))


def normalize_audio(array, sampling_rate, target_sr):
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 2:
        array = array.mean(axis=1)

    if sampling_rate != target_sr:
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=target_sr)

    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak

    return array.astype(np.float32)


def detect_columns(sample):
    keys = list(sample.keys())
    audio_col = "audio" if "audio" in keys else next(
        (k for k in keys if "audio" in k.lower()), None
    )
    text_col = next(
        (k for k in ["text", "transcription", "sentence", "transcript"] if k in keys),
        None,
    )

    if text_col is None:
        text_col = next(
            (
                k
                for k in keys
                if k != audio_col and isinstance(sample.get(k), str) and sample.get(k).strip()
            ),
            None,
        )

    if audio_col is None or text_col is None:
        raise ValueError(f"Could not detect audio/text columns from keys: {keys}")

    return audio_col, text_col


def get_audio_dict(sample, audio_col):
    audio = sample[audio_col]
    if not isinstance(audio, dict) or "array" not in audio:
        raise ValueError("Audio column is not decoded. Expected a datasets Audio dict.")
    return audio


def choose_best_reference(records):
    candidates = [r for r in records if 3.0 <= r["duration"] <= 8.0]
    if not candidates:
        candidates = records
    return min(candidates, key=lambda r: abs(r["duration"] - 5.0))["audio"]


def choose_reference(records, single_ref_audio=False, global_ref_audio=""):
    if not records:
        raise RuntimeError("No valid records were collected. Check dataset access and filters.")

    if global_ref_audio:
        for record in records:
            record["ref_audio"] = str(Path(global_ref_audio).resolve())
        return

    if single_ref_audio:
        ref_audio = choose_best_reference(records)
        for record in records:
            record["ref_audio"] = ref_audio
        return

    refs = {}
    grouped = defaultdict(list)

    for record in records:
        speaker = record.get("speaker_id") or record["id"]
        grouped[speaker].append(record)

    for speaker, items in grouped.items():
        candidates = [r for r in items if 3.0 <= r["duration"] <= 8.0]
        if not candidates:
            candidates = items
        refs[speaker] = min(candidates, key=lambda r: abs(r["duration"] - 5.0))["audio"]

    for record in records:
        speaker = record.get("speaker_id") or record["id"]
        record["ref_audio"] = refs[speaker]


def iter_dataset(repo, split, token, seed, streaming):
    ds = load_dataset(repo, split=split, token=token, streaming=streaming)
    if streaming:
        ds = ds.shuffle(seed=seed, buffer_size=1000)
        return ds

    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=True))
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    for idx in indices:
        yield ds[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/qwen_nepali_small")
    parser.add_argument("--max-samples-per-dataset", type=int, default=150)
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--min-sec", type=float, default=1.0)
    parser.add_argument("--max-sec", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--allow-non-devanagari", action="store_true")
    parser.add_argument(
        "--speaker-id",
        default="",
        help="Keep only one speaker_id/speaker value. Recommended for Qwen single-speaker fine-tuning.",
    )
    parser.add_argument(
        "--single-ref-audio",
        action="store_true",
        help="Use one clean reference audio for every row. Recommended for Qwen single-speaker fine-tuning.",
    )
    parser.add_argument(
        "--global-ref-audio",
        default="",
        help="Use this explicit reference WAV path for every row.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    output_dir = Path(args.output_dir).resolve()
    wav_dir = output_dir / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    report = {
        "datasets": {},
        "target_sr": args.target_sr,
        "min_sec": args.min_sec,
        "max_sec": args.max_sec,
        "speaker_id_filter": args.speaker_id,
        "single_ref_audio": args.single_ref_audio,
        "global_ref_audio": args.global_ref_audio,
    }

    for repo in args.datasets:
        print(f"\nLoading {repo}")
        repo_key = repo.replace("/", "_")
        repo_wav_dir = wav_dir / repo_key
        repo_wav_dir.mkdir(parents=True, exist_ok=True)

        counts = {
            "accepted": 0,
            "skipped_duration": 0,
            "skipped_text": 0,
            "skipped_speaker": 0,
            "skipped_error": 0,
        }

        dataset_iter = iter_dataset(
            repo=repo,
            split=args.split,
            token=token,
            seed=args.seed,
            streaming=not args.no_streaming,
        )

        audio_col = None
        text_col = None

        for sample_index, sample in enumerate(dataset_iter):
            if counts["accepted"] >= args.max_samples_per_dataset:
                break

            try:
                if audio_col is None or text_col is None:
                    audio_col, text_col = detect_columns(sample)
                    print(f"  audio_col={audio_col} text_col={text_col}")

                text = str(sample.get(text_col, "")).strip()
                if not text or len(text) < 2:
                    counts["skipped_text"] += 1
                    continue
                if not args.allow_non_devanagari and not has_devanagari(text):
                    counts["skipped_text"] += 1
                    continue

                audio = get_audio_dict(sample, audio_col)
                array = normalize_audio(
                    audio["array"], audio["sampling_rate"], args.target_sr
                )
                duration = len(array) / args.target_sr
                if duration < args.min_sec or duration > args.max_sec:
                    counts["skipped_duration"] += 1
                    continue

                speaker_id = sample.get("speaker_id") or sample.get("speaker")
                if args.speaker_id and str(speaker_id) != args.speaker_id:
                    counts["skipped_speaker"] += 1
                    continue

                text_description = str(sample.get("text_description", "")).strip()
                phonemes = str(sample.get("phonemes", "")).strip()
                record_id = f"{repo_key}_{sample_index:07d}"
                wav_path = repo_wav_dir / f"{record_id}.wav"
                sf.write(wav_path, array, args.target_sr, subtype="PCM_16")

                all_records.append(
                    {
                        "id": record_id,
                        "source_dataset": repo,
                        "audio": str(wav_path),
                        "text": text,
                        "speaker_id": str(speaker_id) if speaker_id is not None else "",
                        "duration": round(duration, 3),
                        "text_description": text_description,
                        "phonemes": phonemes,
                        "snr": sample.get("snr"),
                        "pesq": sample.get("pesq"),
                        "noise": sample.get("noise"),
                        "reverberation": sample.get("reverberation"),
                        "speech_monotony": sample.get("speech_monotony"),
                    }
                )
                counts["accepted"] += 1

            except Exception as exc:
                counts["skipped_error"] += 1
                if counts["skipped_error"] <= 3:
                    print(f"  skipped error: {exc}")

        report["datasets"][repo] = counts
        print(f"  accepted={counts['accepted']} skipped={counts}")

    choose_reference(
        all_records,
        single_ref_audio=args.single_ref_audio,
        global_ref_audio=args.global_ref_audio,
    )

    raw_jsonl = output_dir / "train_raw.jsonl"
    with raw_jsonl.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report["total_records"] = len(all_records)
    report_path = output_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone")
    print(f"Rows: {len(all_records)}")
    print(f"Raw JSONL: {raw_jsonl}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
