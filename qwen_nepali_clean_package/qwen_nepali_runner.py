import argparse
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from importlib import metadata as importlib_metadata
from pathlib import Path


RUNNER_VERSION = "qwen-nepali-streamlined-v10-hub-upload-2026-06-09"
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def has_devanagari(text):
    return bool(DEVANAGARI_RE.search(text or ""))


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_columns(sample):
    keys = list(sample.keys())
    audio_col = "audio" if "audio" in keys else next((k for k in keys if "audio" in k.lower()), None)
    text_col = next((k for k in ["text", "transcription", "sentence", "transcript"] if k in keys), None)
    speaker_col = next((k for k in ["speaker_id", "speaker", "client_id", "utt_id"] if k in keys), None)

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

    return audio_col, text_col, speaker_col


def read_audio(audio):
    import numpy as np
    import soundfile as sf

    def from_bytes(data):
        raw = bytes(data)
        try:
            array, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            return np.asarray(array, dtype=np.float32), int(sr)
        except Exception:
            import librosa

            with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
                tmp.write(raw)
                tmp.flush()
                array, sr = librosa.load(tmp.name, sr=None, mono=False)
            return np.asarray(array, dtype=np.float32), int(sr)

    if isinstance(audio, (bytes, bytearray, memoryview)):
        return from_bytes(audio)

    if isinstance(audio, dict):
        if "audio" in audio and audio["audio"] is not audio:
            return read_audio(audio["audio"])
        if "array" in audio and audio["array"] is not None:
            sr = int(audio.get("sampling_rate") or 0)
            return np.asarray(audio["array"], dtype=np.float32), sr
        if audio.get("bytes") is not None:
            return from_bytes(audio["bytes"])
        if audio.get("path"):
            audio_path = Path(str(audio["path"]))
            if not audio_path.exists():
                raise ValueError(f"Audio path does not exist locally: {audio['path']}")
            array, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            return np.asarray(array, dtype=np.float32), int(sr)

    if isinstance(audio, str) and audio:
        audio_path = Path(audio)
        if not audio_path.exists():
            raise ValueError(f"Audio path does not exist locally: {audio}")
        array, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        return np.asarray(array, dtype=np.float32), int(sr)

    if isinstance(audio, (list, tuple, np.ndarray)):
        return np.asarray(audio, dtype=np.float32), 0

    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        return np.asarray(data, dtype=np.float32), int(samples.sample_rate)

    raise ValueError("Audio column is not decoded or readable.")


def normalize_audio(audio, target_sr):
    import librosa
    import numpy as np

    array, sr = read_audio(audio)
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 2:
        if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            array = array.mean(axis=0)
        else:
            array = array.mean(axis=1)

    if sr <= 0:
        sr = target_sr

    if sr != target_sr:
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)

    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        array = array / peak

    return array.astype(np.float32)


def iter_dataset(repo, split, seed, streaming):
    from datasets import Audio, load_dataset

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ds = load_dataset(repo, split=split, token=token, streaming=streaming)

    try:
        features = getattr(ds, "features", None)
        if features and "audio" in features:
            ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print(f"Audio cast skipped: {exc}")

    if streaming:
        return ds.shuffle(seed=seed, buffer_size=1000)

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    return [ds[idx] for idx in indices]


def close_dataset_iter(dataset_iter):
    candidates = [
        dataset_iter,
        getattr(dataset_iter, "_ex_iterable", None),
        getattr(dataset_iter, "ex_iterable", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        close = getattr(candidate, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def duration_allowed(duration, min_sec, max_sec):
    if min_sec is not None and duration < min_sec:
        return False
    if max_sec is not None and max_sec > 0 and duration > max_sec:
        return False
    return True


def choose_reference(records, single_ref_audio, global_ref_audio, preferred_min_sec, preferred_max_sec):
    if not records:
        raise RuntimeError("No valid records collected. Check dataset access and filters.")

    if global_ref_audio:
        ref = str(Path(global_ref_audio).resolve())
        for record in records:
            record["ref_audio"] = ref
        return

    def best_ref(items):
        preferred = [
            r
            for r in items
            if duration_allowed(r["duration"], preferred_min_sec, preferred_max_sec)
        ]
        candidates = preferred or items
        return min(candidates, key=lambda r: abs(r["duration"] - 5.0))["audio"]

    if single_ref_audio:
        ref = best_ref(records)
        for record in records:
            record["ref_audio"] = ref
        return

    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("speaker_id") or record["id"]].append(record)

    refs = {speaker: best_ref(items) for speaker, items in grouped.items()}
    for record in records:
        record["ref_audio"] = refs[record.get("speaker_id") or record["id"]]


def cmd_inspect(args):
    ds_iter = iter_dataset(args.dataset, args.split, args.seed, streaming=args.streaming)
    audio_col = text_col = speaker_col = None
    speaker_counts = Counter()
    speaker_seconds = defaultdict(float)
    total = valid = no_speaker = 0

    for sample in ds_iter:
        total += 1
        if total > args.max_rows:
            break

        if audio_col is None:
            audio_col, text_col, speaker_col = detect_columns(sample)
            print(f"audio_col={audio_col} text_col={text_col} speaker_col={speaker_col}")

        text = str(sample.get(text_col, "")).strip()
        if not text or (not args.allow_non_devanagari and not has_devanagari(text)):
            continue

        try:
            audio = normalize_audio(sample[audio_col], args.target_sr)
            duration = len(audio) / args.target_sr
        except Exception:
            continue

        if not duration_allowed(duration, args.min_sec, args.max_sec):
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
        print(f"{speaker}\trows={count}\tminutes={speaker_seconds[speaker] / 60.0:.1f}")


def cmd_prepare(args):
    import soundfile as sf

    output_dir = Path(args.output_dir).resolve()
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    report = {
        "datasets": {},
        "target_sr": args.target_sr,
        "min_sec": args.min_sec,
        "max_sec": args.max_sec,
        "note": "Duration limits are configurable. Preferred reference length is not a hard dataset filter.",
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

        audio_col = text_col = speaker_col = None
        dataset_iter = iter_dataset(repo, args.split, args.seed, streaming=args.streaming)

        try:
            for sample_index, sample in enumerate(dataset_iter):
                if counts["accepted"] >= args.max_samples_per_dataset:
                    break

                try:
                    if audio_col is None:
                        audio_col, text_col, speaker_col = detect_columns(sample)
                        print(f"  audio_col={audio_col} text_col={text_col} speaker_col={speaker_col}")

                    text = str(sample.get(text_col, "")).strip()
                    if not text or len(text) < 2:
                        counts["skipped_text"] += 1
                        continue
                    if not args.allow_non_devanagari and not has_devanagari(text):
                        counts["skipped_text"] += 1
                        continue

                    speaker_id = sample.get(speaker_col) if speaker_col else ""
                    if args.speaker_id and str(speaker_id) != args.speaker_id:
                        counts["skipped_speaker"] += 1
                        continue

                    array = normalize_audio(sample[audio_col], args.target_sr)
                    duration = len(array) / args.target_sr
                    if not duration_allowed(duration, args.min_sec, args.max_sec):
                        counts["skipped_duration"] += 1
                        continue

                    record_id = f"{repo_key}_{sample_index:07d}"
                    wav_path = repo_wav_dir / f"{record_id}.wav"
                    sf.write(wav_path, array, args.target_sr, subtype="PCM_16")

                    all_records.append(
                        {
                            "id": record_id,
                            "source_dataset": repo,
                            "audio": str(wav_path),
                            "ref_audio": "",
                            "text": text,
                            "speaker_id": str(speaker_id) if speaker_id is not None else "",
                            "duration": round(duration, 3),
                            "text_description": str(sample.get("text_description", "")).strip(),
                            "phonemes": str(sample.get("phonemes", "")).strip(),
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
        finally:
            close_dataset_iter(dataset_iter)

        report["datasets"][repo] = counts
        print(f"  accepted={counts['accepted']} skipped={counts}")

    choose_reference(
        all_records,
        args.single_ref_audio,
        args.global_ref_audio,
        args.preferred_ref_min_sec,
        args.preferred_ref_max_sec,
    )

    raw_jsonl = output_dir / "train_raw.jsonl"
    with raw_jsonl.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report["total_records"] = len(all_records)
    write_json(output_dir / "dataset_report.json", report)

    print("\nDone")
    print(f"Rows: {len(all_records)}")
    print(f"Raw JSONL: {raw_jsonl}")
    print(f"Report: {output_dir / 'dataset_report.json'}")


def cmd_codec_check(args):
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl) if args.output_jsonl else input_path.with_name(input_path.stem + "_filtered.jsonl")
    rows = []
    timestep_counts = Counter()
    channel_counts = Counter()
    invalid = 0
    transposed = 0

    def normalize_code_shape(codes):
        if (
            not isinstance(codes, list)
            or not codes
            or not isinstance(codes[0], list)
            or not codes[0]
        ):
            return None

        if all(isinstance(step, list) and len(step) == 16 for step in codes):
            return codes

        if len(codes) == 16 and all(isinstance(channel, list) for channel in codes):
            lengths = {len(channel) for channel in codes}
            if len(lengths) == 1 and next(iter(lengths)) > 0:
                return [list(step) for step in zip(*codes)]

        return None

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            original_codes = row.get("audio_codes")
            codes = normalize_code_shape(original_codes)
            if codes is None:
                invalid += 1
                continue
            if codes is not original_codes:
                transposed += 1
                row["audio_codes"] = codes
            timestep_counts[len(codes)] += 1
            channel_counts[len(codes[0])] += 1
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid codec rows found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = list(timestep_counts.keys())
    print(f"Input rows: {len(rows) + invalid}")
    print(f"Valid rows: {len(rows)}")
    print(f"Invalid rows: {invalid}")
    print(f"Transposed rows: {transposed}")
    print(f"Codec channels distribution: {dict(channel_counts)}")
    print(f"Time steps min/max: {min(lengths)} / {max(lengths)}")
    print("Variable time length is expected; Qwen training pads batches dynamically.")
    print(f"Filtered JSONL: {output_path}")


def run(cmd, cwd=None):
    print("\n$ " + " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def qwen_paths(args, config):
    qwen_repo = Path(args.qwen_repo)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    return {
        "qwen_repo": qwen_repo,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "raw_jsonl": data_dir / "train_raw.jsonl",
        "coded_jsonl": data_dir / "train_with_codes.jsonl",
        "filtered_jsonl": data_dir / "train_with_codes_filtered.jsonl",
        "tokenizer": args.tokenizer_model or config["tokenizer_model_path"],
        "base_model": args.base_model or config["init_model_path"],
        "model_cache_dir": Path(args.model_cache_dir),
    }


def require_qwen_repo(paths):
    repo = paths["qwen_repo"]
    missing = []
    if not repo.exists():
        missing.append(str(repo))
    for rel_path in ["finetuning/prepare_data.py", "finetuning/sft_12hz.py"]:
        path = repo / rel_path
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise RuntimeError(
            "Qwen3-TTS repo path is not valid. "
            f"Received --qwen-repo {repo}. "
            "Set --qwen-repo to the actual Qwen3-TTS folder. "
            f"Missing: {', '.join(missing)}"
        )


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def resolve_base_model_path(paths):
    base_model = str(paths["base_model"])
    base_path = Path(base_model)
    if base_path.exists():
        return str(base_path)

    from huggingface_hub import snapshot_download

    target = paths["model_cache_dir"] / safe_name(base_model)
    if not (target / "config.json").exists():
        print(f"Downloading base model to: {target}")
        snapshot_download(repo_id=base_model, local_dir=str(target))
    else:
        print(f"Using cached base model: {target}")
    return str(target)


def qwen_train_script(paths, attn):
    source = paths["qwen_repo"] / "finetuning" / "sft_12hz.py"
    if attn == "flash_attention_2":
        try:
            importlib_metadata.version("flash_attn")
            return source
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "flash_attn is not installed. Use --attn sdpa, or install flash-attn first."
            ) from exc

    patched = paths["qwen_repo"] / "finetuning" / f"sft_12hz_runner_{safe_name(attn)}.py"
    text = source.read_text(encoding="utf-8")
    if 'attn_implementation="flash_attention_2"' not in text:
        raise RuntimeError("Could not patch Qwen train script attention implementation.")
    text = text.replace('attn_implementation="flash_attention_2"', f'attn_implementation="{attn}"')

    tensorboard_accelerator = (
        'Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")'
    )
    plain_accelerator = 'Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16")'
    if tensorboard_accelerator in text:
        text = text.replace(tensorboard_accelerator, plain_accelerator)
    text = re.sub(r',\s*log_with=["\']tensorboard["\']', "", text)
    text = re.sub(r'log_with=["\']tensorboard["\']\s*,\s*', "", text)
    text = re.sub(r'log_with=["\']tensorboard["\']', "", text)

    patched.write_text(text, encoding="utf-8")
    print(f"Training attention: {attn}")
    print(f"Training script: {patched}")
    return patched


def cmd_preflight(args):
    config = load_config(args.config)
    paths = qwen_paths(args, config)

    print(f"Python: {sys.version.split()[0]}")
    print(f"CWD: {Path.cwd()}")
    print(f"Qwen repo: {paths['qwen_repo']}")
    require_qwen_repo(paths)
    print("Qwen repo check: OK")
    print(f"Base model: {paths['base_model']}")
    print(f"Tokenizer model: {paths['tokenizer']}")
    print(f"Model cache: {paths['model_cache_dir']}")

    sox_path = shutil.which("sox")
    if not sox_path:
        raise RuntimeError(
            "System SoX binary is missing. Install it first: "
            "sudo apt-get update && sudo apt-get install -y sox libsox-fmt-all"
        )
    print(f"SoX: {sox_path}")

    pinned_versions = {
        "qwen-tts": "0.1.1",
        "transformers": "4.57.3",
        "accelerate": "1.12.0",
    }
    for dist_name, expected_version in pinned_versions.items():
        actual_version = importlib_metadata.version(dist_name)
        if actual_version != expected_version:
            raise RuntimeError(
                f"{dist_name} version mismatch: expected {expected_version}, got {actual_version}."
            )
        print(f"{dist_name}: {actual_version}")

    protobuf_version = importlib_metadata.version("protobuf")
    protobuf_major = int(protobuf_version.split(".", 1)[0])
    if protobuf_major >= 7:
        raise RuntimeError(f"protobuf must be <7 for this environment, got {protobuf_version}.")
    print(f"protobuf: {protobuf_version}")

    required_packages = [
        "datasets",
        "huggingface_hub",
        "soundfile",
        "librosa",
        "numpy",
        "torch",
        "qwen_tts",
    ]
    for package in required_packages:
        __import__(package)
        print(f"Package {package}: OK")

    import torch

    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if not cuda_ok:
        raise RuntimeError("CUDA is not available in torch. Check GPU driver, CUDA, and torch install.")

    device_index = 0
    if args.device.startswith("cuda:"):
        try:
            device_index = int(args.device.split(":", 1)[1])
        except ValueError:
            device_index = 0

    props = torch.cuda.get_device_properties(device_index)
    total_gb = props.total_memory / 1024**3
    print(f"GPU device: {torch.cuda.get_device_name(device_index)}")
    print(f"GPU memory GB: {total_gb:.2f}")
    if total_gb < args.min_vram_gb:
        raise RuntimeError(
            f"GPU memory is {total_gb:.2f} GB, below requested minimum {args.min_vram_gb:.1f} GB."
        )

    print("PREFLIGHT PASSED")


def cmd_tokenize(args):
    config = load_config(args.config)
    paths = qwen_paths(args, config)
    require_qwen_repo(paths)
    run(
        [
            sys.executable,
            paths["qwen_repo"] / "finetuning" / "prepare_data.py",
            "--device",
            args.device,
            "--tokenizer_model_path",
            paths["tokenizer"],
            "--input_jsonl",
            paths["raw_jsonl"],
            "--output_jsonl",
            paths["coded_jsonl"],
        ]
    )
    cmd_codec_check(
        argparse.Namespace(
            input_jsonl=str(paths["coded_jsonl"]),
            output_jsonl=str(paths["filtered_jsonl"]),
        )
    )


def cmd_train(args):
    config = load_config(args.config)
    paths = qwen_paths(args, config)
    require_qwen_repo(paths)
    if getattr(args, "push_to_hub", False) and not getattr(args, "_hub_repo_checked", False):
        ensure_hub_repo(args)
        args._hub_repo_checked = True
    train_script = qwen_train_script(paths, args.attn)
    base_model_path = resolve_base_model_path(paths)
    run(
        [
            sys.executable,
            train_script,
            "--init_model_path",
            base_model_path,
            "--output_model_path",
            paths["output_dir"],
            "--train_jsonl",
            paths["filtered_jsonl"],
            "--batch_size",
            args.batch_size,
            "--lr",
            args.lr,
            "--num_epochs",
            args.epochs,
            "--speaker_name",
            args.speaker_name,
        ]
    )
    if getattr(args, "push_to_hub", False) and not getattr(args, "_defer_hub_upload", False):
        upload_to_hub(paths["output_dir"], args)


def load_sentences(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def cmd_generate(args):
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    output_dir = Path(args.sample_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        dtype=dtype,
        attn_implementation=args.attn,
    )

    for index, sentence in enumerate(load_sentences(args.sentences), start=1):
        wavs, sr = model.generate_custom_voice(
            text=sentence,
            speaker=args.speaker_name,
            max_new_tokens=args.max_new_tokens,
        )
        out_path = output_dir / f"qwen_nepali_{index:02d}.wav"
        sf.write(out_path, wavs[0], sr)
        print(f"{out_path}\t{sentence}")


def latest_checkpoint(output_dir):
    output = Path(output_dir)
    checkpoints = sorted(output.glob("checkpoint-epoch-*"))
    if not checkpoints:
        return output
    return checkpoints[-1]


def upload_to_hub(folder_path, args):
    from huggingface_hub import HfApi

    repo_id, token = hub_settings(args)

    folder = Path(folder_path)
    if not folder.exists():
        raise RuntimeError(f"Hub upload folder does not exist: {folder}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=getattr(args, "hub_private", False),
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder),
        path_in_repo=".",
        commit_message=getattr(args, "hub_commit_message", "Upload Qwen Nepali checkpoint"),
    )
    print(f"Pushed checkpoint folder to Hugging Face Hub: {repo_id}")


def hub_settings(args):
    repo_id = getattr(args, "hub_repo_id", "")
    if not repo_id:
        raise RuntimeError("Set --hub-repo-id when using --push-to-hub.")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "HF token missing. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in the environment."
        )

    return repo_id, token


def ensure_hub_repo(args):
    from huggingface_hub import HfApi

    repo_id, token = hub_settings(args)
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=getattr(args, "hub_private", False),
        exist_ok=True,
    )
    print(f"Hub repo ready: {repo_id}")


def cmd_all(args):
    config = load_config(args.config)
    paths = qwen_paths(args, config)
    require_qwen_repo(paths)
    if getattr(args, "push_to_hub", False):
        ensure_hub_repo(args)
        args._hub_repo_checked = True
    datasets = args.datasets or [config["primary_dataset"]]
    cmd_prepare(
        argparse.Namespace(
            datasets=datasets,
            split=args.split,
            output_dir=str(paths["data_dir"]),
            max_samples_per_dataset=args.max_samples,
            target_sr=args.target_sr,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
            preferred_ref_min_sec=args.preferred_ref_min_sec,
            preferred_ref_max_sec=args.preferred_ref_max_sec,
            seed=args.seed,
            streaming=args.streaming,
            allow_non_devanagari=args.allow_non_devanagari,
            speaker_id=args.speaker_id,
            single_ref_audio=True,
            global_ref_audio=args.global_ref_audio,
        )
    )
    cmd_tokenize(args)
    args._defer_hub_upload = True
    cmd_train(args)
    args._defer_hub_upload = False
    checkpoint = latest_checkpoint(paths["output_dir"])
    cmd_generate(
        argparse.Namespace(
            checkpoint=str(checkpoint),
            speaker_name=args.speaker_name,
            sentences=args.sentences,
            sample_output_dir=args.sample_output_dir,
            device=args.device,
            dtype=args.dtype,
            attn=args.attn,
            max_new_tokens=args.max_new_tokens,
        )
    )
    if getattr(args, "push_to_hub", False):
        upload_to_hub(paths["output_dir"], args)


def add_common_data_args(parser):
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--min-sec", type=float, default=1.0)
    parser.add_argument(
        "--max-sec",
        type=float,
        default=20.0,
        help="Set 0 to disable max duration filtering.",
    )
    parser.add_argument("--preferred-ref-min-sec", type=float, default=3.0)
    parser.add_argument("--preferred-ref-max-sec", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(streaming=True)
    parser.add_argument("--streaming", dest="streaming", action="store_true", help="Use Hugging Face streaming mode. This is the default.")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="Disable streaming and cache-load the dataset normally.")
    parser.add_argument("--allow-non-devanagari", action="store_true")
    parser.add_argument("--speaker-id", default="")


def add_qwen_args(parser):
    parser.add_argument("--config", default="qwen_1_7b_config.json")
    parser.add_argument("--qwen-repo", default="Qwen3-TTS")
    parser.add_argument("--data-dir", default="data/tagged_300")
    parser.add_argument("--output-dir", default="outputs/qwen_1_7b_nepali_smoke")
    parser.add_argument("--model-cache-dir", default="models")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--batch-size", default="2")
    parser.add_argument("--lr", default="2e-6")
    parser.add_argument("--epochs", default="3")
    parser.add_argument("--speaker-name", default="nepali_speaker")


def add_hub_args(parser):
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", default="")
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--hub-commit-message", default="Upload Qwen Nepali checkpoint")


def build_parser():
    parser = argparse.ArgumentParser(description="Single runner for Qwen3-TTS Nepali data prep, codec check, training, and samples.")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect speaker counts in a dataset.")
    inspect_p.add_argument("--dataset", required=True)
    inspect_p.add_argument("--max-rows", type=int, default=50000)
    inspect_p.add_argument("--top", type=int, default=20)
    add_common_data_args(inspect_p)
    inspect_p.set_defaults(func=cmd_inspect)

    preflight_p = sub.add_parser("preflight", help="Check Qwen repo, Python packages, CUDA, and GPU memory.")
    add_qwen_args(preflight_p)
    preflight_p.add_argument("--min-vram-gb", type=float, default=16.0)
    preflight_p.set_defaults(func=cmd_preflight)

    prepare_p = sub.add_parser("prepare", help="Create 24 kHz WAV files and train_raw.jsonl.")
    prepare_p.add_argument("--datasets", nargs="+", default=["Titung/nepali-tts-tagged-combined"])
    prepare_p.add_argument("--output-dir", default="data/tagged_300")
    prepare_p.add_argument("--max-samples-per-dataset", type=int, default=300)
    prepare_p.add_argument("--single-ref-audio", action="store_true")
    prepare_p.add_argument("--global-ref-audio", default="")
    add_common_data_args(prepare_p)
    prepare_p.set_defaults(func=cmd_prepare)

    codec_p = sub.add_parser("codec-check", help="Check/filter Qwen codec token JSONL.")
    codec_p.add_argument("--input-jsonl", required=True)
    codec_p.add_argument("--output-jsonl", default="")
    codec_p.set_defaults(func=cmd_codec_check)

    tokenize_p = sub.add_parser("tokenize", help="Run Qwen prepare_data.py and codec check.")
    add_qwen_args(tokenize_p)
    tokenize_p.set_defaults(func=cmd_tokenize)

    train_p = sub.add_parser("train", help="Run Qwen sft_12hz.py.")
    add_qwen_args(train_p)
    add_hub_args(train_p)
    train_p.add_argument("--attn", default="sdpa")
    train_p.set_defaults(func=cmd_train)

    gen_p = sub.add_parser("generate", help="Generate samples from a trained checkpoint.")
    gen_p.add_argument("--checkpoint", required=True)
    gen_p.add_argument("--speaker-name", default="nepali_speaker")
    gen_p.add_argument("--sentences", default="test_sentences_nepali.txt")
    gen_p.add_argument("--sample-output-dir", default="outputs/qwen_test_samples")
    gen_p.add_argument("--device", default="cuda:0")
    gen_p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    gen_p.add_argument("--attn", default="sdpa")
    gen_p.add_argument("--max-new-tokens", type=int, default=1000)
    gen_p.set_defaults(func=cmd_generate)

    all_p = sub.add_parser("all", help="Prepare data, tokenize, train, and generate samples.")
    all_p.add_argument("--datasets", nargs="+", default=[])
    all_p.add_argument("--max-samples", type=int, default=300)
    all_p.add_argument("--global-ref-audio", default="")
    all_p.add_argument("--sentences", default="test_sentences_nepali.txt")
    all_p.add_argument("--sample-output-dir", default="outputs/qwen_test_samples")
    all_p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    all_p.add_argument("--attn", default="sdpa")
    all_p.add_argument("--max-new-tokens", type=int, default=1000)
    add_common_data_args(all_p)
    add_qwen_args(all_p)
    add_hub_args(all_p)
    all_p.set_defaults(func=cmd_all)

    return parser


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args()
    print(f"Runner: {RUNNER_VERSION}")
    exit_code = 0
    try:
        args.func(args)
    except Exception as exc:
        exit_code = 1
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if os.environ.get("QWEN_RUNNER_NORMAL_EXIT", "").lower() not in {"1", "true", "yes"}:
            os._exit(exit_code)
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
