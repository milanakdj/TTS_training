"""
End-to-end smoke test for the synthetic Nepali data pipeline:
  1. Pull N real (text, audio) rows from an existing HF dataset.
     - text -> fed to TTS services (edge primary, mms/parler for comparison on a subset)
     - audio -> used as real reference clips for the VC service (voice timbre targets)
  2. For each synthesized clip, run voice conversion against a *different* row's real
     reference clip (rotated), producing a re-voiced version.
  3. QC both the raw synthetic clip and the VC'd clip against the known source text via
     the ASR QC service (pass/fail + CER/WER) -- this never uses ASR output as a label,
     only as a filter, per CONTRACT.md.
  4. Write everything to audio_out/ and a manifest CSV/JSONL summarizing results.

Run: .venv/bin/python run_test_batch.py
"""
import io
import json
import os
import sys
import wave

import pandas as pd
import requests
import soundfile as sf
from huggingface_hub import hf_hub_download

PIPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_OUT = os.path.join(PIPE_ROOT, "audio_out")
MANIFEST_PATH = os.path.join(PIPE_ROOT, "manifests", "test_batch.jsonl")

TTS_URL = "http://localhost:8001"
VC_URL = "http://localhost:8002"
QC_URL = "http://localhost:8003"

N_ROWS = 10
COMPARISON_ENGINES_FOR_FIRST_N = 3  # how many rows also get mms + parler, beyond edge
DATASET_ID = "Titung/nepali-tts-tagged-combined"

os.makedirs(AUDIO_OUT, exist_ok=True)
os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)


def save_wav(path, array, samplerate):
    sf.write(path, array, samplerate, subtype="PCM_16")


def load_rows(n):
    # Streaming (HTTP range requests via fsspec) timed out repeatedly against this dataset's
    # host, but a plain hf_hub_download of the shard file is reliable -- download one shard
    # directly and read it locally instead of using datasets(streaming=True).
    shard_path = hf_hub_download(DATASET_ID, "data/train-00000-of-00015.parquet", repo_type="dataset")
    df = pd.read_parquet(shard_path)
    print(f"[load_rows] shard columns: {list(df.columns)}, {len(df)} rows available", file=sys.stderr)

    text_field = None
    for cand in ("text", "transcription", "transcript", "sentence", "normalized_text"):
        if cand in df.columns:
            text_field = cand
            break
    audio_field = "audio" if "audio" in df.columns else None
    if text_field is None or audio_field is None:
        raise RuntimeError(f"Could not find text/audio columns in {list(df.columns)}")

    rows = []
    for i, row in df.iterrows():
        text = str(row[text_field]).strip()
        audio = row[audio_field]
        if not text or audio is None:
            continue
        rows.append({"idx": int(i), "text": text, "audio": audio})
        if len(rows) >= n:
            break
    return rows


def audio_field_to_wav_bytes(audio_val):
    """HF audio column is typically {'bytes': <encoded file bytes>, 'path': <name>}."""
    if isinstance(audio_val, dict) and "array" in audio_val and audio_val["array"] is not None:
        buf = io.BytesIO()
        sf.write(buf, audio_val["array"], audio_val["sampling_rate"], format="WAV", subtype="PCM_16")
        return buf.getvalue()
    if isinstance(audio_val, dict) and audio_val.get("bytes"):
        raw = audio_val["bytes"]
        # Decode whatever container format this is (wav/flac/mp3/...) into a clean PCM16 WAV.
        try:
            data, sr = sf.read(io.BytesIO(raw), dtype="int16")
        except Exception:
            import librosa
            data, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    raise ValueError(f"Unrecognized audio field shape: {type(audio_val)} {audio_val if not isinstance(audio_val, dict) else list(audio_val.keys())}")


def tts_synthesize(text, engine, voice=None, speaker_description=None):
    body = {"text": text, "engine": engine}
    if voice:
        body["voice"] = voice
    if speaker_description:
        body["speaker_description"] = speaker_description
    r = requests.post(f"{TTS_URL}/synthesize", json=body, timeout=120)
    r.raise_for_status()
    return r.content  # wav bytes


def vc_convert(source_wav_bytes, reference_wav_bytes):
    files = {
        "source_audio": ("source.wav", source_wav_bytes, "audio/wav"),
        "reference_audio": ("reference.wav", reference_wav_bytes, "audio/wav"),
    }
    r = requests.post(f"{VC_URL}/convert", files=files, timeout=180)
    r.raise_for_status()
    return r.content


def qc_check(wav_bytes, expected_text):
    files = {"audio": ("clip.wav", wav_bytes, "audio/wav")}
    data = {"expected_text": expected_text}
    r = requests.post(f"{QC_URL}/qc", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()


def main():
    for name, url in [("TTS", TTS_URL), ("VC", VC_URL), ("QC", QC_URL)]:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            print(f"[health] {name}: {resp.json()}")
        except Exception as e:
            print(f"[health] {name} UNREACHABLE at {url}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"[main] pulling {N_ROWS} rows from {DATASET_ID} (streaming)...")
    rows = load_rows(N_ROWS)
    if len(rows) < N_ROWS:
        print(f"[main] WARNING: only got {len(rows)} usable rows", file=sys.stderr)

    # Save each row's real reference audio to disk once.
    ref_wavs = []
    for r in rows:
        wav_bytes = audio_field_to_wav_bytes(r["audio"])
        ref_path = os.path.join(AUDIO_OUT, f"row{r['idx']}_real_reference.wav")
        with open(ref_path, "wb") as f:
            f.write(wav_bytes)
        ref_wavs.append(wav_bytes)

    manifest = []
    for i, r in enumerate(rows):
        text = r["text"]
        idx = r["idx"]
        # rotate reference: use a DIFFERENT row's real voice as the VC target
        ref_bytes = ref_wavs[(i + 1) % len(ref_wavs)]

        engines = ["edge"]
        if i < COMPARISON_ENGINES_FOR_FIRST_N:
            engines += ["mms", "parler"]

        for engine in engines:
            entry = {"row_idx": idx, "text": text, "engine": engine}
            try:
                print(f"[main] row {idx} engine={engine}: synthesizing...")
                synth_bytes = tts_synthesize(text, engine)
                synth_path = os.path.join(AUDIO_OUT, f"row{idx}_{engine}_synth.wav")
                with open(synth_path, "wb") as f:
                    f.write(synth_bytes)
                entry["synth_path"] = synth_path

                print(f"[main] row {idx} engine={engine}: QC on raw synth...")
                synth_qc = qc_check(synth_bytes, text)
                entry.update({f"synth_{k}": v for k, v in synth_qc.items()})

                print(f"[main] row {idx} engine={engine}: voice converting...")
                vc_bytes = vc_convert(synth_bytes, ref_bytes)
                vc_path = os.path.join(AUDIO_OUT, f"row{idx}_{engine}_vc.wav")
                with open(vc_path, "wb") as f:
                    f.write(vc_bytes)
                entry["vc_path"] = vc_path

                print(f"[main] row {idx} engine={engine}: QC on VC output...")
                vc_qc = qc_check(vc_bytes, text)
                entry.update({f"vc_{k}": v for k, v in vc_qc.items()})

                entry["status"] = "ok"
            except Exception as e:
                entry["status"] = f"error: {e}"
                print(f"[main] row {idx} engine={engine} FAILED: {e}", file=sys.stderr)

            manifest.append(entry)

    with open(MANIFEST_PATH, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[main] wrote {len(manifest)} entries to {MANIFEST_PATH}")

    ok = [m for m in manifest if m.get("status") == "ok"]
    print(f"\n=== Summary ({len(ok)}/{len(manifest)} completed without error) ===")
    print(f"{'row':>4} {'engine':>8} {'synth_pass':>11} {'synth_cer':>10} {'vc_pass':>8} {'vc_cer':>8}  text")
    for m in manifest:
        if m.get("status") != "ok":
            print(f"{m['row_idx']:>4} {m['engine']:>8}  ERROR: {m['status']}")
            continue
        print(
            f"{m['row_idx']:>4} {m['engine']:>8} "
            f"{str(m.get('synth_pass')):>11} {m.get('synth_cer', -1):>10.3f} "
            f"{str(m.get('vc_pass')):>8} {m.get('vc_cer', -1):>8.3f}  {m['text'][:40]}"
        )


if __name__ == "__main__":
    main()
