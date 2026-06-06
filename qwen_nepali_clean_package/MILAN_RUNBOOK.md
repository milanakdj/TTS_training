# Qwen3-TTS Nepali Training Package For Milan

This package prepares a controlled Qwen3-TTS Nepali experiment. The aim is not blind training. The aim is to test whether Qwen can beat the current Indic Parler result using clean data, correct codec tokenization, and fixed comparison sentences.

## Primary Dataset

Use this first:

```text
Titung/nepali-tts-tagged-combined
```

Hugging Face reports:

- 10,000 train rows
- audio already at 24 kHz
- parquet/audio/text format
- useful fields: `text`, `phonemes`, `snr`, `pesq`, `noise`, `reverberation`, `speech_monotony`, `text_description`
- about 6.4 GB download size

This is better for the first Qwen run than raw mixed data because it is already processed/tagged for fine-tuning.

## What This Package Contains

- `prepare_small_dataset.py`: loads Hugging Face datasets, filters clean Nepali clips, writes 24 kHz WAV files, and creates `train_raw.jsonl`.
- `inspect_dataset_speakers.py`: checks which speakers have enough usable clips.
- `check_codec_jsonl.py`: checks Qwen codec token output and filters broken rows.
- `generate_qwen_samples.py`: generates fixed test WAV files from a trained checkpoint.
- `run_qwen_1_7b_gpu.sh`: one-script GPU runner after setup.
- `qwen_1_7b_config.json`: recommended first-run settings.
- `test_sentences_nepali.txt`: fixed test sentences for Qwen vs Indic comparison.

## Important Decision

Use `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, not CustomVoice, for fine-tuning.

Qwen fine-tuning is single-speaker focused. For the first run, use the processed/tagged dataset and one clean reference audio. If the first run is stable, then scale or combine with other datasets.

## GPU Needed

- Minimum: 16 GB VRAM with small batch size.
- Recommended: 24 GB VRAM or higher.
- If out of memory, reduce `--batch_size` to `1`.

## Setup On GPU Machine

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ./Qwen3-TTS
python -m pip install -U datasets librosa soundfile huggingface_hub
```

Login to Hugging Face if needed:

```bash
huggingface-cli login
```

## Step 1: Prepare Tagged Dataset

```bash
python qwen_nepali_clean/prepare_small_dataset.py \
  --datasets Titung/nepali-tts-tagged-combined \
  --single-ref-audio \
  --max-samples-per-dataset 300 \
  --output-dir qwen_nepali_clean/data/tagged_300
```

This creates:

```text
qwen_nepali_clean/data/tagged_300/train_raw.jsonl
qwen_nepali_clean/data/tagged_300/wavs/
qwen_nepali_clean/data/tagged_300/dataset_report.json
```

## Step 2: Convert Audio To Qwen Codec Tokens

```bash
export PYTHONPATH="$PWD/Qwen3-TTS/finetuning:$PYTHONPATH"

python Qwen3-TTS/finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl qwen_nepali_clean/data/tagged_300/train_raw.jsonl \
  --output_jsonl qwen_nepali_clean/data/tagged_300/train_with_codes.jsonl
```

## Step 3: Check Codec Output

```bash
python qwen_nepali_clean/check_codec_jsonl.py \
  --input-jsonl qwen_nepali_clean/data/tagged_300/train_with_codes.jsonl
```

Use the generated filtered file for training:

```text
qwen_nepali_clean/data/tagged_300/train_with_codes_filtered.jsonl
```

## Step 4: Train Qwen 1.7B

```bash
python Qwen3-TTS/finetuning/sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path qwen_nepali_clean/outputs/qwen_1_7b_nepali_smoke \
  --train_jsonl qwen_nepali_clean/data/tagged_300/train_with_codes_filtered.jsonl \
  --batch_size 2 \
  --lr 2e-6 \
  --num_epochs 3 \
  --speaker_name nepali_speaker
```

If OOM:

```bash
python Qwen3-TTS/finetuning/sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path qwen_nepali_clean/outputs/qwen_1_7b_nepali_smoke \
  --train_jsonl qwen_nepali_clean/data/tagged_300/train_with_codes_filtered.jsonl \
  --batch_size 1 \
  --lr 2e-6 \
  --num_epochs 3 \
  --speaker_name nepali_speaker
```

## One-Script Run

After setup, this runs the small experiment end to end:

```bash
chmod +x qwen_nepali_clean/run_qwen_1_7b_gpu.sh

BATCH_SIZE=2 \
MAX_SAMPLES=300 \
LR=2e-6 \
EPOCHS=3 \
./qwen_nepali_clean/run_qwen_1_7b_gpu.sh
```

If OOM:

```bash
BATCH_SIZE=1 ./qwen_nepali_clean/run_qwen_1_7b_gpu.sh
```

## Step 5: Generate Test Samples

The one-script run generates samples automatically. Manual command:

```bash
python qwen_nepali_clean/generate_qwen_samples.py \
  --checkpoint qwen_nepali_clean/outputs/qwen_1_7b_nepali_smoke/checkpoint-epoch-2 \
  --speaker nepali_speaker \
  --sentences qwen_nepali_clean/test_sentences_nepali.txt \
  --output-dir qwen_nepali_clean/outputs/qwen_test_samples
```

## Optional: Inspect ASR Speakers

Only use this if scaling beyond the tagged dataset:

```bash
python qwen_nepali_clean/inspect_dataset_speakers.py \
  --dataset Titung/nepali-asr \
  --max-rows 50000
```

Then prepare a single-speaker ASR subset:

```bash
python qwen_nepali_clean/prepare_small_dataset.py \
  --datasets Titung/nepali-asr \
  --speaker-id PUT_SPEAKER_ID_HERE \
  --single-ref-audio \
  --max-samples-per-dataset 300 \
  --output-dir qwen_nepali_clean/data/asr_single_speaker_300
```

## What To Share Back

Please share:

- checkpoint folder or Hugging Face model link
- generated WAV samples from `test_sentences_nepali.txt`
- `dataset_report.json`
- codec check output
- training loss/logs
- exact dataset and speaker/ref audio used

## Decision Rule

Compare Qwen and Indic on the same test sentences. Continue Qwen only if it is close to Indic or clearly better in naturalness, pronunciation, and scratch/noise level.
