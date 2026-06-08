# Qwen3-TTS Nepali Runbook

This package is for a clean Qwen3-TTS 1.7B Nepali fine-tuning smoke run.

## What Changed

- One main Python file: `qwen_nepali_runner.py`
- No extra draft message files.
- Dataset duration filtering is configurable.
- `3-12 sec` is only preferred for choosing reference audio, not a hard dataset filter.
- Primary dataset is `Titung/nepali-tts-tagged-combined`.

## Files

- `qwen_nepali_runner.py`: data prep, speaker inspect, codec check, training runner, sample generation
- `qwen_1_7b_config.json`: base model, tokenizer, dataset, and first-run settings
- `test_sentences_nepali.txt`: fixed comparison sentences
- `QWEN_RUNBOOK.md`: this guide

## GPU Setup

Use Linux GPU with 24GB VRAM preferred. 16GB can be tried with batch size 1.

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r Qwen3-TTS/requirements.txt
pip install datasets soundfile librosa
huggingface-cli login
```

Put this package folder beside `Qwen3-TTS/`, then run from inside the package folder:

```bash
python qwen_nepali_runner.py all \
  --qwen-repo ../Qwen3-TTS \
  --max-samples 300 \
  --batch-size 2 \
  --epochs 3 \
  --lr 2e-6
```

If GPU memory fails:

```bash
python qwen_nepali_runner.py all \
  --qwen-repo ../Qwen3-TTS \
  --max-samples 300 \
  --batch-size 1 \
  --epochs 3 \
  --lr 2e-6
```

## Optional Dataset Commands

Inspect speakers if the dataset has speaker IDs:

```bash
python qwen_nepali_runner.py inspect \
  --dataset Titung/nepali-asr \
  --max-rows 50000
```

Prepare data only:

```bash
python qwen_nepali_runner.py prepare \
  --datasets Titung/nepali-tts-tagged-combined \
  --max-samples-per-dataset 300 \
  --output-dir data/tagged_300 \
  --single-ref-audio
```

Use wider or narrower duration limits:

```bash
python qwen_nepali_runner.py prepare \
  --datasets Titung/nepali-tts-tagged-combined \
  --min-sec 1 \
  --max-sec 20 \
  --max-samples-per-dataset 300 \
  --output-dir data/tagged_300 \
  --single-ref-audio
```

Set `--max-sec 0` to disable max duration filtering.

## First Result To Share

After training, share:

- training log
- `data/tagged_300/dataset_report.json`
- codec check output
- checkpoint folder or Hugging Face model link
- generated WAV samples from `outputs/qwen_test_samples`

Compare those samples with the same Indic sentences before scaling.
