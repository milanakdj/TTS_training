# Run the code
HF_TOKEN=hf_xxx WANDB_API_KEY=wandb_xxx python qwen_nepali_runner.py all \
  --qwen-repo ./Qwen3-TTS \
  --datasets Titung/nepali-tts-tagged-combined \
  --max-samples 30000 \
  --data-dir data/train_1000 \
  --output-dir outputs/qwen_nepali_1000 \
  --batch-size 32 \
  --epochs 20 \
  --lr 3e-6 \
  --attn sdpa \
  --validation-split 0.05 \
  --eval-every-steps 100 \
  --wandb-project tts \
  --wandb-entity himalaya-ai-lab \
  --wandb-run-name qwen-nepali-1000-v13 \
  --push-to-hub \
  --hub-milanakdj/qwen-nepali-tts-v2

# Qwen3-TTS Nepali Runbook

This package is for a clean Qwen3-TTS 1.7B Nepali fine-tuning run.

## What Changed

- One main Python file: `qwen_nepali_runner.py`
- Only runtime/config/test files are included.
- Dataset duration filtering is configurable.
- `3-12 sec` is only preferred for choosing reference audio, not a hard dataset filter.
- Primary dataset is `Titung/nepali-tts-tagged-combined`.
- Audio loading now uses Hugging Face audio bytes directly, so it does not depend on auto-decoded audio rows.
- Streaming is the default, so the smoke run does not need to download/cache the full dataset first.
- Default attention is `sdpa`, so training does not require `flash-attn`.
- The runner downloads the base model into a local cache before training so checkpoint saving works.
- The patched training script disables TensorBoard logging, so Accelerate does not require a separate logging directory.
- Optional Hugging Face Hub upload uses `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` from the environment.
- The `all` command does not run inference by default. It only prepares data, tokenizes, trains, and optionally uploads.
- Training now creates a validation split and saves only the best checkpoint by lowest validation loss.
- Hub upload pushes the selected inference checkpoint contents, not the parent folder containing many checkpoints.
- Training prints live progress with estimated time left.
- Optional W&B loss logging uses `WANDB_API_KEY` from the environment and accepts project/entity/run name from CLI.

## Files

- `qwen_nepali_runner.py`: data prep, speaker inspect, codec check, training runner, optional sample generation
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
sudo apt-get update && sudo apt-get install -y sox libsox-fmt-all
pip install "qwen-tts==0.1.1" "transformers==4.57.3" "accelerate==1.12.0" "huggingface_hub==0.36.2" "protobuf<7,>=5" wandb
pip install -e Qwen3-TTS
pip install datasets soundfile librosa numpy pyarrow
huggingface-cli login
```

Put this package folder beside `Qwen3-TTS/`, then run from inside the package folder:

First verify server setup:

```bash
python qwen_nepali_runner.py preflight \
  --qwen-repo ../Qwen3-TTS \
  --min-vram-gb 16
```

If `../Qwen3-TTS` is not the correct repo location, replace it with the actual Qwen3-TTS folder path.

First run this 5-sample data smoke test:

```bash
python qwen_nepali_runner.py prepare \
  --datasets Titung/nepali-tts-tagged-combined \
  --max-samples-per-dataset 5 \
  --output-dir data/smoke_5 \
  --single-ref-audio
```

The command should print `Runner: qwen-nepali-streamlined-v13-eta-wandb-2026-06-10`.

If that shows `accepted=5`, run the train-only smoke test:

```bash
python qwen_nepali_runner.py all \
  --qwen-repo ../Qwen3-TTS \
  --datasets Titung/nepali-tts-tagged-combined \
  --max-samples 20 \
  --data-dir data/smoke_20 \
  --output-dir outputs/qwen_smoke_20 \
  --batch-size 1 \
  --epochs 1 \
  --lr 2e-6 \
  --attn sdpa
```

To push the trained checkpoint folder to Hugging Face Hub after training, set a token first:

```bash
export HF_TOKEN=hf_xxx
```

Then add these flags to the training command:

```bash
  --push-to-hub \
  --hub-repo-id username/qwen-nepali-tts \
  --hub-private
```

The model still trains to `--output-dir` first, then that checkpoint folder is uploaded to the Hub.
With validation enabled, the selected checkpoint is `best_checkpoint`.

Example 1000-sample run with Hub upload:

```bash
HF_TOKEN=hf_xxx WANDB_API_KEY=wandb_xxx python qwen_nepali_runner.py all \
  --qwen-repo ./Qwen3-TTS \
  --datasets Titung/nepali-tts-tagged-combined \
  --max-samples 1000 \
  --data-dir data/train_1000 \
  --output-dir outputs/qwen_nepali_1000 \
  --batch-size 1 \
  --epochs 2 \
  --lr 2e-6 \
  --attn sdpa \
  --validation-split 0.05 \
  --eval-every-steps 100 \
  --wandb-project qwen-nepali-tts \
  --wandb-entity your_wandb_entity \
  --wandb-run-name qwen-nepali-1000-v13 \
  --push-to-hub \
  --hub-repo-id username/qwen-nepali-tts \
  --hub-private
```

The runner checks the Hub repo before training, so missing `HF_TOKEN` or missing `--hub-repo-id` fails early.
If W&B flags are used, missing `WANDB_API_KEY` fails early with a clear error. Without W&B flags/key, training still runs normally.

If the 20-sample train-only test passes, run the first real training pass:

```bash
WANDB_API_KEY=wandb_xxx python qwen_nepali_runner.py all \
  --qwen-repo ../Qwen3-TTS \
  --datasets Titung/nepali-tts-tagged-combined \
  --max-samples 1000 \
  --data-dir data/train_1000 \
  --output-dir outputs/qwen_nepali_1000 \
  --batch-size 1 \
  --epochs 2 \
  --lr 2e-6 \
  --attn sdpa \
  --validation-split 0.05 \
  --eval-every-steps 100 \
  --wandb-project qwen-nepali-tts \
  --wandb-run-name qwen-nepali-1000-v13
```

This saves the best local model at:

```text
outputs/qwen_nepali_1000/best_checkpoint
```

If `--push-to-hub` is used, that checkpoint is uploaded to the Hub repo root so it can be loaded directly for inference.

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

Use `--no-streaming` only if you specifically want to cache-load the dataset normally.

## First Result Artifacts

After training, collect:

- training log
- `data/tagged_300/dataset_report.json`
- codec check output
- `outputs/.../best_checkpoint`
- `outputs/.../best_metrics.json`
- Hugging Face model link, if upload was enabled

For manual sample generation after training, run the `generate` command separately, or add `--generate-samples` to `all`.
