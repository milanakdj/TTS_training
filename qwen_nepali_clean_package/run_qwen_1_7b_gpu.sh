#!/usr/bin/env bash
set -euo pipefail

# Clean first run for Qwen3-TTS Nepali fine-tuning.
# Run this from the parent folder that contains qwen_nepali_clean/.
#
# Required before running:
#   1. Install Qwen3-TTS and dependencies. See MILAN_RUNBOOK.md.
#   2. Login with huggingface-cli if the dataset is gated.
#   3. Default dataset is Titung/nepali-tts-tagged-combined.
#      Use SPEAKER_ID only if the selected dataset has speaker ids.

DEVICE="${DEVICE:-cuda:0}"
DATASET_REPO="${DATASET_REPO:-Titung/nepali-tts-tagged-combined}"
SPEAKER_ID="${SPEAKER_ID:-}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-2e-6}"
EPOCHS="${EPOCHS:-3}"
SPEAKER_NAME="${SPEAKER_NAME:-nepali_speaker}"

QWEN_REPO="${QWEN_REPO:-Qwen3-TTS}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-Qwen/Qwen3-TTS-Tokenizer-12Hz}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"

DATA_DIR="${DATA_DIR:-qwen_nepali_clean/data/tagged_300}"
OUTPUT_DIR="${OUTPUT_DIR:-qwen_nepali_clean/outputs/qwen_1_7b_nepali_smoke}"
CODED_JSONL="${DATA_DIR}/train_with_codes.jsonl"
FILTERED_JSONL="${DATA_DIR}/train_with_codes_filtered.jsonl"

echo "Dataset: ${DATASET_REPO}"
echo "Speaker filter: ${SPEAKER_ID:-none}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Base model: ${BASE_MODEL}"
echo "Output: ${OUTPUT_DIR}"

if [ ! -d "${QWEN_REPO}" ]; then
  echo "Missing ${QWEN_REPO}. Clone it first:"
  echo "git clone https://github.com/QwenLM/Qwen3-TTS.git"
  exit 1
fi

PREPARE_ARGS=(
  qwen_nepali_clean/prepare_small_dataset.py
  --datasets "${DATASET_REPO}"
  --single-ref-audio
  --max-samples-per-dataset "${MAX_SAMPLES}"
  --output-dir "${DATA_DIR}"
)

if [ -n "${SPEAKER_ID}" ]; then
  PREPARE_ARGS+=(--speaker-id "${SPEAKER_ID}")
fi

python "${PREPARE_ARGS[@]}"

export PYTHONPATH="${PWD}/${QWEN_REPO}/finetuning:${PYTHONPATH:-}"

python "${QWEN_REPO}/finetuning/prepare_data.py" \
  --device "${DEVICE}" \
  --tokenizer_model_path "${TOKENIZER_MODEL}" \
  --input_jsonl "${DATA_DIR}/train_raw.jsonl" \
  --output_jsonl "${CODED_JSONL}"

python qwen_nepali_clean/check_codec_jsonl.py \
  --input-jsonl "${CODED_JSONL}" \
  --output-jsonl "${FILTERED_JSONL}"

python "${QWEN_REPO}/finetuning/sft_12hz.py" \
  --init_model_path "${BASE_MODEL}" \
  --output_model_path "${OUTPUT_DIR}" \
  --train_jsonl "${FILTERED_JSONL}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_epochs "${EPOCHS}" \
  --speaker_name "${SPEAKER_NAME}"

LATEST_CHECKPOINT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-epoch-*' | sort | tail -n 1)"

python qwen_nepali_clean/generate_qwen_samples.py \
  --checkpoint "${LATEST_CHECKPOINT}" \
  --speaker "${SPEAKER_NAME}" \
  --sentences qwen_nepali_clean/test_sentences_nepali.txt \
  --output-dir qwen_nepali_clean/outputs/qwen_test_samples

echo "Done."
echo "Checkpoint: ${LATEST_CHECKPOINT}"
echo "Samples: qwen_nepali_clean/outputs/qwen_test_samples"
