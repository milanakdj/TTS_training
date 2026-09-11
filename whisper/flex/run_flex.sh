#!/bin/bash
# Finetune indic-transcribe-flex on human-transcribed Nepali, then score base vs
# finetuned on the held-out gold test split and the cross-source probe.
#
# Gold data only, by measurement: flex is a canary-1b-v2 derivative and our big
# corpora are 64% canary pseudo-labels, so it already "scores" 0.015 there while
# sitting at 0.129-0.177 against human transcripts. See flex_finetune.py.
set -uo pipefail
cd /root/tts/TTS_training/whisper/flex
export HF_HOME=/workspace/hf_cache
# inherited RANK/LOCAL_RANK without WORLD_SIZE makes accelerate init distributed
# and die -- same unset run_queue.sh carries
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
PY=/workspace/venvs/flex/bin/python
LOG=/root/flex-queue.log
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== baseline: flex on held-out GOLD test (600 clips) ==="
FLEX_EVAL=/root/tts/TTS_training/whisper/flex/gold_test_eval.json \
  $PY flex_eval2.py --tag goldtest_base 2>&1 | grep -avE "^\[transformers\]" | tail -3 | tee -a "$LOG"

say "=== finetuning (2 epochs, gold only) ==="
$PY flex_finetune.py --batch 16 --accum 2 --lr 1e-5 --epochs 2 \
    --workers 6 --eval-steps 500 --warmup-steps 200 >> "$LOG" 2>&1
say "finetune exited rc=$?"

say "=== finetuned: GOLD test ==="
FLEX_MODEL=/workspace/flex-nepali \
FLEX_EVAL=/root/tts/TTS_training/whisper/flex/gold_test_eval.json \
  $PY flex_eval2.py --tag goldtest_ft 2>&1 | grep -avE "^\[transformers\]" | tail -3 | tee -a "$LOG"

say "=== finetuned: cross-source probe (contamination control) ==="
FLEX_MODEL=/workspace/flex-nepali \
FLEX_EVAL=/root/tts/TTS_training/whisper/flex/crosssource100.json \
  $PY flex_eval2.py --tag crosssource_ft 2>&1 | grep -avE "^\[transformers\]" | tail -3 | tee -a "$LOG"

say "=== FLEX QUEUE COMPLETE ==="
grep -aE "^== flex" "$LOG" | tail -6
