#!/bin/bash
# Train Whisper for Nepali on the 2,215 h SPONTANEOUS corpus, back to back across
# sizes, scoring each on the same out-of-domain set and pushing finals privately.
#
# Why the corpus and not SLR: every existing Nepali Whisper was finetuned on
# clean studio read speech, which is exactly why they land at 0.245-0.301 WER on
# spontaneous audio. whisper-medium-nepali-final was ALREADY trained on SLR, so
# retraining any size on SLR alone largely reproduces a model we own. Domain
# adaptation is the biggest win available, ahead of any change of model size.
#
# Baselines to beat, all measured on the same 100 unseen mahadhwani clips with the
# STANDARD single-sot prefix (the doubled-prefix warning in the cards is wrong for
# the published checkpoints -- verified 2026-09-08):
#   base large-v3-turbo, no finetune   0.937 / 0.364
#   whisper-large-v3-nepali-final      0.245 / 0.090
#   whisper-medium-nepali-final        0.301 / 0.124
#
# Known risk, accepted deliberately: corpus transcripts are machine-generated
# (transcript_used: saaras, kept where 3 ASRs agreed), so quality is capped near
# those systems' agreement. The result may be better on spontaneous speech and
# worse on clean read speech. That is a finding to report, not a reason to skip.
set -uo pipefail
cd /root/tts/TTS_training/whisper

VENV=/root/tts/TTS_training/synthetic_pipeline/tts_service/.venv/bin/python3
SCORER=/root/tts/TTS_training/synthetic_pipeline/vc_service/.venv/bin/python3
LOGDIR=/root/whisper-queue
MANIFEST=/root/tts/TTS_training/whisper/nepali_corpus_asr.jsonl
mkdir -p "$LOGDIR"

# RANK/LOCAL_RANK are inherited in this environment while WORLD_SIZE is not, so
# accelerate tries to init a distributed process group and dies at
# Seq2SeqTrainingArguments. Same unset that pocket_TTS/scripts/train_*.sh carry.
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

export HF_TOKEN=$(cat /root/.hf_milanakdj)
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export TOKENIZERS_PARALLELISM=false
# $HOME is on `/` at 95% full. Checkpoints go to /workspace (851TB free).
export WHISPER_OUTPUT_ROOT=/workspace/whisper-output

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# turbo first: 809M, large-v3's encoder with a 4-layer decoder, the likeliest to
# ship. small next: 244M, genuinely new and tests how far down the curve holds.
# medium last and only if turbo disappoints -- on the corpus it is informative, but
# it is the least so of the three.
for VARIANT in large-v3-turbo small; do
  LOG="$LOGDIR/$VARIANT.log"
  if grep -qa "TEST RESULTS" "$LOG" 2>/dev/null; then
    say "$VARIANT already finished, skipping"
    continue
  fi
  say "=== training $VARIANT on 2,215 h corpus ==="
  df -h / /workspace | tail -2

  WHISPER_VARIANT="$VARIANT" \
  WHISPER_LOCAL_MANIFEST="$MANIFEST" \
  WHISPER_NUM_ROWS=0 \
  WHISPER_EPOCHS=1 \
  WHISPER_EVAL_SUBSET=500 \
  WHISPER_CKPT_BACKUP=0 \
  WHISPER_FINAL_PRIVATE=1 \
  WHISPER_RUN_SUFFIX=corpus \
  "$VENV" whisper-finetune.py > "$LOG" 2>&1
  say "$VARIANT exited rc=$?"
  grep -aE "TEST RESULTS|test_wer|test_cer|Pushed to|NOT pushing" "$LOG" | tail -6

  # Score on the same 100 unseen mahadhwani clips every baseline was scored on.
  # FT_NSOT=1: this run used the fixed collator, so it takes the standard prefix.
  REPO="milanakdj/whisper-$VARIANT-nepali-final-corpus"
  say "scoring $REPO"
  ASR_MODEL=ft FT_MODEL="$REPO" FT_NSOT=1 \
    "$SCORER" /root/tts/TTS_training/pocket_TTS/infer/asr_test/asr_ab.py \
    > "$LOGDIR/$VARIANT.mahadhwani.log" 2>&1
  grep -aE "decoding|^==" "$LOGDIR/$VARIANT.mahadhwani.log" | tail -2
done

say "=== QUEUE COMPLETE ==="
printf '\n%-40s %8s %8s\n' "model (100 unseen mahadhwani clips)" WER CER
printf '%-40s %8s %8s\n' "base large-v3-turbo (no finetune)" 0.937 0.364
printf '%-40s %8s %8s\n' "large-v3-nepali (SLR, existing)" 0.245 0.090
printf '%-40s %8s %8s\n' "medium-nepali (SLR, existing)" 0.301 0.124
for VARIANT in large-v3-turbo small; do
  line=$(grep -aE "^==" "$LOGDIR/$VARIANT.mahadhwani.log" 2>/dev/null | tail -1)
  [ -n "$line" ] && printf '%-40s %s\n' "NEW $VARIANT (corpus)" "$line"
done
