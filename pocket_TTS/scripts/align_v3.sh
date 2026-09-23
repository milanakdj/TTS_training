#!/bin/bash
# Forced-align the v3 manifests: two passes, two models, one merged output.
#
# The Nepali pass appends to an output already seeded with the v2 alignments that
# are still valid (seed_align_v3.py), so only the ~1.8% of rows whose transcript
# moved get redone. The English pass uses an English CTC model -- the Nepali one
# raises on a Latin transcript rather than aligning it badly.
#
# Word timings are what let the loader cut an utterance between two words and
# trim trailing silence. Without them it falls back to a random prompt window and
# no trim, and 12% of utterances carry >1 s of trailing silence, which teaches the
# model to emit silence instead of EOS so generations never terminate.
set -eu
P=/root/tts/TTS_training/pocket_TTS
cd "$P/repo"
export HF_TOKEN=$(cat /root/.hf_milanakdj 2>/dev/null || echo "${HF_TOKEN:-}")
export HF_HOME=/workspace/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NE_MODEL=gagan3012/wav2vec2-xlsr-nepali
EN_MODEL=facebook/wav2vec2-base-960h

for split in valid_v3 train_v3; do
  for lang in ne en; do
    IN="$P/manifests/${split}_${lang}.jsonl"
    [ -s "$IN" ] || { echo "skip $IN (empty)"; continue; }
    MODEL=$([ "$lang" = ne ] && echo "$NE_MODEL" || echo "$EN_MODEL")
    echo "=== aligning $split/$lang with $MODEL ($(wc -l < "$IN") rows)"
    .venv/bin/python -m training.scripts.align_data \
      "$IN" "$P/manifests/${split}_aligned.jsonl" \
      --model "$MODEL" --device cuda --batch-size 16 --sort-window 64 --resume
  done
  echo "=== $split aligned: $(wc -l < "$P/manifests/${split}_aligned.jsonl") rows"
done
echo ALIGN_V3_DONE
