#!/bin/bash
# Wait for the v1 alignment to finish, then align the v2 manifest with --resume.
# Every v1 row is also in v2 (indicvoices-r + rasa), so seeding the output with
# v1's results means only the ~1,990 new hours get aligned.
set -u
P=/root/tts/TTS_training/pocket_TTS
until ! pgrep -f "align_data.*train.jsonl" >/dev/null 2>&1; do sleep 20; done
echo "v1 alignment finished: $(wc -l < $P/manifests/train_aligned.jsonl) rows"
cp -n $P/manifests/train_aligned.jsonl $P/manifests/train_v2_aligned.jsonl
echo "seeded v2 output with v1 rows"
cd $P/repo
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for split in valid train; do
  .venv/bin/python -m training.scripts.align_data \
    $P/manifests/${split}_v2.jsonl $P/manifests/${split}_v2_aligned.jsonl \
    --model gagan3012/wav2vec2-xlsr-nepali --device cuda \
    --batch-size 4 --sort-window 64 --resume
  echo "=== $split aligned (exit $?) ==="
done
echo "ALIGN_V2_DONE"
