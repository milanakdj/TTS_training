#!/usr/bin/env bash
# Can the training loop memorise 64 utterances at all?
#
# Every run so far ends at >=97% blank frames and loss/token ~8 = ln(4001), i.e.
# the uniform prior. Data is not the cause: diag_batches shows 0 empty targets,
# T/U median 3.04, and targets that decode back to the right text. This arm
# removes generalisation from the question entirely -- 64 fixed human-labelled
# Nepali clips, no SpecAugment, 2,000 steps. A 27M model must drive this to
# ~0 loss. If it cannot, the defect is in the model/loss wiring, not the recipe.
#
# Uniform-prior reference for this set: mean U 21.6 tokens -> ~179 loss/utt.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

for arm in bf16 fp32; do
  prec=bf16-mixed; [ "$arm" = fp32 ] && prec=32
  log "=== arm mem64_$arm (precision=$prec) ==="
  $PY -u "$HERE/pilot_train.py" \
    --train-manifest "$MAN/mem64.jsonl" --val-manifest "$MAN/mem64.jsonl" \
    --exp-dir "$ROOT/ckpt_probe" --name "mem64_$arm" \
    --max-steps 2000 --lr 1e-3 --warmup 100 --val-every 500 \
    --batch-duration 240 --max-duration 20 --workers 2 \
    --grad-clip 0.0 --freq-masks 0 --time-masks 0 --precision "$prec" \
    > "$ROOT/logs/probe_mem64_$arm.log" 2>&1
  log "arm mem64_$arm exited rc=$?"
done
log "=== memorise probe done ==="
