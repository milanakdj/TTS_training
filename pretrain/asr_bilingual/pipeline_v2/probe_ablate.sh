#!/usr/bin/env bash
# Why five runs plateaued at val_wer ~1.0 with a flat CTC loss.
#
# The one probe that ever crushed the loss (a single batch, 1093 -> 12.6 in 300
# steps) did not go through Lightning; the clean-60 probe that did go through it
# only moved 333 -> 184. So the suspects are the two things the Lightning path
# adds and an inline loop does not: gradient clipping, and SpecAugment.
#
# Same clean 60 h of human Nepali (rasa + indicvoices-r) in every arm, 3,000
# steps, lr 1e-3, warmup 500. Only the two knobs move. A model that cannot
# descend on 60 h of studio-clean human speech in 3,000 steps is broken
# regardless of what the 4,200 h mix contains.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

run_arm() {
  local name=$1 clip=$2 fm=$3 tm=$4
  log "=== arm $name: grad_clip=$clip freq_masks=$fm time_masks=$tm ==="
  $PY -u "$HERE/pilot_train.py" \
    --train-manifest "$MAN/probe_clean.jsonl" --val-manifest "$MAN/probe_val.jsonl" \
    --exp-dir "$ROOT/ckpt_probe" --name "$name" \
    --max-steps 3000 --lr 1e-3 --warmup 500 --val-every 1000 \
    --max-duration 20 --workers 8 \
    --grad-clip "$clip" --freq-masks "$fm" --time-masks "$tm" \
    > "$ROOT/logs/probe_$name.log" 2>&1
  log "arm $name exited rc=$?"
}

run_arm clip1_spec   1.0 2 10     # exactly the run_v5 recipe
run_arm clip0_spec   0.0 2 10
run_arm clip0_nospec 0.0 0 0
run_arm clip1_nospec 1.0 0 0
log "=== all arms done ==="
