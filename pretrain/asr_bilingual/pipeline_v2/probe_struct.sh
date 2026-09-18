#!/usr/bin/env bash
# Two candidate causes of the blank collapse, on a testbed that must be fittable.
#
# Fixed facts going in: 64 utterances reach 0.45 loss/token; 2,000 utterances
# (6 h, 37 epochs) stay at 6.95 with 95% blank; gradients from independent
# batches agree at cosine 0.93, so the corpus is coherent and every batch is
# pulling the same way -- toward blank.
#
#   sub4x    encoder frames 12.5/s -> 25/s. diag_batches puts T/U at p05 1.82,
#            which leaves a from-scratch model almost no alignment slack in the
#            bottom of the distribution. Costs ~2x per step.
#   lr*      the update may simply be too large to keep what a batch teaches;
#            a repeating batch is the one case where that cannot hurt.
#
# Pass criterion is the same for all: loss/token on sub2k_score must come off
# 8.29 (= ln 4001, the uniform prior) and blank% off 95.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
# Anchor the pattern at the start of the command line. `pgrep -f` matches the
# whole cmdline, and a shell that merely *writes* this script carries the
# pattern itself in its cmdline -- an unanchored match finds that shell and
# the wait never ends. Only a real trainer starts with the venv python.
while pgrep -f "^/workspace/venvs/nemo/bin/python -u .*pilot_train" >/dev/null; do sleep 30; done

run() {  # name, extra args...
  local name=$1; shift
  log "=== arm $name ==="
  $PY -u "$HERE/pilot_train.py" \
    --train-manifest "$MAN/sub2k.jsonl" --val-manifest "$MAN/sub2k_score.jsonl" \
    --exp-dir "$ROOT/ckpt_probe" --name "$name" \
    --max-steps 4000 --warmup 400 --val-every 2000 \
    --batch-duration 240 --max-duration 20 --workers 8 \
    --grad-clip 0.0 --freq-masks 0 --time-masks 0 "$@" \
    > "$ROOT/logs/probe_$name.log" 2>&1
  log "arm $name exited rc=$?"
}

run fit2k_sub4x  --lr 1e-3 --subsampling 4
for lr in 3e-4 1e-4 3e-5; do run "fit2k_lr$lr" --lr "$lr"; done
log "=== struct+lr sweep done ==="
