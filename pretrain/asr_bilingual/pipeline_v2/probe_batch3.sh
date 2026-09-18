#!/usr/bin/env bash
# The last lever the evidence supports for from-scratch: effective batch size.
#
# Re-scoring the previous session's probe matrix on a fixed held-out set (its
# own train_loss readings were bucket noise) turned up one real signal:
#
#   run_v5   600 s/update, 197,299 steps  -> held-out loss/token 7.988, 98% blank
#   bigbatch 2,400 s/update,  4,000 steps -> held-out loss/token 6.869, 98% blank
#
# 4,000 steps at 4x the batch got further than 197,299 steps at 1x, and run_v5
# saw 37x more audio. So the binding constraint is not audio seen, not steps and
# not lr (1e-3/3e-4/1e-4/3e-5 are all flat) -- it is how much audio each update
# is averaged over. NVIDIA trains this topology at 600 s x 16-64 GPUs, i.e.
# 10,000-38,000 s an update; we have been running at 1/40th of that.
#
# This arm is bigbatch with one variable moved: accum 2 -> 6 (7,200 s an update).
# Same lr, warmup, SpecAugment, clipping and step count, so the comparison is
# exact.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
while pgrep -f "^/workspace/venvs/nemo/bin/python -u .*pilot_train" >/dev/null; do sleep 30; done

log "=== arm bigbatch3 (1200 x accum 6 = 7,200 s an update) ==="
$PY -u "$HERE/pilot_train.py" \
  --train-manifest "$MAN/train_mix.jsonl" --val-manifest "$MAN/probe_val.jsonl" \
  --exp-dir "$ROOT/ckpt_probe" --name bigbatch3 \
  --max-steps 4000 --lr 5e-4 --warmup 800 --val-every 2000 \
  --batch-duration 1200 --accum 6 \
  --max-duration 20 --workers 12 --grad-clip 1.0 --freq-masks 2 --time-masks 10 \
  > "$ROOT/logs/probe_bigbatch3.log" 2>&1
log "arm bigbatch3 exited rc=$?"
log "=== bigbatch3 done ==="
