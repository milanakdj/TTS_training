#!/usr/bin/env bash
# Does a real batch fix it?
#
# Every collapsed run took ~50 utterances an update (batch_duration 600). NeMo's
# published conformer recipes reach lr 2e-3 at 1,000-2,000 utterances across
# 16-64 GPUs -- our updates carry 1/40th the gradient signal at nearly the same
# lr. This arm buys the batch back with accumulation, which is the one thing a
# single GPU can still do, and sets lr by sqrt-scaling from that reference.
#
#   batch_duration 1200 x accum 2 = 2,400 s of audio an update (~200 utts)
#   lr 5e-4, warmup 800, 4,000 optimizer steps ~= 2,670 h seen
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for any running trainer to finish"
# Anchor the pattern at the start of the command line. `pgrep -f` matches the
# whole cmdline, and a shell that merely *writes* this script carries the
# pattern itself in its cmdline -- an unanchored match finds that shell and
# the wait never ends. Only a real trainer starts with the venv python.
while pgrep -f "^/workspace/venvs/nemo/bin/python -u .*pilot_train" >/dev/null; do sleep 30; done

log "=== arm bigbatch ==="
$PY -u "$HERE/pilot_train.py" \
  --train-manifest "$MAN/train_mix.jsonl" --val-manifest "$MAN/probe_val.jsonl" \
  --exp-dir "$ROOT/ckpt_probe" --name bigbatch \
  --max-steps 4000 --lr 5e-4 --warmup 800 --val-every 2000 \
  --batch-duration 1200 --accum 2 \
  --max-duration 20 --workers 12 --grad-clip 1.0 --freq-masks 2 --time-masks 10 \
  > "$ROOT/logs/probe_bigbatch.log" 2>&1
log "arm bigbatch exited rc=$?"
