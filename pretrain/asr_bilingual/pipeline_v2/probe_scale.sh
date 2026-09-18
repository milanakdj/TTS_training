#!/usr/bin/env bash
# Where does learning die between 64 utterances and the full mix?
#
# mem64 proved the wiring: 64 utterances go from 6.91 to 0.45 loss/token in 2,000
# steps, identical in bf16 and fp32. run_v5 on the full mix sat at 8.14 loss/token
# and 99% blank frames for 197,299 steps. Everything except the data scale and
# three hyperparameters is shared, so this bisects one axis at a time, holding the
# *working* mem64 settings fixed (no SpecAugment, warmup 100, batch 240, lr 1e-3).
#
#   fullmix  -- full corpus, mem64 hyperparameters. If this learns, the defect is
#               a run_v5 hyperparameter (SpecAugment / warmup 15k / batch 600).
#   sub50k   -- 148 h, 50,000 utterances
#   sub2k    -- 6 h, 2,000 utterances
#
# Scored afterwards with eval_probe.py on a fixed set: train_loss is not
# comparable across arms because the token count per batch differs.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "waiting for any running trainer to finish"
# Match the python binary, not the bare script name: any shell whose command
# line merely *contains* "pilot_train.py" -- a heredoc that writes this very
# file, an editor -- matches the loose pattern, and then the wait never ends.
# Anchor the pattern at the start of the command line. `pgrep -f` matches the
# whole cmdline, and a shell that merely *writes* this script carries the
# pattern itself in its cmdline -- an unanchored match finds that shell and
# the wait never ends. Only a real trainer starts with the venv python.
while pgrep -f "^/workspace/venvs/nemo/bin/python -u .*pilot_train" >/dev/null; do sleep 30; done

for arm in sub2k sub50k fullmix; do
  man="$MAN/$arm.jsonl"; [ "$arm" = fullmix ] && man="$MAN/train_mix.jsonl"
  log "=== arm $arm ($man) ==="
  $PY -u "$HERE/pilot_train.py" \
    --train-manifest "$man" --val-manifest "$MAN/probe_val.jsonl" \
    --exp-dir "$ROOT/ckpt_probe" --name "scale_$arm" \
    --max-steps 3000 --lr 1e-3 --warmup 100 --val-every 1500 \
    --batch-duration 240 --max-duration 20 --workers 8 \
    --grad-clip 0.0 --freq-masks 0 --time-masks 0 \
    > "$ROOT/logs/probe_scale_$arm.log" 2>&1
  log "arm $arm exited rc=$?"
done
log "=== scale bisect done ==="
