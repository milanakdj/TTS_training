#!/usr/bin/env bash
# Is the learning rate simply too high for a ~50-utterance batch?
#
# The evidence that it is:
#   * the first pilot ran at an effective 2.2e-6 (the Noam rescaling bug) and is
#     the ONLY run whose val_loss descended monotonically -- 777 -> 493 over 34k
#     steps;
#   * run_v4 at 2.5e-3 diverged, train_loss 380 -> 490 as the schedule ramped;
#   * run_v5 at 1e-3 collapsed to 99% blank frames and its val_loss ROSE, 407 at
#     step 5k to 440-490 after.
# NeMo's own conformer recipes use ~2e-3, but at a global batch of 1,000-2,000
# utterances across 16-64 GPUs. batch_duration 600 gives us ~50. Linear scaling
# puts the equivalent at ~6e-5, sqrt scaling at ~3.5e-4 -- both far below 1e-3.
#
# Same data as run_v5 (the flat mix), 6,000 steps, only the lr moves. Scored with
# eval_probe.py against run_v5's own step-5k checkpoint.
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

for lr in 2.5e-4 6e-5; do
  name="mix_lr${lr}"
  log "=== arm $name ==="
  $PY -u "$HERE/pilot_train.py" \
    --train-manifest "$MAN/train_mix.jsonl" --val-manifest "$MAN/probe_val.jsonl" \
    --exp-dir "$ROOT/ckpt_probe" --name "$name" \
    --max-steps 6000 --lr "$lr" --warmup 1000 --val-every 2000 \
    --max-duration 20 --workers 8 --grad-clip 1.0 --freq-masks 2 --time-masks 10 \
    > "$ROOT/logs/probe_$name.log" 2>&1
  log "arm $name exited rc=$?"
done
log "=== lr sweep done ==="
