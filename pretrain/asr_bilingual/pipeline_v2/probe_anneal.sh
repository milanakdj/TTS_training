#!/usr/bin/env bash
# Score mid-flight checkpoints the way the annealed arms were scored.
#
# This morning's table compared like with unlike. bigbatch and bigbatch3 were
# read at the END of their cosine (lr 1e-6, 0.2% of peak); run_v5 and run_v6 were
# read at 84% and 87% of peak, because both had their scheduler sized from
# --hours-seen and stopped long before that horizon. A model sitting at peak lr
# is maximally noisy and scores badly for reasons that have nothing to do with
# what it has learned.
#
# So: take each mid-flight checkpoint, anneal it over 600 steps to min_lr with
# everything else held at its own run's settings, and score that. Now every
# number in the table means the same thing.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

anneal() {  # name, ckpt, lr, batch-duration, accum
  log "=== anneal $1 from lr $3 ==="
  $PY -u "$HERE/pilot_train.py" \
    --train-manifest "$MAN/train_mix.jsonl" --val-manifest "$MAN/probe_val.jsonl" \
    --exp-dir "$ROOT/ckpt_probe" --name "anneal_$1" \
    --restore-from "$2" \
    --max-steps 600 --lr "$3" --warmup 10 --val-every 600 \
    --batch-duration "$4" --accum "$5" \
    --max-duration 20 --workers 12 --grad-clip 1.0 --freq-masks 2 --time-masks 10 \
    > "$ROOT/logs/probe_anneal_$1.log" 2>&1
  log "anneal $1 exited rc=$?"
}

anneal run_v5 "$(ls /workspace/asr_pretrain_v2/ckpt/run_v5/*/checkpoints/run_v5.nemo)" 8.38e-4 600 1
anneal run_v6 "$(ls /workspace/asr_pretrain_v2/ckpt/run_v6/*/checkpoints/run_v6.nemo)" 8.73e-4 1200 6
log "=== anneal probe done ==="
