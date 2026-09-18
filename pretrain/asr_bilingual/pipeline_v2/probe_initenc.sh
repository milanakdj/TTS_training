#!/usr/bin/env bash
# The control that separates "our loop is broken" from "from-scratch is the wall".
#
# nvidia/parakeet-tdt_ctc-110m is on this box already
# (/workspace/hf_cache/...), and its config is where this recipe's topology came
# from. Load ONLY its encoder, leave the 4,000-class CTC head random, and train
# on the same 6 h that every from-scratch arm has failed to fit.
#
#   pretrained encoder + random head descends  -> loop, data, loss, tokenizer and
#     dataloader are all sound; the wall is specific to a randomly initialised
#     encoder, and the fix has to be about how training starts.
#   it fails too -> something in this training loop is broken in a way that no
#     amount of hyperparameter search will find, and that is where to dig next.
#
# Paired against arch110m, which is the identical topology from random init, so
# the two arms differ in exactly one thing.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
DONOR=/workspace/hf_cache/models--nvidia--parakeet-tdt_ctc-110m/snapshots/431a349f3051ab85c22b9b7a2741b5fe77065665/parakeet-tdt_ctc-110m.nemo
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
while pgrep -f "^/workspace/venvs/nemo/bin/python -u .*pilot_train" >/dev/null; do sleep 30; done

log "=== arm initenc (parakeet encoder, random CTC head) ==="
$PY -u "$HERE/pilot_train.py" \
  --train-manifest "$MAN/sub2k.jsonl" --val-manifest "$MAN/sub2k_score.jsonl" \
  --exp-dir "$ROOT/ckpt_probe" --name initenc \
  --init-encoder "$DONOR" --d-model 512 --n-layers 17 --n-heads 8 \
  --max-steps 4000 --lr 1e-3 --warmup 400 --val-every 2000 \
  --batch-duration 240 --max-duration 20 --workers 8 \
  --grad-clip 0.0 --freq-masks 0 --time-masks 0 \
  > "$ROOT/logs/probe_initenc.log" 2>&1
log "arm initenc exited rc=$?"
log "=== initenc done ==="
