#!/usr/bin/env bash
# Which part of the mix kills learning.
#
# The ablation's clip1_spec arm -- the exact run_v5 recipe -- descends 293 -> 121
# in 3,000 steps on 60 h of clean human Nepali. run_v5 itself sat at ~370 for
# 175,000 steps on the 4,200 h mix. Same code, same knobs, different data path,
# so the fault is in the mix or in how the mix is read.
#
# Arms, all 3,000 steps, all clip 1.0 + specaug 2/10 so they compare directly
# against clip1_spec:
#   mixcfg   the multiplexed weighted input_cfg -- exactly what run_v5 reads
#   mixflat  the same rows as one concatenated manifest, no multiplexing
#   nepseudo canary-labelled Nepali alone (1,919 h)
#   enpseudo pseudo-labelled Indian English alone (1,858 h)
# mixcfg vs mixflat isolates the lhotse multiplex; the last two isolate content.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

arm() {  # name, extra args
  local name=$1; shift
  log "=== arm $name: $* ==="
  $PY -u "$HERE/pilot_train.py" --val-manifest "$MAN/probe_val.jsonl" \
    --exp-dir "$ROOT/ckpt_probe" --name "$name" \
    --max-steps 3000 --lr 1e-3 --warmup 500 --val-every 1000 \
    --max-duration 20 --workers 8 --grad-clip 1.0 --freq-masks 2 --time-masks 10 \
    "$@" > "$ROOT/logs/probe_$name.log" 2>&1
  log "arm $name exited rc=$?"
}

arm mixcfg   --train-manifest "$MAN/train_mix.jsonl" --input-cfg "$MAN/train_input_cfg.yaml"
arm mixflat  --train-manifest "$MAN/train_mix.jsonl"
arm nepseudo --train-manifest "$MAN/train_ne_pseudo.jsonl"
arm enpseudo --train-manifest "$MAN/train_en_pseudo.jsonl"
log "=== source bisect done ==="
