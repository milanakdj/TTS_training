#!/usr/bin/env bash
# The last from-scratch run this box gets, and the criterion for killing it.
#
# Where the evidence actually stands. Update size is the only knob that has ever
# moved held-out loss/token, and it moves it far too little:
#
#   run_v5     600 s/update, 197,299 steps (100,000 h seen) -> 7.988, 98% blank
#   bigbatch  2,400 s/update,   4,000 steps (  2,667 h seen) -> 6.869, 98% blank
#   bigbatch3 7,200 s/update,   4,000 steps (  8,000 h seen) -> 6.681, 98% blank
#
# Monotone in batch, but tripling it bought 0.19 nats and the model is still at
# 98% blank with WER 0.998. Against that, parakeet's encoder with a random CTC
# head reaches 3.589 / WER 0.632 in 4,000 steps on 6 h.
#
# So why run at all: no arm has ever been given a healthy update size AND a real
# budget at the same time. PRETRAINING_FROM_SCRATCH.md's own floor is 50,000 h
# seen; bigbatch3 saw 8,000. run_v5 saw 100,000 h but at an eighth of the update
# size the gradient noise scale (~4,600 s) says is the minimum. This run is the
# first to satisfy both, and it is one night of an otherwise idle GPU.
#
#   7,200 s an update x 50,000 steps = 100,000 h seen, ~18 h at 1.29 s/step.
#
# KILL CRITERION, decided now rather than after seeing the curve: score
# eval_probe.py on hum_heldout.jsonl at step 25,000 (= 50,000 h, the doc's floor).
# If loss/token is not below 5.5 and blank% not below 90 by then, from-scratch is
# finished on this hardware and the fallback is encoder-init -- do not let it run
# to 50,000 steps on hope. run_v5 already cost 7.5 h of exactly that.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
while pgrep -f "^/workspace/venvs/nemo/bin/python -u .*pilot_train" >/dev/null; do sleep 30; done

# lr 1e-3 at 7,200 s an update: run_v5's value at 12x the batch, and the same
# effective peak NVIDIA reaches under Noam for this topology (1.25e-3). The
# small-batch sweep found nothing between 3e-5 and 1e-3, so lr is not being
# tuned here -- it is being held at the reference value while the batch moves.
log "=== run_v6: 7,200 s an update, 50,000 steps, 100,000 h seen ==="
$PY -u "$HERE/pilot_train.py" \
  --train-manifest "$MAN/train_mix.jsonl" --val-manifest "$MAN/val_mix.jsonl" \
  --input-cfg "$MAN/train_input_cfg.yaml" \
  --exp-dir "$ROOT/ckpt" --name run_v6 \
  --max-steps 50000 --lr 1e-3 --warmup 2000 --val-every 2500 \
  --batch-duration 1200 --accum 6 \
  --max-duration 20 --workers 12 --grad-clip 1.0 --freq-masks 2 --time-masks 10 \
  > "$ROOT/logs/run_v6.log" 2>&1
log "run_v6 exited rc=$?"
log "=== run_v6 done ==="
