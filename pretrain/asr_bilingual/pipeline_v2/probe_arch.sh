#!/usr/bin/env bash
# Is it the cut-down architecture, or the training loop?
#
# This recipe took parakeet-tdt_ctc-110m's config for topology and then halved
# d_model (512 -> 256), dropped a layer (17 -> 16) and halved the heads (8 -> 4)
# to land near the ~32M the pilot asks for. Everything else -- preprocessor,
# subsampling, dropout, spec_augment, optimiser family, batch_duration 600 --
# already matches what NVIDIA trained that model with, and their effective peak
# lr under Noam works out to 1.25e-3 against our 1e-3.
#
# So restore the one block that was changed: train the *template's own* encoder
# size from random init on the same 6 h. If 110M learns where 27M does not, the
# wall is capacity or optimisation at this width. If it fails identically, the
# architecture is exonerated and the defect is in the training loop or the data
# path, and no amount of hyperparameter search will find it.
#
# ~4x the step cost of the 27M arms; 4,000 steps is roughly half an hour.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
while pgrep -f "^/workspace/venvs/nemo/bin/python -u .*pilot_train" >/dev/null; do sleep 30; done

log "=== arm long2k (20,000 steps on 3.6 h: is 4,000 simply too few?) ==="
# Every arm so far ran 3,000-4,000 steps. Standard Conformer-CTC sits near the
# blank prior for thousands of steps before it breaks out, so a 4,000-step null
# is weak evidence on its own. 20,000 steps over hum2k is ~250 epochs of 3.6 h of
# clean human-labelled read Nepali -- if that does not descend, no horizon will.
# run_v5's 197,299 flat steps say it will not, but run_v5 also carried
# SpecAugment, grad-clip 1.0 and a 15k warmup; this arm carries none of them.
$PY -u "$HERE/pilot_train.py" \
  --train-manifest "$MAN/hum2k.jsonl" --val-manifest "$MAN/hum_heldout.jsonl" \
  --exp-dir "$ROOT/ckpt_probe" --name long2k \
  --max-steps 20000 --lr 1e-3 --warmup 400 --val-every 5000 \
  --batch-duration 240 --max-duration 20 --workers 8 \
  --grad-clip 0.0 --freq-masks 0 --time-masks 0 \
  > "$ROOT/logs/probe_long2k.log" 2>&1
log "arm long2k exited rc=$?"

log "=== arm short20k (2-6 s clips only: does a shorter alignment problem escape?) ==="
# The line search at the collapsed checkpoint finds no descent at any step size
# from 1e-6 to 3e-3 -- the all-blank point is a genuine flat attractor, so the
# only thing that matters is not falling into it. Utterance length is the
# untested lever: a 20 s clip is ~250 encoder frames the model must fill with
# blanks before any label helps, a 4 s clip is ~50. mem64, the one arm that ever
# escaped, was 4-9 s.
$PY -u "$HERE/pilot_train.py" \
  --train-manifest "$MAN/short20k.jsonl" --val-manifest "$MAN/short20k_score.jsonl" \
  --exp-dir "$ROOT/ckpt_probe" --name short20k \
  --max-steps 8000 --lr 1e-3 --warmup 400 --val-every 4000 \
  --batch-duration 240 --max-duration 6 --workers 8 \
  --grad-clip 0.0 --freq-masks 0 --time-masks 0 \
  > "$ROOT/logs/probe_short20k.log" 2>&1
log "arm short20k exited rc=$?"

log "=== arm arch110m (d_model 512, 17 layers, 8 heads) ==="
$PY -u "$HERE/pilot_train.py" \
  --train-manifest "$MAN/sub2k.jsonl" --val-manifest "$MAN/sub2k_score.jsonl" \
  --exp-dir "$ROOT/ckpt_probe" --name arch110m \
  --d-model 512 --n-layers 17 --n-heads 8 \
  --max-steps 4000 --lr 1e-3 --warmup 400 --val-every 2000 \
  --batch-duration 240 --max-duration 20 --workers 8 \
  --grad-clip 0.0 --freq-masks 0 --time-masks 0 \
  > "$ROOT/logs/probe_arch110m.log" 2>&1
log "arm arch110m exited rc=$?"

for lr in 1e-3; do
  log "=== arm vocab1k_lr$lr ==="
  $PY -u "$HERE/pilot_train.py" \
    --train-manifest "$MAN/sub2k.jsonl" --val-manifest "$MAN/sub2k_score.jsonl" \
    --exp-dir "$ROOT/ckpt_probe" --name "vocab1k_lr$lr" \
    --tokenizer-dir "$ROOT/tokenizer_1k" \
    --max-steps 4000 --lr "$lr" --warmup 400 --val-every 2000 \
    --batch-duration 240 --max-duration 20 --workers 8 \
    --grad-clip 0.0 --freq-masks 0 --time-masks 0 \
    > "$ROOT/logs/probe_vocab1k_lr$lr.log" 2>&1
  log "arm vocab1k_lr$lr exited rc=$?"
done
log "=== arch+vocab probe done ==="
