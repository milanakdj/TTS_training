#!/usr/bin/env bash
# run_v6 was the last from-scratch arm this box gets, and it failed its own
# pre-registered kill criterion. Scored with eval_probe.py on hum_heldout.jsonl
# (200 human Nepali clips, the fixed instrument), at step ~47,500:
#
#   run_v6 @ 47.5k, 7,200 s/update, lr 1e-3   8.291 loss/token  98.66% blank  WER 1.000
#   bigbatch3,      7,200 s/update, lr 5e-4   6.681             98%           --
#   initenc (parakeet encoder + random head)  3.589             73.6%         0.632
#   uniform prior ln(4001)                    8.294
#
# 97,500 h seen landed the model *on the uniform prior* -- 0.003 nats from it --
# after 12x the steps of bigbatch3 at the same update size and double the lr.
# The criterion was loss/token < 5.5 and blank% < 90 at step 25,000. It is not
# close, and it is worse than every earlier from-scratch arm, including run_v5's
# 600 s updates. run_v6.sh named the fallback before the result was in:
# encoder-init. This is that run. No anneal of run_v6 can change the decision --
# bigbatch3, read at the end of its cosine (lr 1e-6), was still 98% blank at
# 6.681, so the best a perfectly annealed run_v6 could reach still fails both
# gate thresholds.
#
# What this run is for, stated plainly, because PRETRAINING_FROM_SCRATCH.md's
# Step 0 says to write it down: on-device size (110M vs the finetune's 0.6b) and
# the 2.39 tok/word tokenizer. NOT license independence -- the encoder is
# NVIDIA's parakeet, so that goal is forfeit by this route. NOT Nepali accuracy,
# which Step 0 rules out as a reason to pretrain at all.
#
# lr 5e-4 is the intersection of the two arms that worked: initenc descended at
# 1e-3 (240 s updates, random head), bigbatch3 beat run_v6 at the same 7,200 s
# update size with 5e-4. Run_v6's failure is the only point at 1e-3 + 7,200 s.
# Everything else is run_v6's config, so the numbers stay comparable.
#
# KILL CRITERION, decided now rather than after seeing the curve -- the
# discipline that made today's run_v6 call unambiguous. Score with:
#   python eval_probe.py <ckpt>.nemo --manifest .../hum_heldout.jsonl --n 200
# At step 5,000 (~10,000 h seen, ~2 h wallclock): loss/token < 5.5, blank% < 90,
# WER < 0.9. initenc reached 3.589 / 73.6% / 0.632 on 6 h at 240 s updates, so
# anything at scale that cannot beat those by step 25,000 means the encoder
# carries this and the "pretraining" framing is doing no work -- at which point
# the honest move is to stop and finetune parakeet-110m directly (same encoder,
# same data, ~2 days) rather than keep calling it a pretrain.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
DONOR=/workspace/hf_cache/models--nvidia--parakeet-tdt_ctc-110m/snapshots/431a349f3051ab85c22b9b7a2741b5fe77065665/parakeet-tdt_ctc-110m.nemo
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# Wait on run_v6 specifically. The obvious gate -- pgrep "pilot_train" -- never
# clears: the hung anneal_run_v6 process (PID 4143441, stuck since 19:33 on a
# FileNotFoundError at checkpoint save) is also a pilot_train process and will
# never exit on its own, so that gate waits forever and the next run silently
# never starts. Match the exact name instead; "anneal_run_v6 " does not contain
# the string "--name run_v6 ".
while pgrep -f -- "venvs/nemo/bin/python -u .*pilot_train.py.*--name run_v6 " >/dev/null; do sleep 30; done
log "=== run_v7_encinit: parakeet encoder + random CTC head, 7,200 s/update, 50,000 steps ==="
$PY -u "$HERE/pilot_train.py" \
  --train-manifest "$MAN/train_mix.jsonl" --val-manifest "$MAN/val_mix.jsonl" \
  --input-cfg "$MAN/train_input_cfg.yaml" \
  --init-encoder "$DONOR" --d-model 512 --n-layers 17 --n-heads 8 \
  --exp-dir "$ROOT/ckpt" --name run_v7_encinit \
  --max-steps 50000 --lr 5e-4 --warmup 2000 --val-every 2500 \
  --batch-duration 1200 --accum 6 \
  --max-duration 20 --workers 12 --grad-clip 1.0 --freq-masks 2 --time-masks 10 \
  > "$ROOT/logs/run_v7_encinit.log" 2>&1
log "run_v7_encinit exited rc=$?"
log "=== run_v7_encinit done ==="
