#!/usr/bin/env bash
# Proper from-scratch run, after the phase-3 pilot came back null.
#
# What the pilot established, and what changed because of it:
#   * the model CAN learn -- a single batch goes 1093 -> 12.6 in 300 steps, so
#     architecture, tokenizer, targets and loss are all sound;
#   * 11% of the mix was unalignable. The Premal transcripts are whole articles
#     against excerpt audio: median frames/tokens 0.69, 76% of rows with T < U.
#     CTC returns inf there and zero_infinity drops it silently. Now gated.
#   * the budget was the real error. 5,000 h seen is ~1 epoch; the doc's own
#     "2-3 days" for this phase implies ~300,000 h seen at our step rate. This
#     run is 100,000 h seen ~= 24 h wallclock ~= 24 epochs.
#   * lr is CosineAnnealing so the configured value is the value used; Noam
#     rescaled 2.5e-3 down to an effective 2.2e-6 and cost the first pilot.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "=== preflight ==="
$PY "$HERE/preflight.py" --manifests "$MAN" --max-duration 60 2>&1 | tee "$ROOT/logs/preflight_v4.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "ABORT: preflight failed"; exit 1; }

# lr 2.5e-3 diverged: train_loss climbed 380 -> 490 exactly as the schedule
# ramped to peak, then partly recovered as cosine decayed. 1e-3 is stable in
# every probe (clean-60 overfit 333 -> 184; full mix descends slowly).
# max_duration back to the doc's 20 s -- 60 s was only ever needed for the OOV
# corpus, and the T/U gate has removed most of that anyway.
log "=== training: 100,000 h seen, lr 1e-3, warmup 15k, max_duration 20 ==="
$PY -u "$HERE/pilot_train.py" \
  --train-manifest "$MAN/train_mix.jsonl" --val-manifest "$MAN/val_mix.jsonl" \
  --input-cfg "$MAN/train_input_cfg.yaml" --exp-dir "$ROOT/ckpt" --name run_v5 \
  --hours-seen 100000 --lr 1e-3 --warmup 15000 --val-every 5000 \
  --max-duration 20 --grad-clip 1.0 2>&1 | tee "$ROOT/logs/run_v5.log"
log "run_v5 exited rc=${PIPESTATUS[0]}"
log "=== DONE -- check val_wer trajectory before scaling to 120M ==="
