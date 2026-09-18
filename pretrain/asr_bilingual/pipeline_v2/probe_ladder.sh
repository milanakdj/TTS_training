#!/usr/bin/env bash
# How many distinct utterances can this recipe actually fit?
#
# mem64 is a real pass, not an artefact: rotating the audio against the text
# takes loss/token from 1.53 to 10.80, and the encoder's pooled outputs for
# different clips sit at cosine 0.73, so it is reading the speech. But 64 and
# 2,000 differ in three ways at once -- count, duration spread and label source.
# This ladder holds the population fixed (human-labelled Nepali, 4-9 s, the same
# pool mem64 was drawn from) and moves only the count, at an identical step
# budget so every arm gets the same number of updates and the same utterance
# visits.
#
#   64 -> 512 -> 2,000 distinct utterances, 4,000 steps each.
#
# If the fit degrades smoothly with count, this is an optimisation problem and
# the lr sweep is where the answer is. If it falls off a cliff between 64 and
# 512, something about the recipe stops it generalising past memorisation.
set -u
PY=/workspace/venvs/nemo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
while pgrep -f "^/workspace/venvs/nemo/bin/python -u .*pilot_train" >/dev/null; do sleep 30; done

for set in mem64 hum512 hum2k; do
  score="$MAN/${set}_score.jsonl"; [ "$set" = mem64 ] && score="$MAN/mem64.jsonl"
  log "=== arm ladder_$set ==="
  $PY -u "$HERE/pilot_train.py" \
    --train-manifest "$MAN/$set.jsonl" --val-manifest "$score" \
    --exp-dir "$ROOT/ckpt_probe" --name "ladder_$set" \
    --max-steps 4000 --lr 1e-3 --warmup 400 --val-every 2000 \
    --batch-duration 240 --max-duration 20 --workers 8 \
    --grad-clip 0.0 --freq-masks 0 --time-masks 0 \
    > "$ROOT/logs/probe_ladder_$set.log" 2>&1
  log "arm ladder_$set exited rc=$?"
done
log "=== ladder done ==="
