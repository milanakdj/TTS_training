#!/bin/bash
# Run queued GPU work once the pocket-tts teacher releases the GPU.
# Guard uses a marker file, not pgrep on this script's own name (a pgrep pattern
# matching the wrapper shell has already self-killed two jobs today).
set -u
LOG=/workspace/milan_nepali_parler_ft/queue.log
echo "$(date -Is) queue armed; waiting for teacher to finish" >> $LOG
while pgrep -f "training/train.py" > /dev/null 2>&1; do sleep 120; done
echo "$(date -Is) teacher done -> starting parler gem+replay finetune" >> $LOG
/workspace/milan_nepali_parler_ft/scripts/train_gem_replay.sh > /workspace/milan_nepali_parler_ft/train_gemr.log 2>&1
echo "$(date -Is) gem+replay finetune exit=$?" >> $LOG
