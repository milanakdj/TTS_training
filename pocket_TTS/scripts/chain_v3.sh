#!/bin/bash
# v3 stage 1 -> stage 2 handoff, unattended.
#
# Gate on the CHECKPOINT FILE, not on process absence. The obvious gate --
# "wait until no train.py is running" -- is the one that burned run_v7: a hung
# predecessor (anneal_run_v6, stuck 3 days on a FileNotFoundError at checkpoint
# save) never exits, so a process-absence gate waits forever and the next stage
# silently never starts. A file that only exists when stage 1 genuinely finished
# cannot lie in that direction.
set -u
TEACHER_CKPT=/workspace/nepali_teacher_24l_v3/checkpoint_00200000.pt
HERE=/root/tts/TTS_training/pocket_TTS
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "=== chain_v3: waiting for stage 1 to produce $TEACHER_CKPT ==="
while [ ! -f "$TEACHER_CKPT" ]; do
  # If stage 1 has died without producing the checkpoint, stop rather than spin.
  if ! pgrep -f "training/configs/nepali_finetune_v3.yaml" >/dev/null; then
    sleep 60   # tolerate a brief gap between launch and process registration
    if ! pgrep -f "training/configs/nepali_finetune_v3.yaml" >/dev/null \
       && [ ! -f "$TEACHER_CKPT" ]; then
      log "ABORT: stage 1 is not running and produced no final checkpoint."
      log "       check $HERE/train_teacher_v3.log"
      exit 1
    fi
  fi
  sleep 120
done

# The trainer writes the file before it finishes flushing; let it settle.
log "stage 1 checkpoint present; waiting 120s for the write to settle"
sleep 120
log "=== chain_v3: launching stage 2 (student 6L distill) ==="
cd "$HERE"
bash scripts/train_student_v3.sh > "$HERE/train_student_v3.log" 2>&1
log "stage 2 exited rc=$?"
log "=== chain_v3 done ==="
