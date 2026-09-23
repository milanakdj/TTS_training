#!/bin/bash
# Release the GPU once the probe has produced everything Phase 2 needs.
#
# The probe exists to answer "is the graft sound?", and the Phase 2 arms only run
# to 7,500 steps -- so the probe's reference curve is only needed to 7,500 too.
# Running the remaining 12,500 steps would cost ~1.5 h of H100 and delay every
# queued arm behind it. Waits for the step-7500 CHECKPOINT (not just the valid
# line) so the run stays resumable if a longer reference is ever wanted.
set -u
PID=${PROBE_PID:-2752633}
L=/root/tts/TTS_training/pocket_TTS/probe_ne_graft.log
CK=/workspace/probe_ne_graft/checkpoint_00007500.pt
while kill -0 "$PID" 2>/dev/null; do
  if [ -f "$CK" ] && grep -q "valid @ step 7500" "$L" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] step-7500 checkpoint present; stopping probe (SIGTERM)"
    kill -TERM "$PID"
    sleep 20
    kill -0 "$PID" 2>/dev/null && { echo "still alive, SIGKILL"; kill -9 "$PID"; }
    echo "[$(date '+%H:%M:%S')] probe stopped; Phase 2 queue takes the GPU"
    exit 0
  fi
  sleep 30
done
echo "[$(date '+%H:%M:%S')] probe exited on its own before 7500"
