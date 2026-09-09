#!/bin/bash
# Start pocket-TTS stage 2 (student distill) once the parler run releases the GPU.
# Waits on the exact PID, not a pgrep pattern: a pattern matching this wrapper
# shell has already self-killed jobs in this project.
set -u
LOG=/workspace/milan_nepali_parler_ft/queue.log
WAIT_PID=${1:?usage: queue_student.sh <pid-to-wait-for>}
echo "$(date -Is) student queue armed; waiting for parler pid $WAIT_PID" >> $LOG
while [ -d /proc/$WAIT_PID ]; do sleep 60; done
echo "$(date -Is) parler released GPU -> starting pocket-TTS student distill" >> $LOG
/root/tts/TTS_training/pocket_TTS/scripts/train_student.sh \
  > /root/tts/TTS_training/pocket_TTS/train_student.log 2>&1
echo "$(date -Is) student distill exited rc=$?" >> $LOG
