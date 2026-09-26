#!/bin/bash
# 5 clips per cell (listening only) at 80k, then every 50k, plus final 250k.
# Checkpoints rotate (keep 3, ~2 h window), so poll every 5 min.
set -u
R=/root/tts/TTS_training/pocket_TTS; RUN=/workspace/v5_teacher_24l
PID=$(cat /tmp/v5_teacher.pid)
say() { echo "[$(date '+%F %H:%M')] $*"; }
for S in 80000 130000 180000 230000 250000; do
  C=$RUN/checkpoint_$(printf %08d $S).pt; T=$((S/1000))k
  until [ -f $C ]; do
    kill -0 $PID 2>/dev/null || { sleep 120; [ -f $C ] || { say "teacher gone, no $C; stopping"; break 2; }; }
    sleep 300
  done
  sleep 60
  rm -rf $R/listen/v5_teacher/$T
  (cd $R/repo && CKPT=$C TAG=$T .venv/bin/python3 $R/scripts/gen_listen_v5.py) 2>&1 | grep -vE "Warning|warn"
done
say "V5 TEACHER LISTEN LOOP DONE"
