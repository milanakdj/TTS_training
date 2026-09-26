#!/bin/bash
# Health check every 6 h until the v5 teacher stops, then one final report.
# Appends to health_v5.log so the whole history is in one place.
set -u
R=/root/tts/TTS_training/pocket_TTS
OUT=$R/health_v5.log
PID=$(cat /tmp/v5_teacher.pid 2>/dev/null || echo 0)
while true; do
  bash $R/scripts/health_v5.sh >> "$OUT" 2>&1
  echo "" >> "$OUT"
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[$(date '+%F %H:%M')] teacher process gone -- final report above, loop ends" >> "$OUT"
    exit 0
  fi
  sleep 21600   # 6 h
done
