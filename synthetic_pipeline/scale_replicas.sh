#!/bin/bash
# Add 4 more VC + 4 more QC replicas so the 100h emotion run isn't VC-bound.
# VC at ~3.1s/call over 4 replicas caps throughput at ~1.3 rows/s (12+ hours
# for ~57k rows). Eight replicas roughly doubles that. GPU has ample headroom:
# 20.5/80 GB used by the existing 4+4.
# Same launch pattern as ensure_running.sh so the pid files stay compatible.
set -u
ROOT=/root/tts/TTS_training/synthetic_pipeline
PIDDIR=$ROOT/pids

for p in 8042 8052 8062 8072; do
  cd "$ROOT/vc_service" || exit 1
  PORT=$p nohup .venv/bin/python server.py >> "server_$p.log" 2>&1 &
  echo $! > "$PIDDIR/vc_$p.pid"
  disown
  echo "started VC replica on $p (pid $(cat "$PIDDIR/vc_$p.pid"))"
done

for p in 8043 8053 8063 8073; do
  cd "$ROOT/asr_qc_service" || exit 1
  PORT=$p nohup .venv/bin/python server.py >> "server_$p.log" 2>&1 &
  echo $! > "$PIDDIR/qc_$p.pid"
  disown
  echo "started QC replica on $p (pid $(cat "$PIDDIR/qc_$p.pid"))"
done

echo "all 8 new replicas launched; models still loading"
