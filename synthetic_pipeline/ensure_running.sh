#!/bin/bash
# Idempotent self-heal: restarts any of the 9 pipeline services (or the
# orchestrator) that have actually died, using PID files rather than an HTTP
# health check as the liveness signal -- a busy service under real production
# load can take much longer than any reasonable curl timeout to answer
# /health, which caused false-positive "it's dead" restarts here before.
# Since each service loads a multi-GB model onto GPU *before* binding its
# port, a false restart is wasteful (loads a model, then fails to bind and
# exits) even though it's not actually destructive -- this version avoids
# that entirely by checking real process liveness first.
#
# Safe to run repeatedly (via cron). The orchestrator resumes from its
# checkpoint files, so restarting it after a crash/reboot/power-cut never
# redoes or loses completed work.

set -u
ROOT=/root/tts/TTS_training/synthetic_pipeline
PIDDIR=$ROOT/pids
LOG=$ROOT/ensure_running.log
[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a
mkdir -p "$PIDDIR"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG"; }

is_alive() {
  # $1 = pidfile path
  [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null
}

start_tts() {
  cd "$ROOT/tts_service" && nohup .venv/bin/python server.py >> server.log 2>&1 &
  echo $! > "$PIDDIR/tts_8001.pid"
  disown
  log "started TTS on 8001 (pid $!)"
}

start_vc() {
  local port=$1
  cd "$ROOT/vc_service" && PORT=$port nohup .venv/bin/python server.py >> server_$port.log 2>&1 &
  echo $! > "$PIDDIR/vc_$port.pid"
  disown
  log "started VC replica on $port (pid $!)"
}

start_qc() {
  local port=$1
  cd "$ROOT/asr_qc_service" && PORT=$port nohup .venv/bin/python server.py >> server_$port.log 2>&1 &
  echo $! > "$PIDDIR/qc_$port.pid"
  disown
  log "started QC replica on $port (pid $!)"
}

is_alive "$PIDDIR/tts_8001.pid" || start_tts
for p in 8002 8012 8022 8032; do is_alive "$PIDDIR/vc_$p.pid" || start_vc $p; done
for p in 8003 8013 8023 8033; do is_alive "$PIDDIR/qc_$p.pid" || start_qc $p; done

if ! pgrep -f "run_production.py" >/dev/null 2>&1; then
  cd "$ROOT/orchestrator"
  nohup .venv/bin/python -u run_production.py >> production.log 2>&1 &
  disown
  log "orchestrator was not running -- restarted (resumes from checkpoint), pid $!"
fi
