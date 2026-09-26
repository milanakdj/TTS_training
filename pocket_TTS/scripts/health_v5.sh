#!/bin/bash
# Compact health report for the v5 teacher run. Safe to run any time.
set -u
R=/root/tts/TTS_training/pocket_TTS
L=$R/v5_teacher_24l.log
PID=$(cat /tmp/v5_teacher.pid 2>/dev/null || echo 0)

echo "===== v5 teacher health @ $(date '+%Y-%m-%d %H:%M') ====="
if kill -0 "$PID" 2>/dev/null; then echo "process : ALIVE (pid $PID)"
else echo "process : *** NOT RUNNING *** (pid $PID)"; fi

last=$(grep -oE "INFO train\] step [0-9]+ \| lr [^|]+\| grad [^|]+\| [0-9.]+ it/s" "$L" 2>/dev/null | tail -1)
step=$(echo "$last" | grep -oE "step [0-9]+" | grep -oE "[0-9]+")
rate=$(echo "$last" | grep -oE "[0-9.]+ it/s" | grep -oE "[0-9.]+")
echo "step    : ${step:-none} / 250000   (${rate:-?} it/s)"
if [ -n "${step:-}" ] && [ -n "${rate:-}" ]; then
  awk -v s="$step" -v r="$rate" 'BEGIN{
    left=(250000-s)/r; printf "eta     : %.1f h remaining (%s)\n", left/3600,
      strftime("%m-%d %H:%M", systime()+left)}'
fi

# Is it actually progressing, or wedged with the GPU idle? (v4b failed exactly
# this way: alive, 0 steps, GPU 0%.)
m1=$(stat -c %Y "$L" 2>/dev/null || echo 0); now=$(date +%s)
echo "log age : $(( (now - m1) ))s since last write"
[ $(( now - m1 )) -gt 600 ] && echo "WARNING : log stale >10 min -- possibly wedged"

echo "gpu     : $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
echo "disk /  : $(df -h / | awk 'NR==2{print $4" free ("$5" used)"}')"
echo "ckpts   : $(ls /workspace/v5_teacher_24l/checkpoint_*.pt 2>/dev/null | wc -l) kept"

errs=$(grep -cE "Traceback|CUDA out of memory|ValueError|Killed" "$L" 2>/dev/null); errs=${errs:-0}
echo "errors  : $errs"
[ "$errs" != "0" ] && grep -E "Traceback|CUDA out of memory|ValueError|Killed" "$L" | tail -3

echo "--- Nepali vs v2/v3 (lower better) ---"
$R/repo/.venv/bin/python3 $R/scripts/curve_vs.py "$L" 2>/dev/null | tail -12
echo "--- English (no v2/v3 baseline exists: both are Nepali-only) ---"
SET=en NOBASE=1 $R/repo/.venv/bin/python3 $R/scripts/curve_vs.py "$L" 2>/dev/null | sed -n '2,12p'
