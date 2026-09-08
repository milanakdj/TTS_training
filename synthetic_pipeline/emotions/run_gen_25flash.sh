#!/bin/bash
# Generate the full reference set with gemini-2.5-flash-preview-tts, which still
# has quota after gemini-3.1-flash-tts-preview hit a hard daily cap at 20 clips.
# Retries in a loop because free-tier quota frees up in windows; the generator
# skips clips already on disk, so re-invoking is incremental.
set -u
cd /root/tts/TTS_training/synthetic_pipeline/emotions || exit 1
PY=/tmp/claude-0/-root/eaa18e13-940b-4122-9ec2-5e290a5ccc89/scratchpad/gemvenv/bin/python
DEADLINE=$(( $(date +%s) + 3600 ))   # freeze the set after 60 minutes
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  n=$(find v2_25flash -name '*.wav' 2>/dev/null | wc -l)
  echo "=== attempt $(date +%H:%M:%S): $n clips ==="
  "$PY" -u gen_references_v2.py --model gemini-2.5-flash-preview-tts \
        --target-per-emotion 12 2>&1 | grep -vE "automatic function calling"
  n2=$(find v2_25flash -name '*.wav' 2>/dev/null | wc -l)
  echo "=== attempt done: $n -> $n2 ==="
  [ "$n2" -ge 56 ] && { echo "target reached"; break; }
  [ "$n2" -eq "$n" ] && { echo "no progress this attempt; sleeping 300s"; sleep 300; }
done
echo "=== FROZEN at $(find v2_25flash -name '*.wav' 2>/dev/null | wc -l) clips, $(date +%H:%M:%S) ==="
