#!/bin/bash
# Opportunistically top up v2 references as Gemini quota frees up.
# gen_references_v2.py skips clips already on disk, so re-invoking is safe and
# incremental. Hard deadline: the reference set must be FROZEN before the
# production run starts, otherwise rows generated early and late in the run
# would be conditioned on different reference pools -- bad dataset provenance.
DEADLINE=$(( $(date +%s) + 5400 ))   # 90 minutes
GEN=/root/tts/TTS_training/synthetic_pipeline/emotions/gen_references_v2.py
PY=/tmp/claude-0/-root/eaa18e13-940b-4122-9ec2-5e290a5ccc89/scratchpad/gemvenv/bin/python
cd /root/tts/TTS_training/synthetic_pipeline/emotions
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  n=$(find v2 -name '*.wav' | wc -l)
  echo "=== topup attempt at $(date +%H:%M:%S), $n clips on disk ==="
  "$PY" -u "$GEN" --target-per-emotion 12 2>&1 | grep -vE "automatic function calling"
  n2=$(find v2 -name '*.wav' | wc -l)
  echo "=== attempt done, $n -> $n2 clips ==="
  if [ "$n2" -ge 55 ]; then echo "target reached, stopping topup"; break; fi
  sleep 600
done
echo "=== TOPUP FINISHED at $(date +%H:%M:%S), $(find v2 -name '*.wav' | wc -l) clips ==="
