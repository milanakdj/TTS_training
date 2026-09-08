#!/bin/bash
# Regenerate ONLY the two emotions that failed screening of v2_25flash:
#   excited  1/6 passed -- clips came out at 281-372 Hz, gate window is 162-257 Hz
#   sad      0/6 passed -- clips were breathy/whispered (+19.3 dB hi/lo tilt vs -4.2 dB
#                          for neutral), so CAM++ identity fell to 0.43-0.53 vs 0.85
# Both prompts in gen_references_v2.py were rewritten for those two failure modes.
# Writes to a NEW subdir so old-prompt clips are never mixed in with the new ones.
# Needs GEMINI_KEYS="k1,k2,k3" in the environment; the generator skips clips already
# on disk, so re-invoking is incremental.
set -u
: "${GEMINI_KEYS:?export GEMINI_KEYS=\"k1,k2,k3\" first}"
cd /root/tts/TTS_training/synthetic_pipeline/emotions || exit 1
PY=/tmp/claude-0/-root/eaa18e13-940b-4122-9ec2-5e290a5ccc89/scratchpad/gemvenv/bin/python
SUB=v3_fix
DEADLINE=$(( $(date +%s) + 3600 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  n=$(find "$SUB" -name '*.wav' 2>/dev/null | wc -l)
  echo "=== attempt $(date +%H:%M:%S): $n clips ==="
  "$PY" -u gen_references_v2.py --model gemini-2.5-flash-preview-tts \
        --emotions sad,excited --out-subdir "$SUB" --target-per-emotion 12 \
        2>&1 | grep -vE "automatic function calling"
  n2=$(find "$SUB" -name '*.wav' 2>/dev/null | wc -l)
  echo "=== attempt done: $n -> $n2 ==="
  [ "$n2" -ge 24 ] && { echo "target reached"; break; }
  [ "$n2" -eq "$n" ] && { echo "no progress this attempt; sleeping 300s"; sleep 300; }
done
echo "=== FROZEN at $(find "$SUB" -name '*.wav' 2>/dev/null | wc -l) clips, $(date +%H:%M:%S) ==="
echo "next: cd .. && REF_SET_DIR=$SUB vc_service/.venv/bin/python emotions/screen_references.py \\"
echo "        --neutral-dir emotions/v2_25flash/neutral \\"
echo "        --out manifests/reference_pool_emotion_v3_fix.parquet"
