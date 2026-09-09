#!/bin/bash
# Score all four (model x caption-set) cells once generation finishes.
set -u
cd /workspace/milan_nepali_parler_ft
while pgrep -f eval_gemr.py > /dev/null 2>&1; do sleep 30; done
V=/root/tts/TTS_training/synthetic_pipeline/tts_service/.venv/bin/python
for cell in base__amrita_best base__gem_style gemr__amrita_best gemr__gem_style; do
  echo "############ $cell ############"
  $V /root/tts/TTS_training/eval/ser_eval.py "eval_gemr/$cell" \
     --json-out "eval_gemr/ser_$cell.json" 2>&1 | grep -vE "^\s*$"
done
echo "ALL SCORED"
