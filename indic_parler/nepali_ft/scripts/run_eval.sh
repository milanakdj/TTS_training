#!/bin/bash
# Before/after on the SAME captions: base indic-parler vs the Nepali-Rasa finetune.
# Scored with emotion2vec (a speech-emotion-recognition model), not MFCC/F0 spread --
# "louder and higher-pitched" scores as separated on MFCC while sounding nothing
# like anger, which is exactly the failure the user heard.
set -u
P=/workspace/milan_nepali_parler_ft
V=/root/tts/TTS_training/synthetic_pipeline/tts_service/.venv/bin/python
SER=/tmp/claude-0/-root/065354ff-5600-43b1-ae0d-5441bda8b778/scratchpad/ser/bin/python
CKPT=${1:-$P/out}
export HF_TOKEN=$(cat /root/.hf_milanakdj)

for spec in "base:ai4bharat/indic-parler-tts" "ft:$CKPT"; do
  tag=${spec%%:*}; model=${spec#*:}
  echo "=== generating $tag from $model ==="
  CUDA_VISIBLE_DEVICES=0 $V $P/scripts/gen_compare.py --model "$model" --tag "$tag" --speaker Srijana --n 8 \
    2>&1 | grep -viE "flash attention|warn" | tail -3
done

for tag in base ft; do
  echo; echo "############ SER: $tag ############"
  $SER /root/tts/TTS_training/eval/ser_eval.py $P/samples/$tag --json-out $P/samples/ser_$tag.json 2>&1 \
    | grep -viE "^ *[0-9]+%|it/s\]|rtf_avg|INFO|Loading ckpt|init param|miss key|^ *$" | grep -A12 "intended"
done
