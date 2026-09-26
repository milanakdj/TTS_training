#!/bin/bash
# Pocket TTS v5, stage 2: 6-layer student, depth-distilled from the v5 bilingual teacher.
# Launched by scripts/queue_v5_student.sh once the teacher writes checkpoint_00250000.pt.
#
set -u
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /root/tts/TTS_training/pocket_TTS/repo
exec uv run training/train.py training/configs/nepali_distill_v5.yaml "$@"
