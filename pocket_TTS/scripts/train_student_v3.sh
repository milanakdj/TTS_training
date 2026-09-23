#!/bin/bash
# Pocket TTS v3, stage 2: 6-layer student, depth-distilled from the v3 teacher.
#
# Needs stage 1 finished: distill_teacher_weights must point at a local TRAINING
# checkpoint (builders.py torch.loads it and reads payload["ema"]). Set it in
# configs/nepali_distill_v3.yaml before launching.
set -u
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /root/tts/TTS_training/pocket_TTS/repo
exec uv run training/train.py training/configs/nepali_distill_v3.yaml "$@"
