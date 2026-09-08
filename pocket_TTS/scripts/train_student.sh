#!/bin/bash
# Pocket TTS Nepali, stage 2: depth-distill the 24-layer teacher into a 6-layer
# student. Same d_model, so the flow head and non-backbone weights are copied
# from the teacher and the head stays frozen. distill_cfg_coef 2.0 bakes CFG in.
#
# Teacher: runs/nepali_teacher_24l/checkpoint_00200000.pt (200k steps, finished
# 2026-09-07 02:24, final valid loss -0.0739).
#
# Why this stage matters: the 24l teacher generates at 0.73x real-time on CPU
# (measured), so it is not deployable. The 6-layer student is the shippable model.
set -u
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export TOKENIZERS_PARALLELISM=false
cd /root/tts/TTS_training/pocket_TTS/repo
exec uv run training/train.py training/configs/nepali_distill.yaml
