#!/bin/bash
# Pocket TTS Nepali, stage 1: 24-layer teacher, finetuned from the English release
# with a fresh text embedding (our Nepali tokenizer is not the one those weights
# were trained with). 2,215.7 h / 749,610 clips / 100% word-aligned.
set -u
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export TOKENIZERS_PARALLELISM=false
cd /root/tts/TTS_training/pocket_TTS/repo
exec uv run training/train.py training/configs/nepali_finetune.yaml
