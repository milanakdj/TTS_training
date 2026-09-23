#!/bin/bash
# Pocket TTS v3, stage 1: 24-layer teacher, Nepali + Indian English, grafted tokenizer.
#
# Unlike v1/v2 this does NOT reset the text embedding. tokenizer_v3/ne_en_9682
# extends Kyutai's vocabulary rather than replacing it, so ids 0..3999 still mean
# what the pretrained weights think they mean: graft_text_embedding=4000 keeps
# those rows and freshly initializes only the appended Nepali ones. English
# therefore survives as inherited weights instead of being relearned from replay.
set -u
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /root/tts/TTS_training/pocket_TTS/repo
exec uv run training/train.py training/configs/nepali_finetune_v3.yaml "$@"
