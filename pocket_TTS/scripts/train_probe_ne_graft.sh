#!/bin/bash
# PHASE 1 PROBE: Nepali-only on the v3 grafted tokenizer, 20k steps.
# Isolates "is the graft sound?" from "was the recipe wrong?" -- see the header
# of configs/probe_ne_graft.yaml for the v2/v3 numbers to read it against.
#
# NOTE: configs/ and repo/training/configs/ are two SEPARATE copies, not symlinks.
# Edit both or the run silently uses the stale one.
set -u
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /root/tts/TTS_training/pocket_TTS/repo
exec uv run training/train.py training/configs/probe_ne_graft.yaml "$@"
