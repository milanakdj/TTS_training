#!/bin/bash
# Guided 2x2 for the WiSE-FT blends of v5l (scripts/wise_ft_v5.py). Log: v5_wise_gen.log
R=/root/tts/TTS_training/pocket_TTS
export HF_HOME=/workspace/hf_cache HF_TOKEN=$(cat /root/.hf_milanakdj_token) CUDA_VISIBLE_DEVICES=0
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
cd $R/repo
for a in 25 50 75; do
  RUN=/workspace/v5l_probe_wise$a NAME=v5l_wise$a .venv/bin/python3 $R/infer/final/gen_2x2_guided.py 2>&1 | grep -E "step|->|DONE|Error|Traceback"
done
echo "WISE GEN DONE"
