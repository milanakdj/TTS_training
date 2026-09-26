#!/bin/bash
# Anchor-strength follow-up: v5m_lwf3 (lwf_weight 3.0) vs v5m_lwf (1.0). Log: v5_lwf3.log
set -uo pipefail
R=/root/tts/TTS_training/pocket_TTS; F=$R/infer/final; VENV=$R/repo/.venv/bin/python3
ASR_VENV=/root/tts/TTS_training/synthetic_pipeline/asr_qc_service/.venv/bin/python3
SIM_VENV=/root/tts/TTS_training/synthetic_pipeline/vc_service/.venv/bin/python3
say() { echo "[$(date '+%H:%M:%S')] $*"; }
export HF_HOME=/workspace/hf_cache HF_TOKEN=$(cat /root/.hf_milanakdj_token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
say "=== v5m_lwf3_probe ==="
(cd $R/repo && uv run training/train.py training/configs/v5m_lwf3_probe.yaml) > $R/v5m_lwf3_probe.log 2>&1
say "exited rc=$?"; grep -E "valid \[(ne|en)\] @" $R/v5m_lwf3_probe.log | sed 's/.*valid/valid/' | cut -c1-90
[ -f /workspace/v5m_lwf3_probe/checkpoint_00007500.pt ] || { say "no step-7500 checkpoint"; exit 1; }
(cd $R/repo && RUN=/workspace/v5m_lwf3_probe NAME=v5m_lwf3_7k5 $VENV $F/gen_2x2_guided.py) 2>&1 | grep -E "step|->|DONE"
(cd $F && $SIM_VENV score_2x2.py --part ne v5m_lwf3_7k5 2>&1 | grep -E "scored|DONE|Error")
(cd $F && $ASR_VENV score_2x2.py --part en v5m_lwf3_7k5 2>&1 | grep -E "scored|DONE|Error")
(cd $F && $SIM_VENV score_2x2.py --part merge kyutai_6l v5m_7k5 v5m_lwf_7k5 v5m_lwf3_7k5 2>&1 | tail -20)
$VENV $R/scripts/curve_vs.py $R/v5m_lwf3_probe.log 2>&1 | tail -3
say "LWF3 DONE"
