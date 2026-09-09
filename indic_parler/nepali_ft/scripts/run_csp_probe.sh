#!/bin/bash
# Wait for the steering job to release its GPU memory, then run the CSP-FT probe.
# The pocket-tts teacher holds ~76 GB for another ~13 h; the probe runs batch=1 fp16
# and needs ~4 GB, so it fits in the remainder but not alongside a second job.
set -u
until ! pgrep -f e3_steer.py >/dev/null 2>&1; do sleep 60; done
sleep 20
cd /workspace/milan_nepali_parler_ft
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PER_EMO=150
exec /root/tts/TTS_training/synthetic_pipeline/tts_service/.venv/bin/python scripts/csp_probe.py
