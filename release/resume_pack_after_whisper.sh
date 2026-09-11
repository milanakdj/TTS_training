#!/bin/bash
# Hold the reserve packaging until the whisper queue is done, then resume it.
#
# Why: the packer's 4 ffmpeg workers cost the whisper finetune ~7% throughput
# (53.6 -> 50.0 steps/min, measured) even at nice -n 19, because 8 of this box's
# 16 cores are permanently held by wekanode and the GPU job's dataloader is
# competing for what's left. Whisper has hours left; the upload has days, so the
# upload is the cheap side of the trade.
#
# Safe to kill at any point: pack_reserve.py resumes from state.json, and the
# shard counts in that file were verified against the Hub before the pause.
set -uo pipefail
WQ_PID=${1:?usage: resume_pack_after_whisper.sh <run_queue_pid>}
LOG=/workspace/hf_stage_reserve/pack6.log
V=/root/tts/TTS_training/synthetic_pipeline/tts_service/.venv/bin/python3

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

say "=== packer paused; waiting on whisper queue pid $WQ_PID ==="
while kill -0 "$WQ_PID" 2>/dev/null; do sleep 60; done
say "=== whisper queue exited; resuming packer ==="

cd /root/tts/TTS_training/release
PACK_WORKERS=4 nice -n 19 "$V" pack_reserve.py \
  --sources r3,r4,r2,r5,r1,r6 >> "$LOG" 2>&1
say "=== packer exited rc=$? ==="
