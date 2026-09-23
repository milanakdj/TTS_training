#!/bin/bash
# Phase 2 queue: the three arms, back to back. One H100, so strictly sequential.
#
# PREFLIGHT exists because the first attempt lost the v4b and v4a slots to a
# broken manifest: `latents_file` in a latents manifest is RELATIVE to the
# manifest's own directory, so writing the mix to /workspace orphaned every row.
# The loader logged 131k "skipping unreadable sample" lines, killed its worker
# threads, and left the main process wedged at 0 steps with the GPU idle -- it
# does not exit, so the queue would have sat there. Check before launching.
set -u
R=/root/tts/TTS_training/pocket_TTS
say() { echo "[$(date '+%H:%M:%S')] $*"; }

preflight() {  # $1 = config name
  local cfg=$R/configs/$1.yaml
  local train
  train=$(grep -E "^\s+train_jsonl:" "$cfg" | awk '{print $2}')
  $R/repo/.venv/bin/python3 - "$train" <<'PY'
import json, os, sys
m = sys.argv[1]
if not os.path.exists(m):
    sys.exit(f"PREFLIGHT FAIL: train_jsonl missing: {m}")
base = os.path.dirname(m)
with open(m) as f:
    rows = [json.loads(next(f)) for _ in range(200)]
missing = [r["latents_file"] for r in rows
           if "latents_file" in r and not os.path.exists(os.path.join(base, r["latents_file"]))]
if missing:
    sys.exit(f"PREFLIGHT FAIL: {len(missing)}/200 latents unresolvable, e.g. {missing[0]}")
print(f"preflight ok: {m} ({len(rows)} rows checked)")
PY
}

for arm in v4b_mix5_plain v4a_mix5_protect v4c_ne_protect; do
  say "=== preflight $arm ==="
  if ! preflight "$arm"; then say "SKIPPING $arm (preflight failed)"; continue; fi
  say "=== launching $arm ==="
  unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
  export CUDA_VISIBLE_DEVICES=0
  export HF_TOKEN=$(cat /root/.hf_milanakdj)
  export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
  export HF_HOME=/workspace/hf_cache
  export TOKENIZERS_PARALLELISM=false
  (cd $R/repo && uv run training/train.py training/configs/$arm.yaml) > $R/$arm.log 2>&1
  say "=== $arm exited rc=$? ==="
  if ! grep -q "INFO train] step" $R/$arm.log; then
    say "WARNING: $arm produced ZERO training steps -- check $arm.log"
  fi
  grep -E "valid \[ne\] @|valid \[en\] @" $R/$arm.log | tail -6
done
say "PHASE 2 QUEUE DONE"
$R/repo/.venv/bin/python3 $R/scripts/curve_vs.py \
  $R/v4b_mix5_plain.log $R/v4a_mix5_protect.log $R/v4c_ne_protect.log
