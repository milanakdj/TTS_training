#!/bin/bash
# v5 probe queue: wait for the synthetic + LibriHeavy builds, precompute latents,
# build the S/L/X mixes, then per arm: preflight -> 7.5k steps -> guided 2x2.
# Finally one scored table with v4b (old recipe, same step), untouched Kyutai 6L
# and the v4 student. One H100, so strictly sequential. Log: v5_probes.log
set -uo pipefail
R=/root/tts/TTS_training/pocket_TTS
V=/workspace/v5_synth
F=$R/infer/final
VENV=$R/repo/.venv/bin/python3
ASR_VENV=/root/tts/TTS_training/synthetic_pipeline/asr_qc_service/.venv/bin/python3
SIM_VENV=/root/tts/TTS_training/synthetic_pipeline/vc_service/.venv/bin/python3
say() { echo "[$(date '+%H:%M:%S')] $*"; }
export HF_HOME=/workspace/hf_cache HF_TOKEN=$(cat /root/.hf_milanakdj_token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
# A stray RANK/LOCAL_RANK in the launching shell makes train.py take the
# torch.distributed path and die on the missing WORLD_SIZE (v5s, 2026-09-24).
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

wait_for() {  # $1 log, $2 success marker
  until grep -q "$2" "$1" 2>/dev/null; do
    grep -qE "FATAL|Traceback" "$1" 2>/dev/null && { say "FATAL in $1"; tail -5 "$1"; exit 1; }
    sleep 30
  done
}
say "waiting for synthetic build and LibriHeavy prep"
wait_for $V/run.log "V5 SYNTH DONE"
wait_for $V/libriheavy_prep.log "LIBRIHEAVY DONE"
tail -1 $V/run.log; tail -1 $V/libriheavy_prep.log

say "precomputing latents"
for m in $V/audio/train_v5_synth.jsonl $V/libriheavy/train_v5_libriheavy.jsonl; do
  [ -f ${m%.jsonl}_latents.jsonl ] && { echo "latents exist for $(basename $m)"; continue; }
  cfg=${m%.jsonl}.latcfg.yaml
  sed "s#train_jsonl: .*#train_jsonl: $m#" $R/configs/v5s_probe.yaml > $cfg
  (cd $R/repo && $VENV -m training.scripts.precompute_latents $cfg) 2>&1 | grep -E "wrote|stitch|Error|Traceback" || exit 1
  [ -f ${m%.jsonl}_latents.jsonl ] || { say "FATAL: no latents manifest for $m"; exit 1; }
done

say "building mixes"; $VENV $R/scripts/build_mix_v5.py || exit 1

preflight() {  # every English row + 200 Nepali rows must resolve
  $VENV - "$1" <<'PY'
import json, os, sys
m = sys.argv[1]; base = os.path.dirname(m); bad = n = 0
for i, l in enumerate(open(m)):
    d = json.loads(l)
    if i < 200 or d.get("language") != "ne":
        n += 1; bad += not os.path.exists(os.path.join(base, d["latents_file"]))
sys.exit(f"PREFLIGHT FAIL: {bad}/{n} unresolvable in {m}") if bad else print(f"preflight ok: {n} rows checked")
PY
}

names="v4b_7k5 kyutai_6l student_6l_v4"
for a in s l x; do
  say "=== arm v5$a ==="
  preflight /workspace/manifests_v4/train_v5${a}_latents.jsonl || { say "SKIP v5$a"; continue; }
  (cd $R/repo && uv run training/train.py training/configs/v5${a}_probe.yaml) > $R/v5${a}_probe.log 2>&1
  say "v5$a exited rc=$?"; grep -E "valid \[(ne|en)\] @" $R/v5${a}_probe.log | sed 's/.*valid/valid/' | cut -c1-90
  [ -f /workspace/v5${a}_probe/checkpoint_00007500.pt ] || { say "v5$a: no step-7500 checkpoint, skipping eval"; continue; }
  (cd $R/repo && RUN=/workspace/v5${a}_probe NAME=v5${a}_7k5 $VENV $F/gen_2x2_guided.py) 2>&1 | grep -E "step|->|DONE"
  names="$names v5${a}_7k5"
done

say "scoring: $names"
(cd $F && $SIM_VENV score_2x2.py --part ne $names 2>&1 | grep -E "scored|DONE|Error")
(cd $F && $ASR_VENV score_2x2.py --part en $names 2>&1 | grep -E "scored|DONE|Error")
(cd $F && $SIM_VENV score_2x2.py --part merge $names 2>&1 | tail -30)
say "Nepali curves vs v2 at matched steps"
for a in s l x; do [ -f $R/v5${a}_probe.log ] && $VENV $R/scripts/curve_vs.py $R/v5${a}_probe.log 2>&1 | tail -6; done
say "V5 PROBES DONE"
