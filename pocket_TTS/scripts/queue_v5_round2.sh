#!/bin/bash
# v5 round 2: can we close the gap to untouched Kyutai? Four levers, each judged at
# 7.5k steps on the guided 2x2 (English vs Kyutai 6L) and v2's curve (Nepali):
#   #1 WiSE-FT blends of v5l with Kyutai  (scripts/wise_ft_v5.py + gen_wise_v5.sh, run separately)
#   #5 v5f      v5l mix, top 12 layers + heads frozen at Kyutai
#   #4 v5m      ~15% English (LibriHeavy round 1+2 + cross clips)
#   #3 v5m_lwf  v5m + LwF anchor to frozen Kyutai 24L (lwf_weight 1.0)
# One H100, strictly sequential training. Log: v5_round2.log
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
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

arm() {  # $1 config/run name, $2 2x2 NAME
  say "=== $1 ==="
  (cd $R/repo && uv run training/train.py training/configs/$1.yaml) > $R/$1.log 2>&1
  say "$1 exited rc=$?"; grep -E "valid \[(ne|en)\] @" $R/$1.log | sed 's/.*valid/valid/' | cut -c1-90
  [ -f /workspace/$1/checkpoint_00007500.pt ] || { say "$1: no step-7500 checkpoint, skipping eval"; return 1; }
  (cd $R/repo && RUN=/workspace/$1 NAME=$2 $VENV $F/gen_2x2_guided.py) 2>&1 | grep -E "step|->|DONE"
  NEW="$NEW $2"
}
NEW=""

arm v5f_probe v5f_7k5

say "waiting for LibriHeavy round 2 (4 shards)"
until [ $(cat $V/libriheavy_b/prep.*.log 2>/dev/null | grep -c "SHARD DONE") -eq 4 ]; do
  grep -l Traceback $V/libriheavy_b/prep.*.log 2>/dev/null && { say "FATAL in LibriHeavy prep"; exit 1; }
  sleep 60
done
grep -h "SHARD DONE" $V/libriheavy_b/prep.*.log
M=$V/libriheavy_b/train_v5_libriheavy_b.jsonl
cat $V/libriheavy_b/train_v5_libriheavy_b.[0-3].jsonl > $M
if [ ! -f ${M%.jsonl}_latents.jsonl ]; then
  say "precomputing latents for $(wc -l < $M) clips"
  cfg=${M%.jsonl}.latcfg.yaml
  sed "s#train_jsonl: .*#train_jsonl: $M#" $R/configs/v5s_probe.yaml > $cfg
  (cd $R/repo && $VENV -m training.scripts.precompute_latents $cfg) 2>&1 | grep -E "wrote|stitch|Error|Traceback"
  [ -f ${M%.jsonl}_latents.jsonl ] || { say "FATAL: no latents manifest for $M"; exit 1; }
fi
$VENV $R/scripts/build_mix_v5m.py || exit 1

arm v5m_probe v5m_7k5
arm v5m_lwf_probe v5m_lwf_7k5

say "waiting for WiSE-FT generation"
until grep -q "WISE GEN DONE" $R/v5_wise_gen.log; do sleep 60; done
NEW="$NEW v5l_wise25 v5l_wise50 v5l_wise75"
ALL="kyutai_6l student_6l_v4 v5l_7k5 $NEW"
say "scoring new: $NEW"
(cd $F && $SIM_VENV score_2x2.py --part ne $NEW 2>&1 | grep -E "scored|DONE|Error")
(cd $F && $ASR_VENV score_2x2.py --part en $NEW 2>&1 | grep -E "scored|DONE|Error")
(cd $F && $SIM_VENV score_2x2.py --part merge $ALL 2>&1 | tail -40)
say "Nepali curves vs v2 at matched steps"
for a in v5f_probe v5m_probe v5m_lwf_probe; do [ -f $R/$a.log ] && $VENV $R/scripts/curve_vs.py $R/$a.log 2>&1 | tail -3; done
say "V5 ROUND 2 DONE"
