#!/bin/bash
# Full v5 synthetic build: plan -> Kyutai 6L voicing (3 GPU shards) -> Whisper QC
# -> assembly. Each stage is resumable. Starts from an EMPTY audio/raw because
# plan_v5.py re-maps job ids (see CLAUDE.md, v5 section).
set -uo pipefail
R=/root/tts/TTS_training/pocket_TTS
V=/workspace/v5_synth
ASR_VENV=/root/tts/TTS_training/synthetic_pipeline/asr_qc_service/.venv/bin/python3
say() { echo "[$(date '+%H:%M:%S')] $*"; }
export HF_HOME=/workspace/hf_cache HF_TOKEN=$(cat /root/.hf_milanakdj_token)

if [ -e $V/audio/raw ] && [ -n "$(ls -A $V/audio/raw 2>/dev/null)" ] && [ ! -e $V/audio/.plan_done ]; then
  say "FATAL: $V/audio/raw has audio from an unknown plan; move it aside first"; exit 1
fi
if [ ! -e $V/audio/.plan_done ]; then
  say "plan"; python3 $R/scripts/plan_v5.py || exit 1
  mkdir -p $V/audio/raw && touch $V/audio/.plan_done
fi

say "voicing with Kyutai 6L, 3 shards"
pids=()
for s in 0 1 2; do
  (cd $R/repo && SHARD=$s NSHARDS=3 .venv/bin/python3 ../scripts/gen_synth_v5.py) \
    > $V/gen_shard$s.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
grep -h "^shard" $V/gen_shard*.log
[ $rc -eq 0 ] || { say "FATAL: a generation shard failed, see $V/gen_shard*.log"; exit 1; }

say "Whisper QC + word alignment"
(cd $R/scripts && RAW=$V/audio/raw $ASR_VENV qc_align_v5.py 2>&1 | grep -E "clips to check|pass|QC DONE") || exit 1

say "assembly"
(cd $R/scripts && RAW=$V/audio/raw $R/repo/.venv/bin/python3 assemble_v5.py) || exit 1
say "V5 SYNTH DONE: $V/audio/train_v5_synth.jsonl"
