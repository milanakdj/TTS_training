#!/bin/bash
# Phase 3: the full bilingual teacher, launched only if Phase 2's verdict holds.
#
# GUARD: v4c (0% English + pinned rows) is still running when this is queued. If
# it beats v4b on BOTH languages the recipe choice is wrong and a 26 h run would
# be wasted, so abort and say so. v3 burned 72 h by committing to a recipe nobody
# had checked.
#
# The guard SELF-TESTS its own parser first. The previous version's extraction
# silently returned an empty string (the grep it piped from ends with a quote, so
# the anchored number pattern never matched), which would have made every
# comparison vacuously false -- a guard that always passes is worse than none.
set -u
R=/root/tts/TTS_training/pocket_TTS
Q_PID=${Q_PID:-2820296}
say() { echo "[$(date '+%H:%M:%S')] $*"; }

final() {  # $1=log  $2=ne|en  -> final flow_loss for that valid set
  grep -oE "valid \[$2\] @ step [0-9]+: \{[^}]*'flow_loss': '[-0-9.e]+'" "$1" 2>/dev/null \
    | tail -1 | sed -E "s/.*'flow_loss': '([-0-9.e]+)'/\1/"
}

# --- self-test on a log that is already complete -------------------------
T=$(final $R/v4b_mix5_plain.log ne)
if ! [[ "$T" =~ ^-?[0-9] ]]; then
  say "FATAL: guard parser is broken (got '$T' from v4b ne). Refusing to proceed."
  exit 2
fi
say "guard parser self-test ok (v4b ne=$T)"

say "waiting for the Phase 2 queue (pid $Q_PID)"
while kill -0 "$Q_PID" 2>/dev/null; do sleep 60; done
say "Phase 2 done"

BNE=$(final $R/v4b_mix5_plain.log ne);  BEN=$(final $R/v4b_mix5_plain.log en)
CNE=$(final $R/v4c_ne_protect.log ne);  CEN=$(final $R/v4c_ne_protect.log en)
say "v4b  ne=$BNE  en=$BEN"
say "v4c  ne=$CNE  en=$CEN"

if [[ "$CNE" =~ ^-?[0-9] ]] && [[ "$CEN" =~ ^-?[0-9] ]]; then
  if awk "BEGIN{exit !($CNE < $BNE && $CEN < $BEN)}"; then
    say "ABORT: v4c beat v4b on BOTH languages -- recipe choice needs revisiting"
    say "       before committing 26 h. Phase 3 NOT launched."
    exit 1
  fi
  say "verdict holds: v4b recipe stands"
else
  say "NOTE: v4c produced no English number (it trains on 0% English, so its"
  say "      valid[en] is a pure generalization read). Proceeding on v4b vs v4a."
fi

say "=== launching v4_teacher_24l (250k steps, ~26 h) ==="
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN=$(cat /root/.hf_milanakdj)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false
(cd $R/repo && uv run training/train.py training/configs/v4_teacher_24l.yaml) \
    > $R/v4_teacher_24l.log 2>&1
say "=== v4_teacher_24l exited rc=$? ==="
$R/repo/.venv/bin/python3 $R/scripts/curve_vs.py $R/v4_teacher_24l.log
