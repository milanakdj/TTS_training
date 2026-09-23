#!/bin/bash
# Evaluate the v3 teacher + student on the SAME 100 held-out utterances the v2
# table was measured on (verified: all 100 are still in valid_v3 and none leaked
# into train_v3), so the two generations are directly comparable.
#
# This does NOT purge the v2 audio or results -- it adds the v3 systems to the
# same scorers and re-reports all five rows. Deleting wer_ft.json would destroy
# the only record of the shipped model's numbers.
set -uo pipefail
R=/root/tts/TTS_training/pocket_TTS
F=$R/infer/final
VENV=$R/repo/.venv/bin/python3
ASR_VENV=/root/tts/TTS_training/synthetic_pipeline/asr_qc_service/.venv/bin/python3
SIM_VENV=/root/tts/TTS_training/synthetic_pipeline/vc_service/.venv/bin/python3
say() { echo "[$(date '+%H:%M:%S')] $*"; }

[ -f "$F/eos_thresholds.json" ] || { echo "FATAL: eos_thresholds.json missing -- calibrate first"; exit 1; }
say "eos thresholds: $(cat $F/eos_thresholds.json | tr -d '\n ')"

# Stale audio from a different checkpoint silently entering results is the known
# way to get a wrong answer here. gen_final.py skips ids that already have a wav.
say "purging v3 eval audio only"
rm -rf $F/wav/teacher_24l_v3 $F/wav/student_6l_v3

say "generating 100 held-out utterances x 2 v3 models (CPU)"
(cd $R/repo && GEN_THREADS=16 OMP_NUM_THREADS=16 $VENV $F/gen_final.py teacher_24l_v3) 2>&1 | grep -E "==|FAIL|/100|DONE"
(cd $R/repo && GEN_THREADS=16 OMP_NUM_THREADS=16 $VENV $F/gen_final.py student_6l_v3) 2>&1 | grep -E "==|FAIL|/100|DONE"

say "CPU bench, all four models, threads pinned to 4"
(cd $R/repo && OMP_NUM_THREADS=4 BENCH_THREADS=4 $VENV $R/infer/bench_student.py) 2>&1 | grep -E "RT|==|DONE"
cp $R/infer/bench/results.json $F/bench.json 2>/dev/null

say "WER (base whisper large-v3-turbo) -- kept only to show it is unusable on Nepali"
(cd $F && $ASR_VENV $F/score_wer.py) 2>&1 | grep -E "==|DONE|Error|Traceback"
say "WER (Nepali-finetuned whisper, doubled-sot prefix) -- the real instrument"
(cd $F && $SIM_VENV $F/score_wer_ft.py) 2>&1 | grep -E "==|DONE|Error|Traceback"
say "speaker similarity (resemblyzer)"
(cd $F && $SIM_VENV $F/score_sim.py) 2>&1 | grep -E "==|DONE|Error|Traceback"

$VENV $F/summarize.py
say "ALL DONE"
