#!/usr/bin/env bash
# Queue the from-scratch bilingual pretrain behind the running Seed-VC job.
#
# Scope, deliberately: this runs phases 0-3 of PRETRAINING_FROM_SCRATCH.md and
# STOPS at the pilot. Step 9 calls phase 3 "the real decision point ... do not
# skip it and do not scale past it on faith", so the 120M hybrid is not launched
# unattended. When the pilot lands, its numbers decide whether phase 4 starts.
#
#   nohup setsid bash queue.sh > /workspace/asr_pretrain_v2/queue.log 2>&1 &

set -u
PY="/workspace/venvs/nemo/bin/python -u"   # unbuffered: a queue log that only appears at exit is useless
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/asr_pretrain_v2
MAN=$ROOT/manifests
mkdir -p "$ROOT"/{manifests,logs,ckpt}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { log "ABORT: $*"; exit 1; }

# ---------------------------------------------------------------- 0. wait
log "=== waiting for Seed-VC to finish ==="
while pgrep -f "vc_run.py --exclude-core" >/dev/null; do sleep 120; done
log "Seed-VC process has exited"

VCM=/workspace/oov_distill/out/vc_all/vc_manifest.jsonl
ROWS=$(wc -l < "$VCM" 2>/dev/null || echo 0)
log "vc_manifest rows: $ROWS / 25338"
if [ "$ROWS" -lt 24000 ]; then
  # Do not quietly pretrain on a truncated corpus -- say which it is.
  log "WARNING: Seed-VC ended early ($ROWS rows). Continuing with what exists;"
  log "         the mix report will show the real oov_vc hours."
fi

# ---------------------------------------------------------------- 1. tokenizer
# NeMo resolves tokenizer.dir/{tokenizer.model,tokenizer.vocab}; the BPE is named
# ne_en_bpe.*, so link rather than copy.
# NeMo resolves tokenizer.dir/{tokenizer.model,tokenizer.vocab,vocab.txt}. The
# dir is prebuilt; vocab.txt is generated from the SPE model because the original
# ne_en_bpe drop did not ship one. Validate rather than rebuild -- an earlier
# version of this block symlinked over the real files and broke the dir.
TOK=/workspace/asr_pretrain_v2/tokenizer
for f in tokenizer.model tokenizer.vocab vocab.txt; do
  [ -s "$TOK/$f" ] || die "tokenizer dir incomplete: $TOK/$f missing"
done
log "tokenizer ok: $TOK ($(wc -l < "$TOK/vocab.txt") pieces)"

# ---------------------------------------------------------------- 2. manifests
log "=== building manifests ==="
$PY "$HERE/build_manifests.py" --out "$MAN" 2>&1 | tee "$ROOT/logs/manifests.log"
[ -s "$MAN/train_mix.jsonl" ] || die "no train_mix.jsonl"

# ---------------------------------------------------------------- 3. preflight
# Precompute the lhotse bucket boundaries. Without them DynamicBucketingSampler
# estimates bins by scanning the multiplexed sources and does not finish.
log "=== bucket bins ==="
$PY "$HERE/make_bins.py" --manifest "$MAN/train_mix.jsonl" --max-duration 60 \
    2>&1 | tee "$ROOT/logs/bins.log"

log "=== preflight ==="
$PY "$HERE/preflight.py" --manifests "$MAN" --max-duration 60 2>&1 | tee "$ROOT/logs/preflight.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || die "preflight failed -- see logs/preflight.log"

# ---------------------------------------------------------------- 4. dl ceiling
# Step 1: "this single measurement decides whether the whole plan is 3 weeks or
# 3 months, and it has never been run."
log "=== dataloader ceiling (untarred) ==="
for W in 8 16; do
  $PY "$HERE/bench_dl_v2.py" --manifest "$MAN/train_mix.jsonl" --workers $W --max-duration 60 \
      2>&1 | tee -a "$ROOT/logs/bench_dl_untarred.log"
done

# ---------------------------------------------------------------- 5. shards
log "=== resample + tarred shards ==="
$PY "$HERE/make_shards.py" --manifest "$MAN/train_mix.jsonl" \
    --out "$ROOT/tarred" --max-duration 60 2>&1 | tee "$ROOT/logs/shards.log"
SHARD_RC=${PIPESTATUS[0]}
TARRED=""
if [ "$SHARD_RC" -eq 0 ] && [ -s "$ROOT/tarred/tarred_audio_manifest.json" ]; then
  TARRED="$ROOT/tarred/audio__OP_0..$(( $(ls "$ROOT"/tarred/audio_*.tar 2>/dev/null | wc -l) - 1 ))_CL_.tar"
  log "tarred shards ready: $TARRED"
  log "=== dataloader ceiling (tarred) ==="
  $PY "$HERE/bench_dl_v2.py" --manifest "$ROOT/tarred/tarred_audio_manifest.json" \
      --workers 16 --tarred "$TARRED" 2>&1 | tee "$ROOT/logs/bench_dl_tarred.log"
else
  log "shard packing unavailable (rc=$SHARD_RC) -- pilot runs off plain files."
  log "That is slower, not wrong; bench_dl_untarred.log has the ceiling it pays."
fi

# ---------------------------------------------------------------- 6. pilot
log "=== phase-3 pilot: 32M CTC, 5,000 h seen ==="
$PY "$HERE/pilot_train.py" --params-only 2>&1 | tee "$ROOT/logs/pilot_params.log"
ARGS=(--train-manifest "$MAN/train_mix.jsonl" --val-manifest "$MAN/val_mix.jsonl"
      --exp-dir "$ROOT/ckpt" --hours-seen 5000 --input-cfg "$MAN/train_input_cfg.yaml")
if [ -n "$TARRED" ]; then
  # The weighted input_cfg and a single tarred manifest are mutually exclusive:
  # input_cfg overrides manifest_filepath. Step 5's human-label upsampling changes
  # what the model optimises toward, while tarring only changes how fast it reads,
  # so the weighting wins for the pilot. Per-bucket tarred sets would give both --
  # that is a phase-4 job.
  log "tarred shards exist but the pilot keeps weighted sampling (Step 5 > Step 4);"
  log "shards are ready for phase 4 at $TARRED"
fi
$PY "$HERE/pilot_train.py" "${ARGS[@]}" 2>&1 | tee "$ROOT/logs/pilot_train.log"
log "pilot exited rc=${PIPESTATUS[0]}"

# ---------------------------------------------------------------- 7. stop here
log "=== PILOT COMPLETE -- STOPPING ==="
log "Phase 4 (120M hybrid, ~2 weeks) is NOT auto-started. Decide from:"
log "  $ROOT/logs/pilot_train.log     did the loss converge at all?"
log "  $ROOT/logs/bench_dl_*.log      is the run I/O bound or compute bound?"
log "  $ROOT/logs/manifests.log       what the mix actually came out as"
log "Gold-test eval is the gate that matters -- an in-domain WER of 3-6% on this"
log "corpus measures agreement with canary's labels, not accuracy."
