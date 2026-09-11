# indic-transcribe-flex — Nepali finetune

Finetuning `bodhan-ai/indic-transcribe-flex` (1.22B, FastConformer encoder +
Transformer decoder, from `nvidia/canary-1b-v2`) on Nepali.

## READ THIS FIRST — our usual eval set cannot measure this model

`bodhan-ai/indic-transcribe-flex` is a **Canary derivative**, and our large Nepali
corpora are **Canary pseudo-labels**. Scoring flex on them compares Canary to Canary.

Measured 2026-09-11, base checkpoint, no finetuning, 25 clips per source:

| source | transcripts | WER | CER | exact |
|---|---|---|---|---|
| `ans_snr40-50` | saaras/canary | 0.015 | 0.002 | 20/25 |
| `mahadhwani` | saaras/canary | 0.022 | 0.004 | 17/25 |
| `ai4bharat___rasa` | **human script** | 0.129 | 0.016 | 12/25 |
| `indicvoices-r` | **human verbatim** | 0.177 | 0.038 | 4/25 |

The proof, on the mahadhwani sample — flex's own output scored against each label
source separately:

| flex output vs | WER | exact |
|---|---|---|
| `canary_transcript` | **0.019** | 18/25 |
| `conformer_transcript` | 0.021 | 17/25 |
| `text` (the label we train on) | 0.022 | 17/25 |
| `saaras_transcript` | 0.060 | 8/25 |

flex matches Canary 3x more closely than Saaras, and 16/25 of those rows are
`transcript_used: canary`. On the standard 100-clip mahadhwani set it scores
**WER 0.013 with 85/100 exact matches and no finetuning at all** — better than our
best whisper. That number is an artefact of label provenance.

**Held-out GOLD test (600 human-transcribed clips): WER 0.155 / CER 0.051.** That
is flex's real Nepali error rate, 12x the contaminated figure, and the only number
to quote.

Rule: **never score a Canary-family model on a corpus whose labels came from
Canary.** Check `transcript_used` before trusting any WER on this data.

Our own whisper numbers are NOT affected — verified 0/100 mahadhwani eval clips
appear in `whisper/nepali_corpus_asr.jsonl`. The `in_tts_train: true` flag in the
eval JSON refers to the pocket-TTS training set, not the ASR corpus.

## Result — the finetune is a null (2026-09-11)

289 h of human-transcribed Nepali, 2 epochs, and the gold test did not move:

| run | n | WER | CER |
|---|---|---|---|
| flex base | 600 | **0.155** | 0.051 |
| flex + 289 h gold FT | 600 | **0.157** | 0.053 |

It is a null, not a regression. Per source: ivr 0.138 -> 0.140, ivrl 0.187 -> 0.180,
rasa 0.163 -> 0.175 — all inside the noise. 66 clips improved, 54 got worse, 480
scored identically, and **441/600 hypotheses are byte-identical** to the base
model's. The differences that do exist are single-token swaps
(`हामीले कुदायो बा` -> `हामीले कुदायौँ`), a coin flip in both directions.

Training itself was healthy: train loss 0.78 -> 0.56, and val loss fell 0.370 ->
0.335 by epoch 0.5 and then sat flat for the remaining 1.5 epochs. The model
converged on the objective; the objective just did not buy transcription accuracy.

The contamination control says the same thing, and adds one finding: **gold
finetuning does not wash the Canary fingerprint out.** Base vs FT on the same
100-clip mixed probe:

| probe source | labels | base | ft | delta |
|---|---|---|---|---|
| `ans_snr40-50` | canary/saaras | 0.015 | 0.021 | +0.006 |
| `mahadhwani` | canary/saaras | 0.022 | 0.030 | +0.008 |
| `ai4bharat___rasa` | human | 0.129 | 0.141 | +0.012 |
| `indicvoices-r` | human | 0.177 | 0.169 | -0.008 |
| all | | 0.086 | 0.090 | +0.004 |

After 289 h of human-transcribed training the model still scores 0.021 / 0.030 on
Canary-labelled audio. The gap between pseudo-label agreement (~0.025) and human
accuracy (~0.155) is just as wide after finetuning as before, so the contamination
warning above applies to every checkpoint we produce from this model, not only the
released one.

The delta is not distinguishable from zero — paired bootstrap 95% CI
-0.006 .. +0.011 — so this is "no effect", not "slightly worse".

What this rules out: another pass of the same recipe. lr 1e-5 for 2 epochs is not
under-training; val loss plateaued a quarter of the way in.

What it does NOT rule out, corrected 2026-09-11: **we did not train on all the gold
we have.** `build_gold_manifest.py` drops everything over `MAX_SEC = 30`, and for
`indicvoices-r-long` that is 16,368 clips / **547.8 h** — 2.9x the training set,
same human transcripts, same licence. (rasa loses 0.2 h to the cap, ivr none.)

| dropped band | clips | hours |
|---|---|---|
| 30-45 s | 2,485 | 25.6 |
| 45-60 s | 1,955 | 28.5 |
| 60-120 s | 5,171 | 121.7 |
| 120-300 s | 6,270 | 319.5 |
| > 300 s | 487 | 52.5 |

Those rows carry only whole-clip `text` — no `asr_segments`, `turns` or word
timings — so they cannot be windowed without forced alignment, and bad alignment
would inject label noise into the one clean corpus we have. Cheapest real
experiment: align and window the 60-300 s band (441 h) and retrain.

## Why the finetune trains on gold only

600+ h of pseudo-labelled audio is available and deliberately unused. Training on
it would teach flex to imitate an ASR it already is: the apparent score would move
while real accuracy did not. Gold sources carry no `transcript_used` field —
`ai4bharat___rasa` is the recording script, `indicvoices-r` ships a human
`verbatim`. That is where the headroom is.

| split | rows | hours | mix |
|---|---|---|---|
| train | 121,584 | 288.8 | rasa 27,775 / ivr 63,418 / ivrl 30,391 |
| val | 600 | 1.4 | stratified |
| test | 600 | 1.4 | stratified |

Clips over 30 s are dropped (16,368 of `indicvoices-r-long`): the checkpoint trains
at max_duration 30 and its own wrapper refuses past 45 s.

## What had to be built — the shipped code is an inference port

1. **Training loss.** `forward()` raises `NotImplementedError` when `labels` is
   passed. `training_forward` in `flex_finetune.py` adds teacher forcing + CE. The
   decoder already builds a causal mask on the `seq > 1` path, so teacher forcing
   is correct once a loss exists; `use_cache=False` so no KV cache is allocated.
2. **Text encoding.** The tokenizer is decode + prompt building only. Text ids are
   `spl_size (1152) + multi.encode(text)` — multilingual pieces live at [1152, 7152).
3. **Prompt-masked targets.** `full = prompt(10) + text + eos`; input is `full[:-1]`,
   labels are `full[1:]` with the first 9 positions set to -100. The prompt is
   context, not a prediction target. Unit-tested: prompt prefix, masking, EOS,
   shift-by-one, exact round-trip decode.

## Environment — a separate venv on purpose

`/workspace/venvs/flex` (transformers 5.16.1, torch 2.6.0+cu124). The modelling
code needs the new `Cache.layers` API; the tts_service venv's transformers 4.46.1
fails with `'DynamicCache' object has no attribute 'layers'`. Do not "fix" this by
upgrading the shared venv — whisper and pocket_TTS run against 4.46.

Two traps:
- `warmup_ratio` no longer exists in transformers 5.x — use `warmup_steps`.
- `unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT` or accelerate tries to
  init a distributed group and dies. Same unset `whisper/run_queue.sh` carries.

## Sizing — measured, not assumed

Worst case (batches of the longest 30 s clips) on the 85 GB H100:

| batch | peak | s/step |
|---|---|---|
| 8 | 18.9 GB | 0.42 |
| 12 | 24.7 GB | 0.29 |
| 16 | **30.6 GB** | 0.37 |

Running batch 16 x accum 2 (effective 32), lr 1e-5, 2 epochs = 7,600 steps at
~1.7 it/s (~75 min).

## Commands

```bash
cd /root/tts/TTS_training/whisper/flex
./run_flex.sh                      # baseline -> finetune -> gold test -> probe
python3 build_gold_manifest.py     # rebuild the gold splits

# score any checkpoint on the gold test
FLEX_MODEL=/workspace/flex-nepali \
FLEX_EVAL=$PWD/gold_test_eval.json \
  /workspace/venvs/flex/bin/python flex_eval2.py --tag mytag
```

Log: `/root/flex-queue.log`. Output: `/workspace/flex-nepali`.
