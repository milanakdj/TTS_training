# Whisper Nepali ASR — hard-won notes

Everything in this folder came out of one 80-hour `whisper-large-v3` fine-tune that
produced a model with **588% WER** and a **0.04 loss** at the same time. The cause,
the salvage, and the guards that now prevent a repeat are all recorded here. Read
the scope note and the bug section before touching anything.

## Scope — local GPU only, and how to keep this file honest

**These scripts target ONE machine: the shared DGX Spark box (`spark-ecb7`, NVIDIA
GB10). Nothing here is for Kaggle or Google Colab. Do not add notebook support,
Kaggle Secrets handling, quota workarounds, or `/kaggle/working` paths to
`whisper-finetune.py`.** If a notebook version is ever needed, it is a **separate new
script**, written from scratch. This one stays a plain `python whisper-finetune.py`
run on a real GPU.

Some comments still mention Kaggle — leftovers from the file's origin. They describe
history, not intent. Treat any Kaggle instruction in these scripts as stale.

**Update this file at the end of every session.** It is the only place the expensive
lessons live: what broke, what the measured numbers were, which guard exists because
of which failure. Add what was learned, correct what turned out wrong, delete what no
longer applies. A stale note here costs the next session hours.

## Files

| File | Job |
|---|---|
| `whisper-finetune.py` | The training run. Notebook-style `# %% Cell N` blocks; runs fine as a plain script. |
| `whisper-eval.py` | Score one or many checkpoints on the held-out test split; write/push the model card. |
| `whisper-doctor.py` | Diagnose a checkpoint that decodes garbage. CPU only, no dataset. |
| `whisper-realign.py` | Short corrective fine-tune to undo the doubled-prefix habit (see below). |

Every script has `--self-check`, which runs its pure logic offline with no GPU, no
dataset and no network. Run it after editing.

## THE BUG — doubled decoder prefix

### What it was

The collator stripped the leading `<|startoftranscript|>` from labels only when:

```python
labels[:, 0] == tokenizer.bos_token_id     # WRONG
```

For Whisper, `bos_token_id` is `<|endoftext|>` (50257). Labels start with
`<|startoftranscript|>` (50258). The condition is **never true**, so the token was
never stripped. `Seq2SeqTrainer` then prepended `decoder_start_token_id` again while
shifting labels into `decoder_input_ids`:

```
trained on :  [sot, sot, lang, task, notimestamps, w1, w2, ...]
generate() :  [sot,      lang, task, notimestamps, ...]
```

Every decoder position off by one. The fix is to compare against
`decoder_start_token_id`. It is now in place with an `assert` beside it.

### The symptom to memorise

**WER above 100% together with a low `eval_loss` means broken generation, not a weak
model.** Teacher forcing feeds the model the prefix it learned, so the loss is
perfect and stays perfect. Only autoregressive decoding exposes the mismatch.

Measured on `checkpoint-26550`, same weights, same clips:

| Decoder prefix | WER |
|---|---|
| `[sot, sot, ne, transcribe, notimestamps]` | 10.98% |
| `[sot, ne, transcribe, notimestamps]` | 465% |

A `PrintProgressCallback.on_evaluate` hook now prints a wall of `!` at the first
eval if `eval_wer > 100`. That turns an 80-hour loss into a 45-minute one.

### Dead ends ruled out along the way

- **`proj_out.weight` missing from the checkpoint.** The log line
  `There were missing keys in the checkpoint model loaded: ['proj_out.weight']` is
  **normal and harmless**. Whisper weight-ties the output head to the decoder token
  embeddings, so it is never saved. `whisper-doctor.py` confirms
  `tie_word_embeddings=True` and `proj_out_tied=True`. Not the cause. Do not chase it.
- **Wrong mel bins.** Real, but only for non-large-v3 models: `medium` uses 80 mel
  bands and vocab 51865, `large-v3` uses 128 and 51866. Feeding a `medium` model
  large-v3 features gives the same fluent garbage. `whisper-eval.py` now detects the
  size from the checkpoint's own config and loads the matching processor.

## Data and the split — do not change these

Reproducing the test split exactly is what makes any WER comparable:

```python
HF_DATASET_ID = "lilgoose7777/slr-combined-nepali-tts2"
load_dataset(..., split="train[:177000]")          # slice BEFORE the shuffle
.filter(text is not None and text.strip() != "")
.shuffle(seed=42)
80 / 10 / 10  ->  train 141600 / val 17700 / test 17700
```

**The row slice happens before the shuffle.** A 20000-row run and a 177000-row run
have completely different splits, so scoring a 20k model against the 177k split can
put its own training clips in the "held-out" set. `whisper-eval.py --num-rows` exists
for exactly this; pass the value the model was trained with.

The corpus is clean single-speaker TTS-style studio audio. A 10.98% WER here says
nothing about noisy real-world recordings.

## Hardware reality on this box

Shared NVIDIA GB10 (DGX Spark, 130.7GB) with a vLLM server holding ~100-110GB.
Roughly 20GB is available, sometimes 8GB.

That constraint drives every odd setting for large-v3, none of which are stylistic:

| Setting | Why |
|---|---|
| `PER_DEVICE_TRAIN_BATCH_SIZE = 1`, accum 16 | effective batch 16 in a ~14GB slice |
| `OPTIM = "adafactor"` | factored second moment: optimizer state in MB, not GB. AdamW does not fit. |
| `GRADIENT_CHECKPOINTING = True` | not optional at 1.54B params here |
| `PER_DEVICE_EVAL_BATCH_SIZE = 1` | batch 2 OOM'd during `generate()` (KV cache) |

Loading a model with `device_map="cuda"` instead of `.to("cuda")` avoids a
double-peak (build in CPU RAM, then copy) that OOM'd on an 8GB slice.

## Env switches on `whisper-finetune.py`

| Variable | Effect |
|---|---|
| `WHISPER_VARIANT` | `tiny`/`base`/`small`/`medium`/`large-v3`. Default `large-v3`. |
| `WHISPER_NUM_ROWS` | dataset rows. Default 177000. |
| `WHISPER_EPOCHS` | default 3. Raise it to genuinely continue a finished run. |
| `WHISPER_EVAL_SUBSET` | per-epoch validation size; wins over the per-variant value. |
| `WHISPER_OPTIM` | escape hatch when `bitsandbytes` is missing: `adamw_torch`. |
| `WHISPER_PUSH=0` | skip the per-epoch Hub backup (keeps smoke runs from making junk repos). |
| `WHISPER_RESUME=0` | ignore a local checkpoint and train from the base model. |
| `WHISPER_RUN_SUFFIX` | names the FINAL repo only: `v2` -> `...-nepali-final-v2`. |

### Smoke-test before any long run

Same script, same pipeline, tiny model, ~20 minutes:

```bash
WHISPER_VARIANT=tiny WHISPER_NUM_ROWS=400 WHISPER_EPOCHS=1 \
WHISPER_EVAL_SUBSET=32 WHISPER_PUSH=0 python whisper-finetune.py
```

`eval_wer` under 100% means the collator, the label shift and `generate()` agree.
Accuracy is irrelevant here; only "does it generate at all" matters. The pipeline is
identical for every model size, so a passing tiny run predicts large-v3.

## Guards now in the training script

Each one exists because it already went wrong once.

1. **Collator compares against `decoder_start_token_id`,** with an `assert`.
2. **Alarm at the first eval** when `eval_wer > 100`.
3. **Refusal to resume a finished checkpoint.** A checkpoint at the planned step count
   trains nothing (`train_runtime: 0.0072`), then the downstream cells report and
   *publish* those old weights as new. That happened. It now exits with instructions.
4. **Resume is local only.** The old Hub fallback silently downloaded 6GB of stale
   weights into a run meant to start fresh. Deleting the local checkpoint folder is
   now all it takes to start over.
5. **No publish above 100% WER.** The model stays on disk with an explanation.
6. **Hub push retries 3x, never raises.** The first run died on `RemoteDisconnected`
   *after* 80 hours of training and 29 hours of eval. The card is written to disk
   before any network call, and a failure prints the manual `hf upload` command.

## Eval cost — the other 38 hours

Autoregressive eval dominates everything. Measured: 0.178 examples/second at eval
batch 1 for large-v3.

- Per-epoch eval on 2000 examples cost **3.1 hours**, three times over.
- The final test eval on the full 17700-example split cost **29 hours** and looked
  like a hang, because the only output was a tqdm bar in a log file.

Fixes now in place: `EVAL_SUBSET = 500` for large-v3, `TEST_SUBSET = 2000` at eval
batch 8 in cell 12, and `generation_max_length` **measured from the eval labels**
(`max label + 8`, capped at 96) instead of a flat 225. A hallucinating decode runs to
the cap on every example, so a cap 10x longer than any real transcript multiplies
eval time by 10 for nothing.

Also: preprocessing is cached (`CLEANUP_RAW_CACHE = False`), so a restart takes 1.4
seconds instead of 25 minutes. The cache is keyed on the labels, which the collator
fix did not change, so it stays valid.

## Scoring a checkpoint

```bash
# quick timing + eyeball, prints predictions next to references
python whisper-eval.py --ckpt <dir-or-hub-id> --n 16 --batch 4

# real number
python whisper-eval.py --ckpt <dir-or-hub-id> --n 5000 --batch 16 --dump preds.tsv

# every local checkpoint, ranked, then publish the winner's card
python whisper-eval.py --all --n 2000 --batch 16 --push-card <repo-id>
```

Notes:

- `--n-sot 2` decodes with the doubled prefix. Needed for any model trained before
  the collator fix. `--n-sot 1` is the control. `0` (default) uses `model.generate()`.
- `--n-sot 2` also switches the generated model card to the doubled-prefix warning
  plus a copy-paste `transcribe()` function, so the card can never disagree with how
  the number was measured.
- A checkpoint can be a local dir, a Hub repo, or `user/repo/checkpoint-8850`
  (subfolder inside a repo).
- The dataset loads once; every checkpoint is scored on identical clips.
- `HF_TOKEN` is validated **before** scoring when `--push-card` is passed. A 401 after
  40 minutes of GPU happened once.
- The script refuses to write a card when WER > 100%.

## Published models

| Repo | What |
|---|---|
| `milanakdj/whisper-large-v3-nepali-final-largev3_1` | `checkpoint-26550`, **WER 10.98% / CER 3.43%** on 5000 clips — but only with the doubled prefix. Card documents it. |
| `milanakdj/whisper-large-v3-nepali-checkpoints` | per-epoch crash backup. Overwritten by every run; nothing depends on it. |
| `milanakdj/whisper-medium-nepali-final` | trained by the same script before the fix, so it has the **same doubled-prefix bug**. Every commit in git history carries the bad line. Score it with `--n-sot 2` and the matching `--num-rows`. |

`checkpoint-26550` and `checkpoint-8850` were the only survivors of the first run:
`save_total_limit=2` deleted `checkpoint-17700`, and `load_best_model_at_end` pinned
`checkpoint-8850` because it had the "best" (meaningless) WER of 377%.

### Careful: same-name checkpoints overwrite

A new run writes `checkpoint-8850` / `17700` / `26550` again, into the same
`OUTPUT_DIR` and the same backup repo. Move any checkpoint you care about **out of**
`~/whisper-output/<variant>/` before starting a new run.

## Salvaging a doubled-prefix model

Two options, both proven:

1. **Document it.** Ship the copy-paste `transcribe()` function from the model card.
   Full accuracy, but `pipeline()`, faster-whisper and whisper.cpp all break.
2. **Realign it.** `whisper-realign.py` resumes the good weights, freezes the encoder
   (the misalignment is decoder-side only, and freezing halves the memory), and trains
   a few hundred steps at 1e-6 with the corrected collator. ~1-3 hours instead of
   another 80. Then plain `model.generate()` works and no card warning is needed.

## Lessons worth keeping

- A metric nobody reads is not a metric. `eval_wer` was 377% after epoch 1.
- Loss and WER disagreeing is information, not noise. Teacher forcing hides
  generation bugs completely and will never reveal them.
- Smoke-test the pipeline on `tiny` before spending days on `large-v3`. Nothing about
  the pipeline differs between sizes.
- Never let a long run's last step be an un-retried network call.
- Resume is for crashes, not for code changes. A checkpoint carries the old code's
  habits with it.
