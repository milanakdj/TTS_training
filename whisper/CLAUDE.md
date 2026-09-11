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

## The 2,215 h spontaneous-corpus finetunes (queue finished 2026-09-11)

Both sizes trained 1 epoch on the 2,215 h corpus (599,608 train / 74,951 val /
74,951 test), effective batch 64, lr 1e-5, scored on the SAME 100 unseen mahadhwani
clips as every baseline, standard single-sot prefix.

| model | params | mahadhwani WER | CER | in-domain test WER | h/epoch |
|---|---|---|---|---|---|
| base `large-v3-turbo`, no finetune | 809M | 0.937 | 0.364 | — | — |
| `large-v3-nepali` (SLR, existing) | 1.54B | 0.245 | 0.090 | — | — |
| `medium-nepali` (SLR, existing) | 769M | 0.301 | 0.124 | — | — |
| **NEW `large-v3-turbo` (corpus)** | 809M | **0.033** | **0.011** | 15.44 | 11.7 |
| **NEW `small` (corpus)** | 244M | **0.056** | **0.019** | 18.29 | 2.9 |

**The finding: domain beat scale, by a wide margin.** `small` at 244M beats the
existing 1.54B `large-v3-nepali` by 4.4x on WER while being 6.3x smaller. Every
prior Nepali Whisper was finetuned on clean studio read speech (SLR), which is
exactly why they sit at 0.245-0.301 on spontaneous audio. Going from 244M to 809M
buys 0.056 -> 0.033; going from read speech to spontaneous speech buys 0.245 ->
0.056 at a THIRD of the parameters.

In-domain vs out-of-domain rank the two models the same way, and the CER gap (0.019
vs 0.011) is narrower than the WER gap -- `small`'s errors are more often
orthographic than lexical.

**The caveat that must travel with these numbers.** Corpus transcripts are
machine-generated (`transcript_used: saaras`, kept where 3 ASRs agreed), and the
test split is drawn from the same corpus. A 0.033/0.056 WER therefore partly
measures agreement with other ASR systems rather than with ground truth. Before
either number is quoted externally, score a few hundred clips transcribed by human
native speakers, held out entirely. A result of 0.08 would still beat every public
Nepali checkpoint by 3x and would be defensible; 0.033 as published is not yet.

### Repos

| Repo | Note |
|---|---|
| `milanakdj/whisper-large-v3-turbo-nepali-final-corpus` | private, 12 files |
| `milanakdj/whisper-small-nepali-final-corpus` | private, 12 files |

### The `small` run's failure mode, fixed 2026-09-11

`small` died at `trainer.train()` on the first attempt: the module-level default is
`OPTIM = "adamw_bnb_8bit"` and `bitsandbytes` is not installed in the tts_service
venv. `large-v3-turbo` survived only because its variant branch overrides to
`adamw_torch`; `small` had no branch. Two fixes:

1. The bnb default now degrades to `adamw_torch` via `importlib.util.find_spec`
   with a loud log line. Same Adam update rule -- quantization only shrinks
   optimizer state, so the fallback costs memory, never convergence.
2. `small` got a real variant branch: batch 32 x accum 2 (effective 64, matching
   turbo so model size is the only variable), `DATALOADER_WORKERS = 6`,
   `VRAM_BUDGET_GB = 45`. Measured 49.7 GB of 80 GB at 100% GPU util, 50 steps/min.

**Do not size a variant by the module-level fall-through.** Those defaults are for a
~20 GB shared slice; `small` would have run at effective batch 14 against turbo's
64 and the two runs would not have been comparable.

## Published models

> **CORRECTION, measured 2026-09-08.** The two published checkpoints decode
> correctly with the **STANDARD single-sot prefix** — the doubled-prefix
> requirement below does *not* apply to them. On 100 held-out mahadhwani clips:
>
> | checkpoint | 1x sot | 2x sot |
> |---|---|---|
> | `whisper-large-v3-nepali-final` | **WER 0.245 / CER 0.090** | 0.282 / 0.106 |
> | `whisper-medium-nepali-final` | 0.301 / 0.124 | 0.300 / 0.122 |
>
> large-v3 is *better* with the single prefix; medium is indifferent. Both repos
> have only two commits (2026-08-24, no realignment), and both declare
> `decoder_start_token_id: 50258` with `forced_decoder_ids: null`, so they were not
> patched after the fact — the doubled-prefix pathology evidently applied to
> intermediate checkpoints during training rather than to what was published.
>
> **Practical consequence: both models work with `faster-whisper`, `pipeline()` and
> `whisper.cpp` as-is.** No custom decode loop is needed, and the HF model card's
> doubled-prefix warning is misleading. Verify with
> `pocket_TTS/infer/asr_test/asr_ab.py` (`FT_NSOT=1` vs `2`) before trusting either
> convention on any new checkpoint — getting it wrong does not error, it silently
> returns worse text.


**These moved to the `himalaya-ai` org on/before 2026-09-08 and are private there.**
The old `milanakdj/...` names below 404 even for the owning account, so a fine-grained
token scoped to `milanakdj` is not enough — it must grant read on `himalaya-ai`.

| Repo | What |
|---|---|
| `himalaya-ai/whisper-large-v3-nepali-final` | was `milanakdj/whisper-large-v3-nepali-final-largev3_1`. **WER 10.98% / CER 3.43%** on 5000 clips of its own clean test split — but only with the doubled prefix. Weights sit at the repo ROOT now, so `checkpoint-26550` is no longer a subfolder path: pass the bare repo id. Re-measured 2026-09-08 on 100 real-human Nepali clips from the TTS eval set: **WER 0.343 / CER 0.122**, versus 0.917 / 0.344 for base `large-v3-turbo` on the same clips. |
| `himalaya-ai/whisper-medium-nepali-final` | was `milanakdj/whisper-medium-nepali-final`. Trained by the same script before the fix, so it has the **same doubled-prefix bug**. Every commit in git history carries the bad line. Score it with `--n-sot 2` and the matching `--num-rows`. |
| ~~`milanakdj/whisper-large-v3-nepali-checkpoints`~~ | **gone.** The per-epoch crash backup no longer exists under either account; nothing depended on it. |

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
