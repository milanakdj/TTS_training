# Nepali Speech Models — Training Repo

Fine-tuning scripts for Nepali speech models: **TTS** (Indic Parler-TTS, Qwen3-TTS) and **ASR** (Whisper).
Everything runs from a single Python file per model — no framework, no config system, no CLI wrapper
(except the Qwen runner, which has one).

## Contents

| File | What it does |
|---|---|
| `training.py` | Indic Parler-TTS Nepali fine-tune (main TTS trainer). Custom loop, bf16, 8-bit AdamW, W&B, pushes to HF. |
| `whisper-finetune.py` | Whisper Nepali ASR fine-tune. HF `Seq2SeqTrainer`, WER/CER, per-epoch checkpoint backup to HF + auto-resume. |
| `push_checkpoint_to_hub.py` | One-off: upload a local `best_checkpoint` folder to a HF model repo. |
| `test_model_card.py` | Renders `whisper-finetune.py`'s model card against stub state. Run it after editing the card. |
| `test_vram_fit.py` | Runs a few training steps on synthetic batches to check the config fits in free VRAM. ~1 min vs ~25 min for a real attempt. |
| `qwen_nepali_clean_package/` | Qwen3-TTS 1.7B Nepali package — see its own `QWEN_RUNBOOK.md`. |
| `inference_indic_tts_new.ipynb` | Colab inference for the fine-tuned Parler-TTS model (batch generation over test sentences). |
| `inference_test_collab.ipynb` | Older/scratch inference notebook. |
| `training_kaggle.ipynb` | Notebook ancestor of `training.py` (Kaggle T4 era, fp16). Kept for history — prefer `training.py`. |

## Prerequisites

- Linux + NVIDIA GPU. Reference machines: DGX A100 80GB (Parler-TTS), DGX Spark 128GB unified (Whisper medium/large-v3).
- Python 3.10+, CUDA-enabled PyTorch installed for your driver.
- A **write-scoped** HuggingFace token. Both trainers push checkpoints and models to the Hub.

```bash
pip install torch torchaudio transformers datasets accelerate huggingface_hub \
            evaluate jiwer librosa soundfile bitsandbytes wandb numpy
pip install git+https://github.com/huggingface/parler-tts.git   # TTS only
```

`whisper-finetune.py` pip-installs its own deps in its first cell, so on a bare Kaggle/Colab image you can
just run it.

## Environment variables

Both trainers read secrets from the environment — nothing is hardcoded, and a missing `HF_TOKEN`
fails immediately with a `KeyError` rather than 40 minutes into a run.

```bash
export HF_TOKEN=hf_...           # required, must have WRITE scope
export WANDB_API_KEY=...         # required by training.py (it calls wandb.init unconditionally)
```

On Kaggle, store the token as a Secret and set it into `os.environ["HF_TOKEN"]` in the first cell.

Read-only tokens fail with a 403 at the push step, after training has already completed. Check the scope first.

## TTS — `training.py`

Fine-tunes `ai4bharat/indic-parler-tts-pretrained` on Nepali audio, with a **single fixed speaker
description** applied to every sample (`SPEAKER_DESCRIPTION`). Any description column in the dataset is
deliberately ignored, so the model learns one consistent voice rather than a caption-conditioned mixture.

```bash
HF_TOKEN=hf_... WANDB_API_KEY=... python training.py
```

All configuration lives in the `2. Config` block at the top of the file:

| Setting | Default | Notes |
|---|---|---|
| `OUTPUT_REPO` | `milanakdj/indic-parler-tts-nepali-finetuned-dgx-v4.1-slr` | Where the best checkpoint is pushed. |
| `DATASET_REPO` | `lilgoose7777/slr-combined-nepali-tts2` | Single `train` split; 5% is held out for validation. |
| `FINETUNE_BASE` | `ai4bharat/indic-parler-tts-pretrained` | Base model for a fresh run. |
| `RESUME_FROM_HF` | `False` | `True` loads weights from `OUTPUT_REPO` instead of the base. |
| `RESUME_STATE_PATH` | `None` | Path to a `training_state.pt` to restore optimizer/scheduler/step. |
| `BATCH_SIZE` / `GRAD_ACCUM_STEPS` | 8 / 8 | Effective batch 64. Sized for 80GB with `MAX_AUDIO_TOKENS=1856`. |
| `LEARNING_RATE` | `1e-5` | Cosine schedule; warmup defaults to `total_steps / 10`. |
| `MAX_AUDIO_TOKENS` | `1856` | ≈21.5s of audio at 86.13 DAC frames/sec. |
| `SAVE_STEPS` | `200` | Validate; save `best_checkpoint` if val loss improved. |
| `ZIP_EVERY_STEPS` | `600` | Periodic checkpoint zipped to `./checkpoints_<uuid>/` and the folder deleted. |

Details worth knowing before you change anything:

- **Long clips are dropped, not truncated.** Anything over `MAX_AUDIO_TOKENS` is filtered out during
  dataset load. The earlier behaviour chopped clips and glued an EOS onto the cutoff point, which taught the
  model to stop mid-sentence — with the old 6.4s cap that corrupted essentially every example. A 1s
  `SAFETY_MARGIN_SEC` covers DAC framing rounding.
- **Audio is DAC-encoded on the fly** by a frozen `ylacombe/dac_44khz` (9 codebooks, 44.1kHz) inside
  `__getitem__`, so `num_workers=0` — the encoder lives on the GPU and forked workers would deadlock.
- **bf16 weights + 8-bit AdamW** (`bitsandbytes`) — installed automatically if missing.
- A **sanity forward pass** runs before the optimizer is built. If labels are all `-100` or the loss looks
  wrong, you find out in seconds instead of after the first save.
- `ParlerTTSConfig.has_no_defaults_at_init = True` is set at import — without it transformers' `to_diff_dict()`
  crashes on config repr/logging, because `ParlerTTSConfig` can't be instantiated with no arguments.

### Outputs

```
checkpoints_<run-uuid>/
  best_checkpoint/              model + prompt tokenizer + desc_tokenizer/ + README.md + training_state.pt
  checkpoint_step_<n>.zip       periodic snapshots (folder removed after zipping)
```

Pushed to `OUTPUT_REPO`: model weights, prompt tokenizer, a generated model card with the full training
config and best val loss, and `training_state.pt`.

The **description tokenizer is not pushed to the repo root** — on resume it is reloaded from
`model.config.text_encoder._name_or_path`, so vocab and special tokens stay consistent no matter what
was pushed. It is saved locally under `best_checkpoint/desc_tokenizer/` for inspection.

### Resuming

- Weights only: `RESUME_FROM_HF = True`. Optimizer/scheduler start fresh.
- Full state: also set `RESUME_STATE_PATH` to a `training_state.pt` (from a periodic zip or the pushed repo).
  If the restored step is already ≥ `total_steps`, the schedule is extended by another `NUM_EPOCHS` worth
  of steps with zero warmup.

### Inference

See `inference_indic_tts_new.ipynb`, or the usage snippet in the generated model card:

```python
model            = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
prompt_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
desc_tokenizer   = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
```

Use a description close to `SPEAKER_DESCRIPTION` from training — the model was only ever conditioned on
that one caption, so wandering far from it gives unpredictable results.

## ASR — `whisper-finetune.py`

Fine-tunes `openai/whisper-<variant>` on the same Nepali corpus, split 80/10/10, reporting WER and CER.

```bash
HF_TOKEN=hf_... python whisper-finetune.py
```

Written as `# %%` cells: run it as a plain script, or paste the blocks into notebook cells. It detects
which it is in (`ipykernel in sys.modules`) and drops `num_proc` to 1 in a notebook, because forking
preprocessing workers inside Jupyter deadlocks silently.

Config lives in Cell 3:

| Setting | Default | Notes |
|---|---|---|
| `MODEL_VARIANT` | `medium` | `tiny` / `base` / `small` / `medium` / `large-v3`. |
| `NUM_ROWS` | `177000` | Full dataset. Lower it for a quick run — it slices at download time. |
| `LEARNING_RATE` | per-variant | `LR_OVERRIDES` maps variant → LR (5e-5 tiny … 5e-6 large-v3). Set `LR_OVERRIDES = {}` to force 1e-5 everywhere. |
| batch size | 4×4 (medium), 2×8 (large-v3), else 7×2 | Effective batch stays 14–16 across variants. Sized for a ~20GB slice of the shared GPU. |
| `OPTIM` | `adamw_bnb_8bit` | AdamW's two fp32 moments are ~6.2 GB for whisper-medium. 8-bit states cut that to ~1.5 GB with identical optimizer behaviour. `adafactor` saves ~6 GB with no dependency but is not Adam — retune the LR if you use it. |
| `GRADIENT_CHECKPOINTING` | `False` | ~half the activation memory for ~25–30% slower steps, but it errors on this transformers/torch build. `gradient_checkpointing_kwargs={"use_reentrant": False}` is already set for when you re-enable it. |
| `EVAL_SUBSET` | `2000` | Per-epoch validation size. The test split is still evaluated in full. |
| `VRAM_BUDGET_GB` | `20` | Only drives the startup warning. Set it to what you actually expect to get. |
| `CLEANUP_RAW_CACHE` | `False` | See the cache note below. |
| `NUM_EPOCHS` | `3` | `eval_strategy` and `save_strategy` are both `"epoch"`. |
| `HF_USER` | `milanakdj` | **Set this to your own account.** Both repo ids derive from it. |
| `OUTPUT_DIR` | `~/whisper-output` | Local checkpoints; `save_total_limit=2`. |

Repo names are derived, never repeated:

```python
RUN_NAME      = f"whisper-{MODEL_VARIANT}-nepali"
CKPT_REPO_ID  = f"{HF_USER}/{RUN_NAME}-checkpoints"   # private, per-epoch backups
FINAL_REPO_ID = f"{HF_USER}/{RUN_NAME}-final"         # public, model + model card
```

Switching `MODEL_VARIANT` from `medium` to `large-v3` therefore retargets both repos on its own — the
checkpoints repo and the final repo cannot drift onto different models.

Details worth knowing:

- **The model is loaded *after* preprocessing** (Cell 7, not Cell 5). Loading it earlier initialises a CUDA
  context, and `datasets.map(num_proc>1)` forks after that — workers then hang at 0% CPU forever with no error.
- **Every epoch checkpoint is pushed to `CKPT_REPO_ID`** as it is written (`PushCheckpointToHubCallback`),
  so a checkpoint survives a session dying mid-run. The upload is wrapped in a `try/except` — a backup
  must never kill the run it is backing up. An HF storage quota did exactly that once, throwing away a
  finished epoch at the moment of success. The local checkpoint is written before the callback fires, so
  training continues and the failure is just logged.
- **The checkpoints repo is public.** Free accounts have a small *private* storage quota and whisper-medium
  checkpoints are several GB each, so `CKPT_PRIVATE = True` fails partway through training with
  `403 Private repository storage limit reached`.
- **Resume is automatic.** On start it looks for a local checkpoint in `OUTPUT_DIR`; if there is none it
  lists `CKPT_REPO_ID`, downloads the highest-numbered `checkpoint-*`, and resumes from it. No checkpoint
  anywhere → fresh start.
- **The GPU is shared** with a vLLM server holding ~99 GB of the 122 GB card, leaving ~21 GB. A `[gpu]`
  line at startup prints free vs total VRAM and warns below `VRAM_BUDGET_GB`. Preprocessing runs ~25 minutes
  before the first training step, so without that check an already-full GPU costs half an hour to discover.
  `torch.cuda.mem_get_info()` reports memory across all processes, so another user's job shows up as
  missing VRAM.
- **On a small slice, the optimizer is the biggest lever.** For whisper-medium, fp32 weights + grads +
  AdamW moments are ~12.4 GB static before a single activation; the moments alone are ~6.2 GB of that.
  `OPTIM = "adamw_bnb_8bit"` cuts them to ~1.5 GB without changing what the optimizer does — a larger
  saving than gradient checkpointing, and it composes with it.
- **Per-epoch validation runs on `EVAL_SUBSET` examples, not all 17.7k.** `predict_with_generate` decodes
  autoregressively, so the full split would be hours per epoch for a number used only to rank checkpoints.
  The split is pre-shuffled and `select()` is deterministic, so epochs stay comparable. Cell 12 still
  evaluates the full test split — that runs once and is the number you report.
- **Disk usage is printed at every milestone** (`[disk] ...`). A full disk causes silent stalls, not clean
  errors — this is how you see it coming.
- **`CLEANUP_RAW_CACHE` is off by default.** `cleanup_cache_files()` deletes every `cache-*.arrow` in the
  dataset's cache directory except the ones `raw_datasets` is using — and the `map()` output backing
  `vectorized_datasets` lives in that same directory. The current run is unaffected (the files stay open
  until the process exits), but the next run redoes all ~25 minutes of preprocessing. Turn it on only when
  disk is actually tight, as it was on Kaggle's quota.
- **bf16 where supported**, fp16 fallback otherwise. tqdm is disabled and progress printed as flushed plain
  text, because background log capture (Kaggle "Save Version") doesn't render progress bars.
- Labels longer than 448 tokens (Whisper's decoder limit) are filtered out.
- **A model card is generated and pushed with the model** (Cell 14, `build_model_card`). It records every
  hyperparameter, the data split and example counts, precision, optimizer, and the test WER/CER — read from
  the live config rather than retyped, so it can't go stale. That is how you tell months later exactly what
  produced a given repo. It is uploaded *after* `push_to_hub`, which writes its own stub card first.
- Labels/metrics aside, all config lives in Cell 3. Imports are all in Cell 2 — except the deliberately late
  model load noted above.

Cell 16 is a standalone utility: download one specific checkpoint from the checkpoints repo and promote it
into the final repo, model card included. Set `CHECKPOINT` to the one you want; the repo ids come from Cell 3.

After editing the card, `python test_model_card.py` renders it against stub state (no GPU, no dataset) and
checks the placeholders resolve — otherwise a typo in a 90-line f-string surfaces only at the end of a run.

### Checking a config fits before running it

Preprocessing takes ~25 minutes before the first training step, so a batch size that doesn't fit is
expensive to discover. `test_vram_fit.py` reads the real config out of `whisper-finetune.py` (no duplicated
numbers), builds the model, and runs a few steps on synthetic batches. Whisper's encoder input is a fixed
3000 frames, so a fake batch has the same memory profile as a real one.

```bash
python test_vram_fit.py              # the config as written
python test_vram_fit.py --batch 8    # try bigger
```

It reports peak VRAM against what's free, tells you whether to step up or down, and fails clearly if
`adamw_bnb_8bit` doesn't load on this machine.

## Qwen3-TTS

Separate, self-contained package with its own CLI. See `qwen_nepali_clean_package/QWEN_RUNBOOK.md` for
GPU setup, the smoke tests, and the full training commands. It also takes `HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN` and `WANDB_API_KEY` from the environment, and fails early if a Hub push is
requested without them.

## Datasets

| Repo | Use |
|---|---|
| `lilgoose7777/slr-combined-nepali-tts2` | 177k rows, single `train` split, columns `audio` / `text` / `text_description`. Used by both `training.py` and `whisper-finetune.py`. |
| `Titung/nepali-tts-tagged-combined` | Qwen3-TTS runs. |

Note on the SLR corpus: it is clean single-speaker studio TTS audio. Excellent for TTS; for ASR it is a fast
proof-of-concept, but a Whisper model trained only on it will underperform on noisy real-world audio with
background noise, multiple speakers, and varied accents.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `KeyError: 'HF_TOKEN'` at startup | Token not exported. This is intentional — fail at second 0, not after training. |
| 403 on push, training already done | Token is read-only. Needs write scope. |
| `403 Private repository storage limit reached` | Private storage quota exhausted. Set `CKPT_PRIVATE = False`, or delete old checkpoint repos. |
| Preprocessing hangs at 0% CPU, no error | CUDA initialised before a forking `map()`. Don't move the model load earlier. |
| Silent stall mid-training | Disk full. Check the `[disk]` lines. |
| OOM in `training.py` | Lower `BATCH_SIZE`, raise `GRAD_ACCUM_STEPS` by the same factor to keep the effective batch. |
| OOM in `whisper-finetune.py` | Read the OOM text: if it says another process holds most of the card, you're sharing it — check `nvidia-smi`. Otherwise set `GRADIENT_CHECKPOINTING = True` and make the same batch/accum trade in the per-variant block in Cell 3. |
| `num_proc` and OOM | Unrelated. `PREPROCESS_NUM_PROC` is CPU-side and only affects preprocessing, which has already finished by the time training allocates VRAM. |
| Generated speech cuts off mid-sentence | `MAX_AUDIO_TOKENS` too low for the corpus — clips are being filtered, or an older run truncated them. |
