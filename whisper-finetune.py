#!/usr/bin/env python3
# %% [markdown] Cell 0
# Whisper Fine-Tuning for Nepali ASR — Kaggle Notebook Script
# =============================================================
#
# Data source: HuggingFace Hub dataset `lilgoose7777/slr-combined-nepali-tts2`
# (177k rows, single "train" split, columns: audio, text, text_description).
# This script pulls NUM_ROWS rows from it and splits 80:10:10 locally.
#
# NOTE: this dataset looks like single-speaker studio TTS audio (clean, one female
# voice, described via `text_description` captions), not noisy multi-speaker podcast
# recordings. It is a fine, fast dataset to prototype with, but a model fine-tuned
# only on this will likely underperform on messy real-world audio (background noise,
# multiple speakers, accents). Good for a first proof-of-concept run.
#
# Configuration (all of it lives in cell 3 -- nothing is configured anywhere else):
#   Model variants      : whisper-tiny / base / small / medium / large-v3
#                         (set MODEL_VARIANT; repo names derive from it)
#   Epochs              : 3
#   Learning rate       : per-variant via LR_OVERRIDES, default 1e-5
#   Batch size          : 7 x accum 2 (= 14), retuned per variant for a ~20GB
#                         VRAM slice: 4 x 4 for medium, 2 x 8 for large-v3
#                         (= 16 effective in both cases).
#   Split               : 80 / 10 / 10 (train/val/test)
#   Precision           : bf16 where supported, fp16 fallback
#   Metrics             : WER, CER
#
# SHARED GPU
# ----------
# This box's GPU is shared with a vLLM server holding ~99GB of the 122GB card, so
# this run gets ~21GB. A startup check ([gpu] ... line) prints free vs total VRAM
# and warns below VRAM_BUDGET_GB, because preprocessing runs ~25 minutes before
# the first training step -- without it, a full GPU costs half an hour to find out
# about. Batch sizes and OPTIM in cell 3 are both sized for that slice.
#
# DISK SAFETY
# -----------
# - Disk usage is printed at every major milestone ([disk] ... lines) so you can
#   see exactly when/if you're approaching Kaggle's quota, instead of finding out
#   only after something silently stalls (a full disk causes silent hangs, not
#   clean errors).
# - Right after preprocessing, the raw decoded-audio cache is deleted (it's no
#   longer needed once everything is converted into log-mel features + token
#   ids) -- this was the main disk consumer that caused the earlier stall.
# - save_total_limit=2 in training args caps how many checkpoint folders
#   accumulate during training.
#
# HOW TO USE ON KAGGLE
# ---------------------
# 1. Turn each "# %%" block into its own Kaggle notebook cell (or just run the whole
#    file top to bottom as one cell -- it works either way).
# 2. Enable GPU: Settings -> Accelerator -> GPU T4 x2 / P100 / etc.
# 3. Export a WRITE-scoped HF token in the environment before running:
#       export HF_TOKEN=hf_...
#    (on Kaggle: store it as a Secret and os.environ["HF_TOKEN"] = <secret> in cell 1).
#    The script reads it from os.environ and calls login() -- it is required, since
#    checkpoints and the final model are pushed to the Hub.
# 4. Run. No manifest CSV needed -- it streams/downloads directly from the Hub.
#
# CHECKPOINT RESUME + HF BACKUP (added)
# --------------------------------------
# - Every epoch checkpoint saved by the Trainer is now also pushed to a
#   dedicated HF checkpoints repo right after it's written (cell 10's
#   PushCheckpointToHubCallback) -- so a checkpoint survives even if the
#   Kaggle session dies before the next epoch finishes.
# - On (re)start, cell 11 looks for a checkpoint locally first; if none is
#   found (e.g. fresh Kaggle session), it downloads the latest one from the
#   HF checkpoints repo and resumes training from there automatically.
# - Cell 14 pushes the final model plus a generated model card recording every
#   setting this run used (hyperparameters, data splits, metrics), so the Hub
#   repo documents itself. Cell 16 does the same for a promoted checkpoint.
#

# %% Cell 1
# Run this cell first, then restart nothing — imports below will just work.
import subprocess
import sys


def pip_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *pkgs]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Some environments (Debian/Ubuntu 23+, this sandbox, etc.) mark the
        # system Python as "externally managed" (PEP 668) and refuse installs
        # unless this flag is passed. Kaggle images don't hit this, but it's
        # a one-line safety net for environments that do.
        subprocess.run([*cmd, "--break-system-packages"], check=True)


pip_install(
    [
        "transformers>=4.42.0",
        "datasets>=2.20.0",
        "accelerate>=0.31.0",
        "evaluate",
        "jiwer",
        "librosa",
        "soundfile",
    ]
)

# %% Cell 2
import os

# Must be set before torch initialises its CUDA allocator. Reduces fragmentation,
# which matters most when the GPU is shared and you only get a slice of it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import evaluate
import datasets as hf_datasets_module

from datasets import Audio, DatasetDict, load_dataset
from huggingface_hub import HfApi, login, snapshot_download
from transformers import (
    TrainerCallback,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import get_last_checkpoint

# %% Cell 3
HF_DATASET_ID = "lilgoose7777/slr-combined-nepali-tts2"
HF_SPLIT = "train"  # this dataset only has one split; we carve our own below

# Token comes from the environment (same as training.py) -- never hardcode it here.
# Needs WRITE access: checkpoints and the final model are pushed to the Hub.
#   export HF_TOKEN=hf_...        (or set it as a Kaggle Secret / env var)
HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN)
NUM_ROWS = int(os.environ.get("WHISPER_NUM_ROWS", 177000))  # full dataset
# (was 20000 subset for the Kaggle run)
# Scoped by MODEL_VARIANT: checkpoint dirs are not compatible across model
# sizes (different d_model/mel dims), and the resume logic picks whatever is
# in OUTPUT_DIR. Without the variant in the path, switching medium -> large-v3
# would load a medium checkpoint into a large-v3 model and die on state_dict
# size mismatch.
# Env overrides exist for ONE reason: to smoke-test this exact script end to end on
# a tiny model in minutes before committing 80 hours to large-v3. Nothing about the
# pipeline differs between sizes, so a passing tiny run proves the collator, the
# label shifting and generate() all agree. Defaults are the real large-v3 run.
#   WHISPER_VARIANT=tiny WHISPER_NUM_ROWS=400 WHISPER_EPOCHS=1 \
#   WHISPER_EVAL_SUBSET=32 python whisper-finetune.py
MODEL_VARIANT = os.environ.get("WHISPER_VARIANT", "large-v3")
# one of: "tiny", "base", "small", "medium", "large-v3"
OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), "whisper-output", MODEL_VARIANT
)  # local path on DGX Spark

LANGUAGE = "nepali"  # Whisper's language tag for decoding
TASK = "transcribe"  # "transcribe" (ne->ne) or "translate" (ne audio -> en text)

TRAIN_VAL_TEST_SPLIT = (0.80, 0.10, 0.10)
SEED = 42
EVAL_SUBSET = 2000  # per-epoch validation size; see the note above the Trainer

# Autoregressive eval is the single most expensive thing in this script. The
# large-v3 run measured 0.178 examples/second at eval batch 1, so 2000 validation
# examples cost 3.1 HOURS per epoch -- 9 hours of the 80-hour run spent on a number
# only used to rank three checkpoints. Two levers, both applied for large-v3 below:
#   EVAL_SUBSET       -- fewer examples
#   GENERATION_MAX_LEN -- fewer tokens per example. This dataset's transcripts are
#                        ~24 characters (~15 tokens); 225 let every hallucinating
#                        decode run 15x longer than any real answer.
GENERATION_MAX_LEN = 225

# --- Hyperparameters from your table ---
NUM_EPOCHS = float(os.environ.get("WHISPER_EPOCHS", 3))
BASE_LEARNING_RATE = 1e-5

# Optional: per-model LR override. Smaller models often converge better with a
# slightly higher LR than 1e-5. Comment this dict out (set to {}) to force the
# exact 1e-5 for every model as in your original table.
LR_OVERRIDES = {
    "tiny": 5e-5,
    "base": 3e-5,
    "small": 1e-5,
    "medium": 1e-5,
    "large-v3": 5e-6,  # larger pretrained models generally fine-tune better with a
    # lower LR -- less risk of overwriting what it already learned
}
LEARNING_RATE = LR_OVERRIDES.get(MODEL_VARIANT, BASE_LEARNING_RATE)

# The GPU is shared: a vLLM server (PID 735719) holds ~99GB of the 122GB card,
# leaving this run ~21GB. Everything below is sized for that slice, not the full
# card. If you get the whole GPU back, raise PER_DEVICE_TRAIN_BATCH_SIZE and lower
# GRADIENT_ACCUMULATION_STEPS by the same factor -- the effective batch (and
# therefore the result) is unchanged.
# VRAM_BUDGET_GB only drives the startup check, which fails fast instead of 25
# minutes into preprocessing.
VRAM_BUDGET_GB = 20

# Recomputes activations in the backward pass instead of storing them: ~half the
# activation memory for ~25-30% slower steps. Left OFF because it errors on this
# transformers/torch build -- OPTIM below buys more memory anyway. If you turn it
# back on, use_reentrant=False (set in training_args) is the non-deprecated path
# and fixes most of the ways it breaks.
GRADIENT_CHECKPOINTING = False

# The optimizer is the biggest lever on a small VRAM slice. For whisper-medium's
# 769M params, AdamW keeps two fp32 moments = ~6.2GB of the ~12.4GB static
# footprint, before a single activation is stored.
#   "adamw_torch"     -- baseline, ~6.2GB of states
#   "adamw_bnb_8bit"  -- same Adam behaviour, states quantized to ~1.5GB (saves
#                        ~4.7GB). Needs bitsandbytes; training.py already uses it.
#   "adafactor"       -- factored second moment, states are megabytes (saves
#                        ~6GB), no extra dependency, but it is NOT Adam -- the
#                        1e-5 LR was tuned for AdamW, so expect to retune.
# Try adamw_bnb_8bit first: it is the only option that saves memory without
# changing what the optimizer does.
OPTIM = os.environ.get("WHISPER_OPTIM", "adamw_bnb_8bit")
# adamw_bnb_8bit needs bitsandbytes. On a box without it, a smoke run would die on
# an unrelated dependency, so WHISPER_OPTIM=adamw_torch is the escape hatch.

PER_DEVICE_TRAIN_BATCH_SIZE = 7
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch size = 7 * 2 = 14
PER_DEVICE_EVAL_BATCH_SIZE = 7

if MODEL_VARIANT == "medium":
    # ~769M params. fp32 weights + grads + Adam states alone are ~12GB before a
    # single activation, so batch 16 needs the whole card. 4 x 4 fits a ~20GB slice.
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4  # 4 * 4 = 16 effective batch
    PER_DEVICE_EVAL_BATCH_SIZE = 4  # eval has no grads, but keep headroom for generate()
elif MODEL_VARIANT == "large-v3":
    # ~1.54B params. fp32 weights (6.2GB) + grads (6.2GB) + 8-bit Adam (3.1GB) is
    # ~15.5GB of STATIC memory before a single activation -- batch 2 OOM'd on a
    # 17GB slice with room to spare, and no batch size fixes that. Adafactor's
    # factored second moment drops the optimizer to megabytes, taking static to
    # ~12.4GB; checkpointing + batch 1 keeps activations near 1GB. ~13.5GB total.
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16  # 1 * 16 = 16 effective batch, unchanged
    PER_DEVICE_EVAL_BATCH_SIZE = 1  # generate() holds a KV cache -- 2 OOMs at eval
    GRADIENT_CHECKPOINTING = True  # no longer optional at this size
    OPTIM = "adafactor"  # NOT free: different update rule to AdamW, so the loss
    # curve won't match a medium run. It's the only lever that touches static
    # memory. Put "adamw_bnb_8bit" back the moment you get >25GB of the card.
    VRAM_BUDGET_GB = 14  # what the above actually needs; the startup check uses it
    EVAL_SUBSET = 500  # 2000 cost 3.1 hours per epoch at batch 1; 500 ranks
    # checkpoints just as well in ~45 minutes
    GENERATION_MAX_LEN = 96  # ceiling only -- the real value is measured from the
    # eval labels just before the Trainer is built, and is usually far lower

# Applied after the per-variant block above so the env value always wins.
if "WHISPER_EVAL_SUBSET" in os.environ:
    EVAL_SUBSET = int(os.environ["WHISPER_EVAL_SUBSET"])

MODEL_NAME = f"openai/whisper-{MODEL_VARIANT}"

# --- HF repos: derived from one user + one run name, so switching MODEL_VARIANT
# or account is a single edit and the checkpoints/final repos can never drift
# apart. Both are created on first push if they don't exist.
HF_USER = "milanakdj"
RUN_NAME = f"whisper-{MODEL_VARIANT}-nepali"
CKPT_REPO_ID = f"{HF_USER}/{RUN_NAME}-checkpoints"  # per-epoch backups (cell 10/11)
FINAL_REPO_ID = f"{HF_USER}/{RUN_NAME}-final-largev3_v2"  # trained model + model card (cell 14)
# Public: free accounts have a small PRIVATE storage quota, and whisper-medium
# checkpoints are several GB each. A private checkpoints repo hits
# "403 Private repository storage limit reached" partway through training.
CKPT_PRIVATE = False
FINAL_PRIVATE = False
# -------------------------------------------------------------------------

print(
    f"Fine-tuning {MODEL_NAME} | task={TASK} | lr={LEARNING_RATE} | "
    f"effective_batch={PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)
print(f"Checkpoints -> {CKPT_REPO_ID} | Final -> {FINAL_REPO_ID}")


def check_gpu_budget():
    """Preprocessing takes ~25 minutes before a single training step runs, so a
    GPU that is already full costs half an hour to discover. Check it now.

    This is a shared machine -- torch reports free memory across ALL processes,
    so another user's job shows up here as missing VRAM."""
    if not torch.cuda.is_available():
        print("[gpu] no CUDA device -- training will be unusably slow on CPU")
        return
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb, total_gb = free_b / 1e9, total_b / 1e9
    print(
        f"[gpu] {torch.cuda.get_device_name(0)}: "
        f"{free_gb:.1f}GB free / {total_gb:.1f}GB total "
        f"({total_gb - free_gb:.1f}GB held by other processes)",
        flush=True,
    )
    if free_gb < VRAM_BUDGET_GB:
        print(
            f"\n  WARNING: only {free_gb:.1f}GB free, config is sized for "
            f"{VRAM_BUDGET_GB}GB.\n"
            f"  Either wait for the other job, or lower PER_DEVICE_TRAIN_BATCH_SIZE\n"
            f"  (currently {PER_DEVICE_TRAIN_BATCH_SIZE}) and raise "
            f"GRADIENT_ACCUMULATION_STEPS by the same\n"
            f"  factor to keep the effective batch at "
            f"{PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}.\n",
            flush=True,
        )


check_gpu_budget()


def print_disk_usage(label=""):
    """Quick disk usage check -- call this at key milestones so you can see
    when/if you're approaching your disk quota, instead of finding out
    only after something silently stalls."""
    check_path = OUTPUT_DIR if os.path.isdir(OUTPUT_DIR) else os.path.expanduser("~")
    total, used, free = shutil.disk_usage(check_path)
    print(
        f"[disk]{' ' + label if label else ''}: "
        f"{used / 1e9:.1f}GB used / {total / 1e9:.1f}GB total "
        f"({free / 1e9:.1f}GB free)",
        flush=True,
    )


print_disk_usage("at start")

# %% Cell 4
print(f"Loading {NUM_ROWS} rows from {HF_DATASET_ID} ({HF_SPLIT} split)...")

# Slice syntax pulls only the requested rows instead of downloading the full 177k-row
# dataset -- much faster and lighter on Kaggle disk/session time.
full_dataset = load_dataset(
    HF_DATASET_ID,
    split=f"{HF_SPLIT}[:{NUM_ROWS}]",
    token=HF_TOKEN,
)
print(f"Loaded {len(full_dataset)} rows. Columns: {full_dataset.column_names}")

# drop rows with missing/empty transcripts
full_dataset = full_dataset.filter(
    lambda ex: ex["text"] is not None and ex["text"].strip() != ""
)
print(f"Usable examples after cleaning: {len(full_dataset)}")

# shuffle, then carve 80:10:10
full_dataset = full_dataset.shuffle(seed=SEED)

n = len(full_dataset)
n_train = int(n * TRAIN_VAL_TEST_SPLIT[0])
n_val = int(n * TRAIN_VAL_TEST_SPLIT[1])

raw_datasets = DatasetDict(
    {
        "train": full_dataset.select(range(0, n_train)),
        "validation": full_dataset.select(range(n_train, n_train + n_val)),
        "test": full_dataset.select(range(n_train + n_val, n)),
    }
)
print(
    f"train={len(raw_datasets['train'])}  "
    f"val={len(raw_datasets['validation'])}  "
    f"test={len(raw_datasets['test'])}"
)
print_disk_usage("after dataset load")

# Whisper expects 16kHz mono audio; this resamples on the fly when loading.
# The Hub dataset's audio column is named "audio" (not "audio_path").
raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))

# %% Cell 5
# IMPORTANT: we deliberately do NOT load the model here. Loading the model
# initializes a CUDA context, and datasets.map(num_proc>1) forks worker
# processes -- forking after CUDA init causes a silent deadlock (workers hang
# at 0% CPU forever, no error). So: tokenizer/feature_extractor first,
# preprocessing next, model load happens afterward in cell [6b].
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

# %% Cell 6
# Auto-detect whether we're inside a Jupyter kernel (deadlock risk with num_proc>1)
# or a plain terminal script (safe to use most of the machine's CPU cores).
# This matters a lot more now: 177k examples single-process would take a very
# long time on this dataset, vs. minutes with real parallelism on 20 CPU cores.
IS_NOTEBOOK = "ipykernel" in sys.modules
PREPROCESS_NUM_PROC = 5 #1 if IS_NOTEBOOK else max(1, (os.cpu_count() or 1) - 2)
print(
    f"[preprocess] environment={'notebook' if IS_NOTEBOOK else 'script'} -- "
    f"using num_proc={PREPROCESS_NUM_PROC}",
    flush=True,
)

# Kaggle's "Save Version" (background) log capture doesn't render tqdm bars, so we
# force plain-text logging and print explicit milestones with flush=True instead.
hf_datasets_module.disable_progress_bar()

MAX_LABEL_LENGTH = 448  # Whisper decoder max target length


def prepare_example(batch):
    audio = batch["audio"]

    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    labels = tokenizer(batch["text"]).input_ids
    batch["labels"] = labels
    return batch


print(
    "[preprocess] starting feature extraction + tokenization on all splits...",
    flush=True,
)
_t0 = time.time()

vectorized_datasets = DatasetDict()
for split_name, split_ds in raw_datasets.items():
    print(
        f"[preprocess] {split_name}: {len(split_ds)} examples -- processing now "
        f"(no live bar; this prints once at start and once at finish per split)",
        flush=True,
    )
    _split_t0 = time.time()
    vectorized_datasets[split_name] = split_ds.map(
        prepare_example,
        remove_columns=split_ds.column_names,
        num_proc=PREPROCESS_NUM_PROC,  # auto-detected above: 1 if notebook (avoids the
        # known Jupyter fork-deadlock), else most of the machine's CPU cores.
    )
    print(
        f"[preprocess] {split_name}: done in {time.time() - _split_t0:.1f}s", flush=True
    )

print(f"[preprocess] all splits done in {time.time() - _t0:.1f}s total", flush=True)


# drop any example whose tokenized label is too long for the decoder
def is_valid_length(labels):
    return len(labels) <= MAX_LABEL_LENGTH


print("[preprocess] filtering overlength labels...", flush=True)
_t0 = time.time()
for split_name in list(vectorized_datasets.keys()):
    before = len(vectorized_datasets[split_name])
    vectorized_datasets[split_name] = vectorized_datasets[split_name].filter(
        is_valid_length, input_columns=["labels"]
    )
    after = len(vectorized_datasets[split_name])
    print(f"[preprocess] {split_name}: kept {after}/{before} examples", flush=True)
print(f"[preprocess] filtering done in {time.time() - _t0:.1f}s", flush=True)
print_disk_usage("after preprocessing")

# Frees the raw decoded-audio cache, which is no longer needed once everything is
# log-mel features + token ids. This was essential on Kaggle's ~20GB quota.
#
# It is OFF by default here because it is blunter than it looks:
# cleanup_cache_files() deletes every cache-*.arrow in the dataset's cache
# directory except the ones raw_datasets itself is using -- and the map() output
# backing vectorized_datasets sits in that same directory. Deleting it costs
# nothing this run (the files stay open until the process exits) but throws away
# the ~25 minutes of preprocessing, so the next run redoes all of it from scratch.
# Leave it False on a machine with disk to spare and restarts get cheap.
CLEANUP_RAW_CACHE = False

if CLEANUP_RAW_CACHE:
    print("[cleanup] removing raw audio dataset cache...", flush=True)
    raw_datasets.cleanup_cache_files()
else:
    print(
        "[cleanup] skipped (CLEANUP_RAW_CACHE=False) -- keeps the preprocessing "
        "cache so a restart doesn't redo it",
        flush=True,
    )
del raw_datasets, full_dataset
gc.collect()
print_disk_usage("after cleanup")

# %% Cell 7
# multiprocessing (the map() calls above) has already finished.
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = (
    None  # required for recent transformers versions
)


# %% Cell 8
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip the leading <|startoftranscript|> the tokenizer added, because
        # Trainer re-adds it when it shifts labels into decoder_input_ids.
        #
        # This MUST compare against decoder_start_token_id (<|startoftranscript|>,
        # 50258). It used to compare against tokenizer.bos_token_id, which for
        # Whisper is <|endoftext|> (50257) -- a token that never appears first, so
        # the strip never happened. The model then trained on a doubled prefix
        # ([sot, sot, lang, task, notimestamps]) while generate() feeds the normal
        # single-sot prefix: every position off by one, output is fluent garbage,
        # WER in the hundreds, and the training loss looks perfect because training
        # is self-consistent. Do not "simplify" this back to bos_token_id.
        start = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        assert start == model.config.decoder_start_token_id, (
            f"decoder_start_token_id {model.config.decoder_start_token_id} is not "
            f"<|startoftranscript|> ({start}) -- label shifting would be wrong"
        )
        if (labels[:, 0] == start).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# %% Cell 9
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


# %% Cell 10
class PrintProgressCallback(TrainerCallback):
    """Prints plain flushed text on every log/eval/epoch event so progress is
    visible in Kaggle's 'Save Version' background logs, which don't render
    tqdm bars or the Trainer's rich console tables."""

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps if state.max_steps > 0 else "?"
        print(
            f"[train] starting -- total steps: {total_steps}, "
            f"epochs: {args.num_train_epochs}",
            flush=True,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        total = state.max_steps if state.max_steps > 0 else "?"
        parts = [f"step {step}/{total}"]
        for k in ("loss", "eval_loss", "eval_wer", "eval_cer", "learning_rate"):
            if k in logs:
                parts.append(f"{k}={logs[k]:.4f}")
        print(f"[train] {' | '.join(parts)}", flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """A WER above 100% with a low eval_loss means the model predicts perfectly
        under teacher forcing but cannot generate -- a decoder prefix / label
        alignment bug, not a weak model. The large-v3 run burned 80 hours before
        anyone read this number. Say it loudly at the FIRST eval instead."""
        if not metrics:
            return
        wer = metrics.get("eval_wer")
        loss = metrics.get("eval_loss")
        if wer is not None and wer > 100:
            print(
                "\n" + "!" * 78 + "\n"
                f"[ALARM] eval_wer={wer:.1f}% with eval_loss={loss:.4f}.\n"
                "  A WER above 100% is not a weak model -- generation is broken.\n"
                "  Teacher forcing hides it, so the loss looks fine and always will.\n"
                "  Usual cause: the labels still carry <|startoftranscript|>, so the\n"
                "  Trainer's shift produced a DOUBLED prefix and every position is off\n"
                "  by one. Check the strip in DataCollatorSpeechSeq2SeqWithPadding.\n"
                "  STOP THE RUN and fix it -- more epochs will not help.\n"
                + "!" * 78 + "\n",
                flush=True,
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        print(
            f"[train] epoch {state.epoch:.0f}/{args.num_train_epochs} complete "
            f"(step {state.global_step})",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        print(f"[train] training finished at step {state.global_step}", flush=True)


class PushCheckpointToHubCallback(TrainerCallback):
    """Uploads each epoch checkpoint to a HF model repo right after it's
    saved locally -- so a checkpoint survives even if the Kaggle session
    dies before training finishes. Runs on every save_strategy='epoch' save.
    """

    def __init__(self, repo_id, token, private=True):
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        # create_repo(exist_ok=True) does NOT change the settings of a repo that
        # already exists, so a repo created private by an earlier run stays
        # private and keeps hitting the private-storage quota. Force it.
        try:
            self.api.update_repo_settings(repo_id=repo_id, private=private)
        except AttributeError:  # huggingface_hub < 0.26
            self.api.update_repo_visibility(repo_id=repo_id, private=private)
        except Exception as e:
            print(f"[hf-backup] could not set visibility: {e}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return
        print(f"[hf-backup] uploading {ckpt_dir} to {self.repo_id} ...", flush=True)
        try:
            self.api.upload_folder(
                folder_path=ckpt_dir,
                path_in_repo=f"checkpoint-{state.global_step}",
                repo_id=self.repo_id,
                repo_type="model",
            )
        except Exception as e:
            # NEVER let a backup failure kill the run it is backing up. An HF
            # storage quota once destroyed a completed epoch this way. The local
            # checkpoint is already on disk (the Trainer writes it before this
            # callback fires), so training can safely continue and resume later.
            print(
                f"\n[hf-backup] FAILED, continuing anyway: {type(e).__name__}: {e}\n"
                f"[hf-backup] local checkpoint is intact at {ckpt_dir}\n",
                flush=True,
            )
            return
        print(
            f"[hf-backup] done: https://huggingface.co/{self.repo_id}/tree/main/checkpoint-{state.global_step}",
            flush=True,
        )


# predict_with_generate runs autoregressive decoding, so per-epoch eval on the
# full 17.7k validation split means ~17.7k generate() calls -- hours per epoch,
# for a number only used to rank checkpoints. A fixed subset gives the same
# ranking far cheaper. The split was already shuffled, so the first N are random,
# and select() is deterministic so epochs stay comparable. Cell 12 scores a larger
# TEST_SUBSET once at the end -- that is the number you report.
eval_subset = vectorized_datasets["validation"].select(
    range(min(EVAL_SUBSET, len(vectorized_datasets["validation"])))
)
print(f"[eval] validating on {len(eval_subset)} of {len(vectorized_datasets['validation'])} examples per epoch")

# Derive the generation cap from the labels instead of guessing. A hallucinating
# decode runs to this limit on EVERY example, so a cap 10x longer than the longest
# real transcript multiplies eval time by 10 for nothing. Measured on the eval
# subset itself, +8 tokens of headroom, and never above GENERATION_MAX_LEN.
_max_label = max(len(l) for l in eval_subset["labels"])
GENERATION_MAX_LEN = min(GENERATION_MAX_LEN, _max_label + 8)
print(f"[eval] longest eval label is {_max_label} tokens -> "
      f"generation_max_length={GENERATION_MAX_LEN}", flush=True)


training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_ratio=0.05,
    optim=OPTIM,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    # Reentrant checkpointing is deprecated in torch and is what breaks on most
    # recent builds; this kwarg is ignored when checkpointing is off.
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # Blackwell
    # supports bf16 natively -- more numerically stable than fp16
    # (no loss-scaling needed) and just as fast on this hardware.
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),  # fallback
    # only if bf16 isn't available (e.g. testing on older hardware)
    dataloader_num_workers=4,  # DGX Spark has 20 CPU cores -- use a few for data
    # loading so the GPU isn't waiting on audio decode between steps
    eval_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LEN,
    logging_steps=25,  # on_log fires every 25 steps -- adjust for more/less frequent prints
    disable_tqdm=False,  # bars ON: the eval bar is the only sign of life during a
    # multi-hour generate() pass. PrintProgressCallback still prints the milestones.
    report_to=["none"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=2,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vectorized_datasets["train"],
    eval_dataset=eval_subset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    # WHISPER_PUSH=0 keeps a smoke run from creating a junk checkpoints repo and
    # uploading gigabytes nobody wants.
    callbacks=(
        [PrintProgressCallback()]
        + (
            []
            if os.environ.get("WHISPER_PUSH") == "0"
            else [PushCheckpointToHubCallback(CKPT_REPO_ID, HF_TOKEN, private=CKPT_PRIVATE)]
        )
    ),
)

# %% Cell 11
# 1. Look for a checkpoint already sitting in this session's OUTPUT_DIR
resume_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    resume_checkpoint = get_last_checkpoint(OUTPUT_DIR)

# 2. If nothing local (e.g. fresh Kaggle session after a disconnect),
#    check the HF backup repo for the most recent checkpoint and pull it down.
if resume_checkpoint is None:
    print("[resume] no local checkpoint -- checking HF backup repo...", flush=True)
    try:
        api = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(CKPT_REPO_ID, repo_type="model")
        steps = sorted(
            {
                int(f.split("/")[0].split("-")[1])
                for f in files
                if f.startswith("checkpoint-")
            }
        )
        if steps:
            latest_step = steps[-1]
            print(
                f"[resume] found checkpoint-{latest_step} on HF -- downloading...",
                flush=True,
            )
            snapshot_download(
                repo_id=CKPT_REPO_ID,
                repo_type="model",
                allow_patterns=[f"checkpoint-{latest_step}/*"],
                local_dir=OUTPUT_DIR,
                token=HF_TOKEN,
            )
            resume_checkpoint = os.path.join(OUTPUT_DIR, f"checkpoint-{latest_step}")
            print(f"[resume] downloaded to {resume_checkpoint}", flush=True)
        else:
            print(
                "[resume] no checkpoints found on HF either -- starting fresh",
                flush=True,
            )
    except Exception as e:
        print(
            f"[resume] couldn't check HF backup repo ({e}) -- starting fresh",
            flush=True,
        )
else:
    print(f"[resume] found local checkpoint: {resume_checkpoint}", flush=True)

print_disk_usage("before training")
print("[train] entering trainer.train() ...", flush=True)
trainer.train(resume_from_checkpoint=resume_checkpoint)
print_disk_usage("after training")


# %% Cell 12
# The full 17.7k test split at eval batch 1 means 17.7k autoregressive generate()
# calls on large-v3 -- 20+ hours, and it looks like a hang because tqdm is the only
# output. 2000 examples is plenty for a reported WER. Training is over, so no
# optimizer/grad memory is held: a bigger eval batch fits now.
TEST_SUBSET = 2000
trainer.args.per_device_eval_batch_size = max(PER_DEVICE_EVAL_BATCH_SIZE, 8)
test_set = vectorized_datasets["test"].select(
    range(min(TEST_SUBSET, len(vectorized_datasets["test"])))
)
print(
    f"[test] evaluating {len(test_set)} of {len(vectorized_datasets['test'])} examples "
    f"at batch {trainer.args.per_device_eval_batch_size}",
    flush=True,
)
test_metrics = trainer.evaluate(
    eval_dataset=test_set,
    metric_key_prefix="test",
)
print(f"\n=== {MODEL_NAME} TEST RESULTS ===")
print(f"Test WER: {test_metrics['test_wer']:.2f}%")
print(f"Test CER: {test_metrics['test_cer']:.2f}%")

# %% Cell 13
final_dir = os.path.join(OUTPUT_DIR, f"final-{MODEL_VARIANT}")
trainer.save_model(final_dir)
processor.save_pretrained(final_dir)
print(f"Saved fine-tuned model to: {final_dir}")
print_disk_usage("after saving final model")

# %% Cell 14
def build_model_card(repo_id, wer=None, cer=None, steps=None, best_wer=None):
    """Model card recording every setting this run actually used, so a repo on the
    Hub is self-documenting -- you can tell months later exactly what produced it
    without digging for the script version. Values are read from the live config,
    never retyped."""
    eff_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    precision = "bf16" if training_args.bf16 else ("fp16" if training_args.fp16 else "fp32")
    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"

    return f"""---
language:
- ne
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- nepali
- asr
- speech-recognition
- finetuned
base_model: {MODEL_NAME}
datasets:
- {HF_DATASET_ID}
metrics:
- wer
- cer
---

# Whisper {MODEL_VARIANT} -- Nepali ASR (fine-tuned)

Nepali fine-tune of [`{MODEL_NAME}`](https://huggingface.co/{MODEL_NAME}) for `{TASK}`.

## Results

| Metric | Value |
|---|---|
| Test WER | {fmt(wer)}% |
| Test CER | {fmt(cer)}% |
| Best eval WER | {fmt(best_wer)}% |
| Steps trained | {steps if steps is not None else "n/a"} |

## Training configuration

| Parameter | Value |
|---|---|
| Base model | `{MODEL_NAME}` |
| Language / task | `{LANGUAGE}` / `{TASK}` |
| Epochs | {NUM_EPOCHS} |
| Learning rate | {LEARNING_RATE:.0e} |
| LR schedule | linear with {training_args.warmup_ratio:.0%} warmup ratio |
| Per-device train batch | {PER_DEVICE_TRAIN_BATCH_SIZE} |
| Grad accumulation steps | {GRADIENT_ACCUMULATION_STEPS} |
| Effective batch size | {eff_batch} |
| Per-device eval batch | {PER_DEVICE_EVAL_BATCH_SIZE} |
| Precision | {precision} |
| Gradient checkpointing | {training_args.gradient_checkpointing} |
| Optimizer | {training_args.optim} |
| Weight decay | {training_args.weight_decay} |
| Max grad norm | {training_args.max_grad_norm} |
| Max label length | {MAX_LABEL_LENGTH} tokens |
| Generation max length | {training_args.generation_max_length} |
| Eval / save strategy | per epoch (best model by WER kept) |
| Seed | {SEED} |

## Data

| Parameter | Value |
|---|---|
| Dataset | [`{HF_DATASET_ID}`](https://huggingface.co/datasets/{HF_DATASET_ID}) |
| Rows requested | {NUM_ROWS} |
| Split | {TRAIN_VAL_TEST_SPLIT[0]:.0%} / {TRAIN_VAL_TEST_SPLIT[1]:.0%} / {TRAIN_VAL_TEST_SPLIT[2]:.0%} (train/val/test) |
| Train / val / test examples | {len(vectorized_datasets["train"])} / {len(vectorized_datasets["validation"])} / {len(vectorized_datasets["test"])} |
| Per-epoch validation subset | {min(EVAL_SUBSET, len(vectorized_datasets["validation"]))} (test metrics use {min(TEST_SUBSET, len(vectorized_datasets["test"]))} test examples) |
| Audio sampling rate | 16 kHz mono |
| Checkpoint backups | `{CKPT_REPO_ID}` |

The training corpus is clean single-speaker studio audio, so expect degraded accuracy
on noisy real-world recordings with background noise, multiple speakers, or strong accents.

## Usage

```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model_id = "{repo_id}"
device   = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(model_id, language="{LANGUAGE}", task="{TASK}")
model     = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)

# audio: 1-D float32 numpy array at 16kHz
inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
with torch.inference_mode():
    ids = model.generate(inputs.input_features, max_new_tokens=225)

print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```
"""


# The card is written to disk BEFORE any network call. The large-v3 run died right
# here on a RemoteDisconnected from huggingface.co, after 80 hours of training and
# 29 hours of test eval -- everything after this point was lost to one TCP reset.
# Weights and card are already on local disk; the push is retried, and a failure
# prints the manual command instead of raising.
card = build_model_card(
    FINAL_REPO_ID,
    wer=test_metrics["test_wer"],
    cer=test_metrics["test_cer"],
    steps=trainer.state.global_step,
    best_wer=trainer.state.best_metric,
)
card_path = os.path.join(final_dir, "README.md")
with open(card_path, "w", encoding="utf-8") as f:
    f.write(card)
print(f"[push] card written to {card_path}", flush=True)


def push_final(attempts=3):
    """Returns True on success. Never raises -- the run is already finished and the
    artifacts are on disk, so a network error is a nuisance, not a failure."""
    api = HfApi(token=HF_TOKEN)
    for attempt in range(1, attempts + 1):
        try:
            api.create_repo(
                FINAL_REPO_ID, repo_type="model", exist_ok=True, private=FINAL_PRIVATE
            )
            model.push_to_hub(FINAL_REPO_ID, private=FINAL_PRIVATE)
            processor.push_to_hub(FINAL_REPO_ID, private=FINAL_PRIVATE)
            # After the model: push_to_hub writes its own stub card, so ours lands last.
            api.upload_file(
                path_or_fileobj=card_path,
                path_in_repo="README.md",
                repo_id=FINAL_REPO_ID,
                commit_message=f"Model card -- test WER {test_metrics['test_wer']:.2f}%",
            )
            print(f"Pushed to https://huggingface.co/{FINAL_REPO_ID}", flush=True)
            return True
        except Exception as e:
            print(f"[push] attempt {attempt}/{attempts} failed: "
                  f"{type(e).__name__}: {e}", flush=True)
            if attempt < attempts:
                time.sleep(30 * attempt)
    print(
        f"\n[push] could not reach the Hub. NOTHING IS LOST -- the model is at\n"
        f"  {final_dir}\n"
        f"and the card at\n  {card_path}\n"
        f"Upload it by hand when the network is back:\n"
        f"  HF_TOKEN=... hf upload {FINAL_REPO_ID} {final_dir} . \\\n"
        f"    --commit-message 'test WER {test_metrics['test_wer']:.2f}%'\n",
        flush=True,
    )
    return False


push_final()

# %% Cell 15
sample = vectorized_datasets["test"].select(
    range(min(5, len(vectorized_datasets["test"])))
)
inputs = data_collator([sample[i] for i in range(len(sample))])
with torch.no_grad():
    generated_ids = model.generate(
        input_features=inputs["input_features"].to(model.device),
        max_new_tokens=GENERATION_MAX_LEN,
    )
preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
labels = inputs["labels"].clone()
labels[labels == -100] = tokenizer.pad_token_id
refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

for i, (p, r) in enumerate(zip(preds, refs)):
    print(f"\n[{i}] REF : {r}")
    print(f"[{i}] PRED: {p}")

# %% Cell 16
# Standalone utility -- promote one specific backed-up checkpoint into the final
# repo. Use this when a session died after a good epoch and you never reached
# cell 14. Repo ids come from cell 3; only pick which checkpoint you want.
CHECKPOINT = os.environ.get("PROMOTE_CHECKPOINT")
if not CHECKPOINT:
    print("[promote] set PROMOTE_CHECKPOINT=checkpoint-N to promote a backup; skipping")
    raise SystemExit(0)

print(f"Downloading {CHECKPOINT} from {CKPT_REPO_ID}...")
local_dir = snapshot_download(
    repo_id=CKPT_REPO_ID,
    token=HF_TOKEN,
    allow_patterns=[f"{CHECKPOINT}/*"],
    local_dir=os.path.join(os.path.expanduser("~"), "final_checkpoint_download"),
)
checkpoint_path = os.path.join(local_dir, CHECKPOINT)
if not os.path.isfile(os.path.join(checkpoint_path, "config.json")):
    raise SystemExit(
        f"{CHECKPOINT} not in {CKPT_REPO_ID} -- check the repo's file list for the "
        f"available checkpoint-N dirs"
    )
print(f"Downloaded to: {checkpoint_path}")

# Load model + processor straight from the checkpoint -- everything needed
# (tokenizer, processor config, model weights) is already in there
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
processor = WhisperProcessor.from_pretrained(checkpoint_path)

print(f"Pushing to {FINAL_REPO_ID}...")
model.push_to_hub(FINAL_REPO_ID, private=FINAL_PRIVATE, token=HF_TOKEN)
processor.push_to_hub(FINAL_REPO_ID, private=FINAL_PRIVATE, token=HF_TOKEN)

# Same card as cell 14, minus the test metrics (no eval was run on this path) --
# the step count comes from the checkpoint name.
card_path = os.path.join(checkpoint_path, "README.md")
with open(card_path, "w", encoding="utf-8") as f:
    f.write(build_model_card(FINAL_REPO_ID, steps=int(CHECKPOINT.split("-")[1])))
HfApi(token=HF_TOKEN).upload_file(
    path_or_fileobj=card_path,
    path_in_repo="README.md",
    repo_id=FINAL_REPO_ID,
    commit_message=f"Model card -- promoted {CHECKPOINT}",
)

print(f"Done: https://huggingface.co/{FINAL_REPO_ID}")
