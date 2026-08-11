#!/usr/bin/env python3
# %% [markdown] Cell 0
# Whisper Fine-Tuning for Nepali ASR — Kaggle Notebook Script
# =============================================================
#
# Data source: HuggingFace Hub dataset `lilgoose7777/slr-combined-nepali-tts2`
# (177k rows, single "train" split, columns: audio, text, text_description).
# This script pulls a 20,000-row subset from it and splits 80:10:10 locally.
#
# NOTE: this dataset looks like single-speaker studio TTS audio (clean, one female
# voice, described via `text_description` captions), not noisy multi-speaker podcast
# recordings. It is a fine, fast dataset to prototype with, but a model fine-tuned
# only on this will likely underperform on messy real-world audio (background noise,
# multiple speakers, accents). Good for a first proof-of-concept run.
#
# Matches the configuration:
#   Model variants     : whisper-tiny / base / small / medium (set MODEL_VARIANT below)
#   Epochs              : 3
#   Learning rate       : 1e-5 (see LR_OVERRIDES note for tiny/base)
#   Batch size          : 7   (grad accumulation 2 -> effective batch 14)
#                         NOTE: automatically drops to 2/accum-7 for "medium" to
#                         avoid OOM on a T4's ~15GB VRAM -- same effective batch.
#   Split               : 80 / 10 / 10 (train/val/test), taken from a 20k-row subset
#   GPU                 : RTX 4090 / Kaggle GPU (fp16 enabled)
#   Metrics             : WER, CER
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
# 3. If the dataset is gated/private, add your HF token as a Kaggle Secret and set
#    HF_TOKEN below. If it is public, leave HF_TOKEN as None.
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
# - IMPORTANT: rotate HF_TOKEN below before sharing/re-uploading this file --
#   a live token pasted in plain text should be treated as compromised.
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
import numpy as np
import pandas as pd
import torch
import evaluate

from datasets import Dataset, Audio, DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# %% Cell 3
HF_DATASET_ID = "lilgoose7777/slr-combined-nepali-tts2"
HF_SPLIT = "train"  # this dataset only has one split; we carve our own below
HF_TOKEN = None  # set to a string (or use Kaggle Secrets) if the dataset is gated
# (must be None, not "", when unset -- an empty string
# gets sent as a literal "Bearer " auth header and errors out)
NUM_ROWS = 177000  # full dataset (was 20000 subset for the Kaggle run)
OUTPUT_DIR = os.path.join(
    os.path.expanduser("~"), "whisper-output"
)  # local path on DGX Spark

MODEL_VARIANT = "medium"  # one of: "tiny", "base", "small", "medium", "large-v3"
LANGUAGE = "nepali"  # Whisper's language tag for decoding
TASK = "transcribe"  # "transcribe" (ne->ne) or "translate" (ne audio -> en text)

TRAIN_VAL_TEST_SPLIT = (0.80, 0.10, 0.10)
SEED = 42

# --- Hyperparameters from your table ---
NUM_EPOCHS = 3
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

PER_DEVICE_TRAIN_BATCH_SIZE = 7
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch size = 7 * 2 = 14
PER_DEVICE_EVAL_BATCH_SIZE = 7

# The batch-size logic below was tuned for Kaggle's dual T4s (~15GB VRAM each,
# where even batch size 2 OOM'd on Medium, and DataParallel's gradient-gather
# added real overhead). DGX Spark is a different machine: ONE Blackwell GPU
# with 128GB unified memory shared between CPU and GPU -- no second GPU to
# restrict away, and far more headroom, so bumping batch size up and dropping
# the CUDA_VISIBLE_DEVICES restriction (there's only one GPU anyway).
if MODEL_VARIANT == "medium":
    PER_DEVICE_TRAIN_BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 1  # 16 * 1 = 16 effective batch
    PER_DEVICE_EVAL_BATCH_SIZE = 16
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # still
    # cheap insurance against fragmentation
    # If you hit an OOM anyway (unlikely with 128GB, but audio batches vary in
    # length), just lower PER_DEVICE_TRAIN_BATCH_SIZE and raise
    # GRADIENT_ACCUMULATION_STEPS to keep the same effective batch of 16.
elif MODEL_VARIANT == "large-v3":
    # ~1.5B params -- roughly 2x Medium's ~769M, so a smaller per-device batch,
    # made up for with gradient accumulation to keep the same effective batch
    # of 16. Still comfortably fits your 128GB unified memory with room to spare;
    # lower this further only if you actually hit an OOM.
    PER_DEVICE_TRAIN_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2  # 8 * 2 = 16 effective batch
    PER_DEVICE_EVAL_BATCH_SIZE = 8
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = f"openai/whisper-{MODEL_VARIANT}"
# -------------------------------------------------------------------------

print(
    f"Fine-tuning {MODEL_NAME} | task={TASK} | lr={LEARNING_RATE} | "
    f"effective_batch={PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)

import shutil


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

# --- Checkpoint backup/resume settings (added) ---
CKPT_REPO_ID = "lilgoose7777/whisper-medium-nepali-checkpoints"  # <-- set to your OWN HF repo id; must not be ""
# (if a friend is running this independently, they should
# just set this to their own repo id -- new checkpoints
# get pushed here, and resuming after a restart also
# reads from here, so it's fully self-contained per person)
CKPT_PRIVATE = True  # keep in-progress checkpoints private; flip the FINAL model's visibility separately in cell 14


# %% Cell 4
from datasets import load_dataset

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
PREPROCESS_NUM_PROC = 1 if IS_NOTEBOOK else max(1, (os.cpu_count() or 1) - 2)
print(
    f"[preprocess] environment={'notebook' if IS_NOTEBOOK else 'script'} -- "
    f"using num_proc={PREPROCESS_NUM_PROC}",
    flush=True,
)

import time
import datasets as hf_datasets_module

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

# Free disk space: the raw decoded-audio cache is no longer needed now that
# everything is converted into vectorized_datasets (log-mel features + token
# ids). This is what was filling up disk earlier -- clean it up before loading
# the model, so training doesn't risk hitting the quota mid-run.
print("[cleanup] removing raw audio dataset cache (no longer needed)...", flush=True)
raw_datasets.cleanup_cache_files()
del raw_datasets, full_dataset
import gc

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

        # strip the BOS token if the tokenizer already added one that the model re-adds
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
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
from transformers import TrainerCallback
from huggingface_hub import HfApi


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

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            print(f"[hf-backup] uploading {ckpt_dir} to {self.repo_id} ...", flush=True)
            self.api.upload_folder(
                folder_path=ckpt_dir,
                path_in_repo=f"checkpoint-{state.global_step}",
                repo_id=self.repo_id,
                repo_type="model",
            )
            print(
                f"[hf-backup] done: https://huggingface.co/{self.repo_id}/tree/main/checkpoint-{state.global_step}",
                flush=True,
            )


training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    warmup_ratio=0.05,
    gradient_checkpointing=True,
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
    generation_max_length=225,
    logging_steps=25,  # on_log fires every 25 steps -- adjust for more/less frequent prints
    disable_tqdm=True,  # suppress rich progress bars; we print explicitly instead
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
    eval_dataset=vectorized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    callbacks=[
        PrintProgressCallback(),
        PushCheckpointToHubCallback(CKPT_REPO_ID, HF_TOKEN, private=CKPT_PRIVATE),
    ],
)

# %% Cell 11
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import HfApi, snapshot_download

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
test_metrics = trainer.evaluate(
    eval_dataset=vectorized_datasets["test"],
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
from huggingface_hub import login

HF_REPO_ID = (
    "lilgoose7777/whisper-medium-nepali-final"  # will be created if it doesn't exist
)
HF_PRIVATE = False  # set True to keep the repo private

# Reuses HF_TOKEN from the config cell -- make sure that token has WRITE access,
# not just read (read-only tokens will fail here with a 403).
login(token=HF_TOKEN)

model.push_to_hub(HF_REPO_ID, private=HF_PRIVATE)
processor.push_to_hub(HF_REPO_ID, private=HF_PRIVATE)

print(f"Pushed to https://huggingface.co/{HF_REPO_ID}")

# %% Cell 15
sample = vectorized_datasets["test"].select(
    range(min(5, len(vectorized_datasets["test"])))
)
inputs = data_collator([sample[i] for i in range(len(sample))])
with torch.no_grad():
    generated_ids = model.generate(
        input_features=inputs["input_features"].to(model.device),
        max_new_tokens=225,
    )
preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
labels = inputs["labels"].clone()
labels[labels == -100] = tokenizer.pad_token_id
refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

for i, (p, r) in enumerate(zip(preds, refs)):
    print(f"\n[{i}] REF : {r}")
    print(f"[{i}] PRED: {p}")

# %% Cell 16
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor

CKPT_REPO_ID = "lilgoose7777/whisper-medium-nepali-checkpoints"
CHECKPOINT = "checkpoint-3429"
FINAL_REPO_ID = (
    "lilgoose7777/whisper-medium-nepali-final"  # new repo, created if missing
)
FINAL_PRIVATE = False
HF_TOKEN = None  # <-- SECURITY: a live token was hardcoded here in the original
# notebook. It has been removed. Set this via an environment
# variable or Kaggle Secret instead -- never hardcode a real
# token in a file you might share or upload. Rotate/revoke
# the original token immediately if you haven't already.

# Download just this one checkpoint folder
print(f"Downloading {CHECKPOINT} from {CKPT_REPO_ID}...")
local_dir = snapshot_download(
    repo_id=CKPT_REPO_ID,
    token=HF_TOKEN,
    allow_patterns=[f"{CHECKPOINT}/*"],
    local_dir=os.path.join(os.path.expanduser("~"), "final_checkpoint_download"),
)
checkpoint_path = f"{local_dir}/{CHECKPOINT}"
print(f"Downloaded to: {checkpoint_path}")

# Load model + processor straight from the checkpoint -- everything needed
# (tokenizer, processor config, model weights) is already in there
model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
processor = WhisperProcessor.from_pretrained(checkpoint_path)

# Push both to the new, clean "final" repo
print(f"Pushing to {FINAL_REPO_ID}...")
model.push_to_hub(FINAL_REPO_ID, private=FINAL_PRIVATE, token=HF_TOKEN)
processor.push_to_hub(FINAL_REPO_ID, private=FINAL_PRIVATE, token=HF_TOKEN)

print(f"Done: https://huggingface.co/{FINAL_REPO_ID}")
