# -*- coding: utf-8 -*-
"""
Indic Parler TTS — Nepali Finetuning
Supports resuming from a HuggingFace repo checkpoint.
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"]    = "1"
os.environ["TORCH_USE_CUDA_DSA"]      = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Guard spawn before any other imports that touch multiprocessing
import torch.multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

import gc
import os
import time
import uuid
import zipfile
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import wandb

from datasets import load_dataset, Audio
from huggingface_hub import login, HfApi
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    DacModel,
    get_cosine_schedule_with_warmup,
)
from parler_tts import ParlerTTSForConditionalGeneration

try:
    import bitsandbytes as bnb
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "bitsandbytes"], check=True)
    import bitsandbytes as bnb

# =============================================================================
# 1. Auth
# =============================================================================

hf_token = os.environ["HF_TOKEN"]
login(token=hf_token)

# =============================================================================
# 2. Config
# =============================================================================

OUTPUT_REPO   = "milanakdj/indic-parler-tts-nepali-finetuned-dgx-v12-cosine"
DATASET_REPO  = "Titung/nepali-tts-tagged-combined"
FINETUNE_BASE = "ai4bharat/indic-parler-tts-pretrained"

# Set True to load model weights from OUTPUT_REPO and continue training.
# Set False to start fresh from FINETUNE_BASE.
RESUME_FROM_HF = False

# Path to a local training_state.pt to restore optimizer/scheduler/step.
# Set to None to start from step 0 (even if RESUME_FROM_HF=True).
RESUME_STATE_PATH = None

NUM_EPOCHS       = 6
BATCH_SIZE       = 32
GRAD_ACCUM_STEPS = 2          # effective batch = 64
LEARNING_RATE    = 3e-6
WARMUP_STEPS     = 60
SAVE_STEPS       = 200        # validate + save best
ZIP_EVERY_STEPS  = 600        # periodic checkpoint zip
MAX_STEPS        = None       # override total steps (None = auto from epochs)
MAX_AUDIO_TOKENS = 550
TARGET_SR        = 44100      # DAC model sample rate
GPU_TAG          = "DGX"

RUN_UUID = uuid.uuid4().hex[:8]
# We intentionally ignore any description column that may exist in the dataset.
# All samples are trained with this single fixed description so the model learns
# to associate one consistent voice profile with Nepali speech. Change this
# string if you want a different target voice.
SPEAKER_DESCRIPTION = (
    "Amrita speaks with a clear, natural Nepali voice at a steady pace. "
    "The recording is of very high quality with no background noise."
)

ckpt_dir = f"./checkpoints_{RUN_UUID}"
os.makedirs(ckpt_dir, exist_ok=True)

print(f"Run UUID       : {RUN_UUID}")
print(f"Output repo    : {OUTPUT_REPO}")
print(f"Dataset        : {DATASET_REPO}")
print(f"Base model     : {FINETUNE_BASE}")
print(f"Resume from HF : {RESUME_FROM_HF}")
print(f"Resume state   : {RESUME_STATE_PATH}")
print(f"Eff. batch     : {BATCH_SIZE * GRAD_ACCUM_STEPS}")

# =============================================================================
# 3. Load model and tokenizers
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice : {device}")
if device == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

if RESUME_FROM_HF:
    print(f"\nResuming weights from : {OUTPUT_REPO}")
    model = ParlerTTSForConditionalGeneration.from_pretrained(OUTPUT_REPO).to(device)
    # Prompt tokenizer lives at the repo root.
    prompt_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_REPO)
    # Description tokenizer is the text-encoder tokenizer -- always load from
    # the text encoder's own name, not from the fine-tuned repo root, so that
    # vocab and special tokens stay consistent regardless of what was pushed.
    desc_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )
    print("Loaded finetuned weights.")
else:
    print(f"\nFresh start from : {FINETUNE_BASE}")
    model = ParlerTTSForConditionalGeneration.from_pretrained(FINETUNE_BASE).to(device)
    prompt_tokenizer = AutoTokenizer.from_pretrained(FINETUNE_BASE)
    desc_tokenizer   = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )
    print("Loaded base model.")

model.gradient_checkpointing_enable()
if hasattr(model, "decoder"):
    model.decoder.gradient_checkpointing_enable()
model.train()

SAMPLE_RATE   = model.config.sampling_rate
NUM_CODEBOOKS = model.config.decoder.num_codebooks

print(f"\nParams        : {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
print(f"Sampling rate : {SAMPLE_RATE} Hz")
print(f"Num codebooks : {NUM_CODEBOOKS}")

# =============================================================================
# 4. Load DAC encoder (frozen)
# =============================================================================

dac_model = DacModel.from_pretrained("ylacombe/dac_44khz").to(device)
dac_model.eval()
for p in dac_model.parameters():
    p.requires_grad = False

# =============================================================================
# 5. Audio encoding helper
# =============================================================================

def encode_audio(audio_array: np.ndarray, src_sr: int) -> torch.Tensor:
    """
    Encode a 1-D float32 numpy array to DAC tokens.

    Returns:
        codes: LongTensor of shape [NUM_CODEBOOKS, T]  (on CPU)
    """
    waveform = torch.from_numpy(audio_array.astype(np.float32)).unsqueeze(0)  # [1, T]
    if src_sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, src_sr, TARGET_SR)
    waveform = waveform.unsqueeze(0).to(device)   # [1, 1, T]

    with torch.no_grad():
        encoded = dac_model.encode(waveform)
        codes   = encoded.audio_codes[0]          # [n_codebooks, T]

    assert codes.shape[0] == 9, f"Expected 9 DAC codebooks, got {codes.shape[0]}"
    return codes.cpu()

# =============================================================================
# 6. Dataset
# =============================================================================

class NepaliTTSDataset(Dataset):
    def __init__(self, hf_dataset, text_col, audio_col):
        self.ds        = hf_dataset
        self.text_col  = text_col
        self.audio_col = audio_col

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        text   = sample[self.text_col].strip()

        prompt_enc = prompt_tokenizer(
            text, return_tensors="pt",
            max_length=128, truncation=True, padding="max_length",
        )

        # Always use the fixed speaker description -- see SPEAKER_DESCRIPTION above.
        # Per-sample description columns in the dataset are intentionally ignored.
        desc_enc = desc_tokenizer(
            SPEAKER_DESCRIPTION, return_tensors="pt",
            max_length=256, truncation=True, padding="max_length",
        )

        audio_info  = sample[self.audio_col]
        codes       = encode_audio(audio_info["array"], audio_info["sampling_rate"])
        # [NUM_CODEBOOKS, T]

        # Truncate and append EOS (token 1024)
        max_t   = MAX_AUDIO_TOKENS - 1
        codes   = codes[:, :max_t]
        eos_col = torch.full((codes.shape[0], 1), 1024, dtype=codes.dtype)
        codes   = torch.cat([codes, eos_col], dim=1)[:NUM_CODEBOOKS, :]

        return {
            "input_ids":           prompt_enc.input_ids.squeeze(0),
            "attention_mask":      prompt_enc.attention_mask.squeeze(0),
            "desc_input_ids":      desc_enc.input_ids.squeeze(0),
            "desc_attention_mask": desc_enc.attention_mask.squeeze(0),
            "audio_codes":         codes,
        }


def collate_fn(batch):
    max_t    = max(b["audio_codes"].shape[1] for b in batch)
    num_cb   = batch[0]["audio_codes"].shape[0]
    padded, masks = [], []
    for b in batch:
        T   = b["audio_codes"].shape[1]
        pad = max_t - T
        padded.append(F.pad(b["audio_codes"], (0, pad), value=1025))
        masks.append(F.pad(torch.ones(T, dtype=torch.long), (0, pad), value=0))

    decoder_input_ids = torch.stack(padded)         # [B, num_cb, max_t]
    decoder_attn_mask = torch.stack(masks)          # [B, max_t]

    labels = decoder_input_ids.permute(0, 2, 1).clone()  # [B, max_t, num_cb]
    labels[labels == 1025] = -100

    return {
        "input_ids":              torch.stack([b["input_ids"]          for b in batch]),
        "attention_mask":         torch.stack([b["attention_mask"]      for b in batch]),
        "desc_input_ids":         torch.stack([b["desc_input_ids"]      for b in batch]),
        "desc_attention_mask":    torch.stack([b["desc_attention_mask"] for b in batch]),
        "decoder_input_ids":      decoder_input_ids,
        "decoder_attention_mask": decoder_attn_mask,
        "labels":                 labels,
    }

# =============================================================================
# 7. Load dataset
# =============================================================================

print(f"\nLoading {DATASET_REPO}...")
raw_ds = load_dataset(DATASET_REPO, split="train")
print(f"Dataset size : {len(raw_ds)}")
print(f"Columns      : {raw_ds.column_names}")

cols     = raw_ds.column_names
TEXT_COL = next(
    (c for c in ["text", "transcription", "sentence", "nepali_text"] if c in cols),
    next(c for c in cols if "audio" not in c.lower()),
)
AUDIO_COL = next(
    (c for c in cols if "audio" in c.lower()),
    "audio",
)
print(f"Text col  : {TEXT_COL}")
print(f"Audio col : {AUDIO_COL}")

raw_ds = raw_ds.cast_column(AUDIO_COL, Audio(sampling_rate=SAMPLE_RATE))

split    = raw_ds.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
val_ds   = split["test"]
print(f"Train : {len(train_ds)} | Val : {len(val_ds)}")

train_dataset = NepaliTTSDataset(train_ds, TEXT_COL, AUDIO_COL)
val_dataset   = NepaliTTSDataset(val_ds,   TEXT_COL, AUDIO_COL)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, collate_fn=collate_fn,
)
print(f"Train batches : {len(train_loader)} | Val batches : {len(val_loader)}")

# =============================================================================
# 8. Sanity check
# =============================================================================

print("\nSanity check...")
sample_batch = {k: v.to(device) for k, v in next(iter(train_loader)).items()}
valid_labels = sample_batch["labels"][sample_batch["labels"] != -100]
assert valid_labels.numel() > 0, "All labels are -100 -- check audio encoding"

model.eval()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = model(
        input_ids=              sample_batch["desc_input_ids"],
        attention_mask=         sample_batch["desc_attention_mask"],
        prompt_input_ids=       sample_batch["input_ids"],
        prompt_attention_mask=  sample_batch["attention_mask"],
        decoder_attention_mask= sample_batch["decoder_attention_mask"],
        labels=                 sample_batch["labels"],
    )
print(f"Sanity loss : {out.loss.item():.4f}  -- OK")
model.train()
del sample_batch

# =============================================================================
# 9. Optimizer and scheduler
# =============================================================================

model = model.to(torch.bfloat16)

optimizer = bnb.optim.AdamW8bit(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE, weight_decay=0.01, eps=1e-8,
)

steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
total_steps     = MAX_STEPS if MAX_STEPS is not None else NUM_EPOCHS * steps_per_epoch

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)

print(f"\nLR           : {LEARNING_RATE:.1e}")
print(f"Warmup steps : {WARMUP_STEPS}")
print(f"Total steps  : {total_steps}")

# =============================================================================
# 10. Restore training state (optimizer / scheduler / step counter)
# =============================================================================

global_step   = 0
best_val_loss = float("inf")
train_losses  = []

if RESUME_STATE_PATH and os.path.exists(RESUME_STATE_PATH):
    print(f"\nRestoring training state from : {RESUME_STATE_PATH}")
    state = torch.load(RESUME_STATE_PATH, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    global_step   = state["global_step"]
    best_val_loss = state["best_val_loss"]
    train_losses  = state["train_losses"]
    print(f"  Resumed at step {global_step} | best_val_loss={best_val_loss:.4f}")

    if global_step >= total_steps:
        extra        = NUM_EPOCHS * steps_per_epoch
        total_steps  = global_step + extra
        scheduler    = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps,
        )
        print(f"  Extended total_steps to {total_steps}")
elif RESUME_FROM_HF:
    print("\nWeights loaded from HF but no RESUME_STATE_PATH -- optimizer starts fresh.")
else:
    print("\nFresh training run.")

print(f"Starting from step {global_step} / {total_steps}")

# =============================================================================
# 11. W&B
# =============================================================================

run_name = (
    f"resume-bf16-bs{BATCH_SIZE}-lr{LEARNING_RATE:.0e}-{RUN_UUID}"
    if RESUME_FROM_HF
    else f"fresh-bf16-bs{BATCH_SIZE}-lr{LEARNING_RATE:.0e}-{RUN_UUID}"
)
wandb.init(
    project="tts",
    entity="himalaya-ai-lab",
    name=run_name,
    config=dict(
        base_model=FINETUNE_BASE,
        resume=RESUME_FROM_HF,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum=GRAD_ACCUM_STEPS,
        lr=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        total_steps=total_steps,
        max_audio_tokens=MAX_AUDIO_TOKENS,
    ),
)

# =============================================================================
# 12. Training loop
# =============================================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.set_float32_matmul_precision("high")


def save_checkpoint(path: str):
    """Save model + both tokenizers to path."""
    model.eval()
    model.save_pretrained(path)
    prompt_tokenizer.save_pretrained(path)
    # Keep desc_tokenizer in a dedicated subfolder so it is never confused
    # with the prompt tokenizer and is easy to find on resume.
    desc_tokenizer.save_pretrained(os.path.join(path, "desc_tokenizer"))
    model.train()


def zip_and_delete(src_dir: str, step: int) -> str:
    zip_path = f"{ckpt_dir}/checkpoint_step_{step}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for fname in files:
                fpath   = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, start=os.path.dirname(src_dir))
                zf.write(fpath, arcname)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"  Zipped {zip_path} ({size_mb:.1f} MB)")
    shutil.rmtree(src_dir)
    return zip_path


print("\n" + "=" * 65)
print("STARTING FINETUNING")
print("=" * 65)

model.train()
optimizer.zero_grad()
start_time = time.time()
epoch      = 0

while global_step < total_steps:
    epoch += 1
    for batch_idx, batch in enumerate(train_loader):
        if global_step >= total_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out  = model(
                input_ids=              batch["desc_input_ids"],
                attention_mask=         batch["desc_attention_mask"],
                prompt_input_ids=       batch["input_ids"],
                prompt_attention_mask=  batch["attention_mask"],
                decoder_attention_mask= batch["decoder_attention_mask"],
                labels=                 batch["labels"],
            )
            loss = out.loss / GRAD_ACCUM_STEPS

        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
            continue

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        actual_loss = loss.item() * GRAD_ACCUM_STEPS
        train_losses.append(actual_loss)

        if global_step % 10 == 0:
            elapsed   = time.time() - start_time
            sps       = global_step / max(elapsed, 1e-9)
            eta_min   = (total_steps - global_step) / max(sps, 1e-9) / 60
            vram_gb   = torch.cuda.memory_allocated() / 1e9
            print(
                f"  Step {global_step:4d}/{total_steps} | "
                f"Loss: {actual_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"VRAM: {vram_gb:.1f} GB | "
                f"ETA: {eta_min:.1f} min"
            )
            wandb.log({
                "train/loss":    actual_loss,
                "train/lr":      scheduler.get_last_lr()[0],
                "train/vram_gb": vram_gb,
            }, step=global_step)

        # Validate and save best
        if global_step % SAVE_STEPS == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        val_out = model(
                            input_ids=              val_batch["desc_input_ids"],
                            attention_mask=         val_batch["desc_attention_mask"],
                            prompt_input_ids=       val_batch["input_ids"],
                            prompt_attention_mask=  val_batch["attention_mask"],
                            decoder_attention_mask= val_batch["decoder_attention_mask"],
                            labels=                 val_batch["labels"],
                        )
                    val_losses.append(val_out.loss.item())

            val_loss = float(np.mean(val_losses))
            print(f"\n  Step {global_step} | Val loss: {val_loss:.4f}")
            wandb.log({"val/loss": val_loss}, step=global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(f"{ckpt_dir}/best_checkpoint")
                print(f"  New best val loss: {best_val_loss:.4f}\n")

            model.train()
            torch.cuda.empty_cache()

        # Periodic checkpoint zip
        if global_step % ZIP_EVERY_STEPS == 0:
            periodic_dir = f"{ckpt_dir}/checkpoint_step_{global_step}"
            save_checkpoint(periodic_dir)
            torch.save({
                "epoch":         epoch,
                "global_step":   global_step,
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_losses":  train_losses,
            }, os.path.join(periodic_dir, "training_state.pt"))
            zip_and_delete(periodic_dir, global_step)
            torch.cuda.empty_cache()

    if global_step >= total_steps:
        break

training_time_min = (time.time() - start_time) / 60

print("\n" + "=" * 65)
print("FINETUNING COMPLETE")
print(f"  Epochs        : {epoch}")
print(f"  Steps         : {global_step}")
print(f"  Best val loss : {best_val_loss:.4f}")
print(f"  Time          : {training_time_min:.1f} min")
print("=" * 65)

# If no validation checkpoint was ever saved (e.g. SAVE_STEPS > total_steps),
# save the final state as the best checkpoint.
best_ckpt = f"{ckpt_dir}/best_checkpoint"
if not os.path.exists(best_ckpt):
    print("No best_checkpoint found -- saving final model.")
    save_checkpoint(best_ckpt)

wandb.finish()

# =============================================================================
# 13. Push to HuggingFace Hub
# =============================================================================

api = HfApi()

model_card = f"""---
language:
- ne
library_name: parler-tts
tags:
- text-to-speech
- nepali
- parler-tts
- finetuned
---

# Nepali Parler-TTS -- Finetuned

Nepali finetuned version of [Indic Parler-TTS]({FINETUNE_BASE}).

## Training config

| Parameter | Value |
|---|---|
| Base model | `{FINETUNE_BASE}` |
| Resumed from | `{OUTPUT_REPO if RESUME_FROM_HF else 'N/A'}` |
| GPU | {GPU_TAG} |
| Epochs | {NUM_EPOCHS} |
| Batch size | {BATCH_SIZE} |
| Grad accum steps | {GRAD_ACCUM_STEPS} |
| Effective batch size | {BATCH_SIZE * GRAD_ACCUM_STEPS} |
| Learning rate | {LEARNING_RATE} |
| LR schedule | cosine with warmup |
| Warmup steps | {WARMUP_STEPS} |
| Total steps | {total_steps} |
| Max audio tokens | {MAX_AUDIO_TOKENS} |
| Precision | bfloat16 |
| Grad clip norm | 1.0 |

## Results

| Metric | Value |
|---|---|
| Best val loss | {best_val_loss:.4f} |
| Steps trained | {global_step} |
| Training time | {training_time_min:.1f} min |

## Usage

```python
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import torch, soundfile as sf

model_id = "{OUTPUT_REPO}"
device   = "cuda" if torch.cuda.is_available() else "cpu"

model            = ParlerTTSForConditionalGeneration.from_pretrained(model_id).to(device)
prompt_tokenizer = AutoTokenizer.from_pretrained(model_id)
desc_tokenizer   = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

prompt      = "नमस्ते, तपाईंलाई कस्तो छ?"
description = "Shristi speaks clearly in Nepali at a steady pace. Very high quality audio."

desc_enc   = desc_tokenizer(description, return_tensors="pt").to(device)
prompt_enc = prompt_tokenizer(prompt,    return_tensors="pt").to(device)

with torch.inference_mode():
    gen = model.generate(
        input_ids=desc_enc.input_ids,
        attention_mask=desc_enc.attention_mask,
        prompt_input_ids=prompt_enc.input_ids,
        prompt_attention_mask=prompt_enc.attention_mask,
    )

sf.write("out.wav", gen.cpu().numpy().squeeze(), model.config.sampling_rate)
```
"""

with open(f"{best_ckpt}/README.md", "w") as f:
    f.write(model_card)

# Save training state alongside the best checkpoint so it can be used on the
# next resume via RESUME_STATE_PATH.
torch.save({
    "epoch":         epoch,
    "global_step":   global_step,
    "optimizer":     optimizer.state_dict(),
    "scheduler":     scheduler.state_dict(),
    "best_val_loss": best_val_loss,
    "train_losses":  train_losses,
}, f"{best_ckpt}/training_state.pt")

print(f"\nPushing to {OUTPUT_REPO} ...")
pushed_model = ParlerTTSForConditionalGeneration.from_pretrained(best_ckpt)
pushed_model.push_to_hub(OUTPUT_REPO, commit_message=f"best val_loss={best_val_loss:.4f}")
del pushed_model
gc.collect()

prompt_tokenizer.save_pretrained(best_ckpt)
AutoTokenizer.from_pretrained(best_ckpt).push_to_hub(
    OUTPUT_REPO, commit_message="Prompt tokenizer"
)
# Push desc_tokenizer separately so it is clearly distinct from prompt_tokenizer.
# On the next resume we reload it via model.config.text_encoder._name_or_path,
# so we do NOT push it to the repo root -- just note the source in the card.

api.upload_file(
    path_or_fileobj=f"{best_ckpt}/README.md",
    path_in_repo="README.md",
    repo_id=OUTPUT_REPO,
    commit_message="Add model card",
)
api.upload_file(
    path_or_fileobj=f"{best_ckpt}/training_state.pt",
    path_in_repo="training_state.pt",
    repo_id=OUTPUT_REPO,
    commit_message=f"Training state -- step={global_step}",
)

torch.cuda.empty_cache()
print(f"\nDone. Model pushed to https://huggingface.co/{OUTPUT_REPO}")