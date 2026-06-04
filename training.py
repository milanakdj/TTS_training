# -*- coding: utf-8 -*-
"""
Indic Parler TTS — Nepali Finetuning
Supports resuming from a HuggingFace repo checkpoint.
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"]   = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("✅ Environment variables set")

# !pip install -q git+https://github.com/huggingface/parler-tts.git
# !pip install -q transformers>=4.40.0 datasets soundfile librosa accelerate huggingface_hub bitsandbytes
# !pip install "protobuf>=5.29.1,<6.0.0" --upgrade -q
# !pip install wandb -q


import google.protobuf
print(f"protobuf: {google.protobuf.__version__}")

from transformers import AutoTokenizer, EncodecModel, AutoProcessor
from parler_tts import ParlerTTSForConditionalGeneration
print("✅ All imports OK")

import transformers
print(transformers.__version__)

# from huggingface_hub import notebook_login
# notebook_login()

# from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

# user_secrets = UserSecretsClient()
# hf_token = user_secrets.get_secret("HF_TOKEN")

hf_token = os.environ["HF_TOKEN"]
login(token=hf_token)

# =============================================================================
# ## 1. Config
# =============================================================================

OUTPUT_REPO   = "milanakdj/indic-parler-tts-nepali-finetuned-dgx-v7-cosine"
DATASET_REPO  = "Titung/nepali-tts-tagged-combined"
FINETUNE_BASE = "ai4bharat/indic-parler-tts-pretrained"

# ── RESUME CONFIG ─────────────────────────────────────────────────────────────
# Set RESUME_FROM_HF = True to download weights from OUTPUT_REPO and continue.
# Set RESUME_FROM_HF = False to start fresh from FINETUNE_BASE.
RESUME_FROM_HF = False

# If you have a local training_state.pt (optimizer + scheduler + step),
# point RESUME_STATE_PATH to it. Set to None to restart step count from 0.
# Example: "./checkpoints_abc123/checkpoint_step_600/training_state.pt"
RESUME_STATE_PATH = None
# ─────────────────────────────────────────────────────────────────────────────

MAX_STEPS        = None
NUM_EPOCHS       = 4
BATCH_SIZE       = 32
GRAD_ACCUM_STEPS = 2        # eff. batch = 64
LEARNING_RATE    = 3e-6
WARMUP_STEPS     = 300
SAVE_STEPS       = 200
ZIP_EVERY_STEPS  = 600
GPU              = "DGX"

MAX_AUDIO_SEC    = 20
MAX_AUDIO_TOKENS = 550

import uuid
RUN_UUID = uuid.uuid4().hex[:8]
print(f"🆔 Run UUID: {RUN_UUID}")

ckpt_dir = f"./checkpoints_{RUN_UUID}"
os.makedirs(ckpt_dir, exist_ok=True)

print(f"Output repo    : {OUTPUT_REPO}")
print(f"Dataset        : {DATASET_REPO}")
print(f"Base model     : {FINETUNE_BASE}")
print(f"Resume from HF : {RESUME_FROM_HF}")
print(f"Resume state   : {RESUME_STATE_PATH}")
print(f"Eff. batch     : {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"Warmup steps   : {WARMUP_STEPS}")

import wandb

wandb.init(
    project = "tts",
    entity  = "himalaya-ai-lab",
    name    = f"resume-gb10-bf16-bs32-lr5e6-{RUN_UUID}" if RESUME_FROM_HF else f"fresh-gb10-bf16-bs32-lr3e6-{RUN_UUID}",
    config  = { 
    }
)

# =============================================================================
# ## 2. Load Model & Tokenizers
# ── CHANGE: loads from OUTPUT_REPO if RESUME_FROM_HF=True ───────────────────
# =============================================================================

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

if RESUME_FROM_HF:
    # ── Resume: load finetuned weights from HuggingFace ──────────────────────
    print(f"\n⏩ Resuming from HuggingFace: {OUTPUT_REPO}")
    ft_model = ParlerTTSForConditionalGeneration.from_pretrained(
        OUTPUT_REPO         # ← your previously pushed checkpoint
    ).to(device)
    ft_tokenizer      = AutoTokenizer.from_pretrained(OUTPUT_REPO)
    ft_desc_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_REPO)
    # Note: desc_tokenizer is pushed to the same repo root in your push code,
    # so loading from OUTPUT_REPO is correct here.
    print(f"✅ Loaded finetuned weights from {OUTPUT_REPO}")
else:
    # ── Fresh start: load pretrained base model ───────────────────────────────
    print(f"\n🆕 Starting fresh from {FINETUNE_BASE}")
    ft_model = ParlerTTSForConditionalGeneration.from_pretrained(
        FINETUNE_BASE
    ).to(device)
    ft_tokenizer      = AutoTokenizer.from_pretrained(FINETUNE_BASE)
    ft_desc_tokenizer = AutoTokenizer.from_pretrained(
        ft_model.config.text_encoder._name_or_path
    )
    print(f"✅ Loaded base model from {FINETUNE_BASE}")

ft_model.gradient_checkpointing_enable()
if hasattr(ft_model, "decoder"):
    ft_model.decoder.gradient_checkpointing_enable()
ft_model.train()

parler_sample_rate = ft_model.config.sampling_rate
NUM_CODEBOOKS      = ft_model.config.decoder.num_codebooks

print(f"\n   Params            : {sum(p.numel() for p in ft_model.parameters())/1e6:.0f}M")
print(f"   Sampling rate     : {parler_sample_rate} Hz")
print(f"   Num codebooks     : {NUM_CODEBOOKS}")
print(f"   VRAM used         : {torch.cuda.memory_allocated()/1e9:.2f} GB")

# =============================================================================
# ## 3. Load DAC
# =============================================================================

import torchaudio
from transformers import AutoModel, DacModel

DAC_MODEL_ID = "ylacombe/dac_44khz"
audio_decoder = DacModel.from_pretrained(DAC_MODEL_ID).to(device)
audio_decoder.eval()
for param in audio_decoder.parameters():
    param.requires_grad = False

def encode_audio_with_dac(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, sr, 44100)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # DAC expects (batch, channels, time)
    waveform = waveform.unsqueeze(0).to(device)  # (1, 1, T)

    with torch.no_grad():
        encoded = audio_decoder.encode(waveform)

    # audio_codes shape: (batch, n_codebooks, time) → (9, T)
    audio_codes = encoded.audio_codes[0]

    assert audio_codes.shape[0] == 9, f"Expected 9 codebooks, got {audio_codes.shape[0]}"
    return audio_codes

"""## 4. Load & Prepare Dataset"""

from datasets import load_dataset, Audio
import numpy as np

print(f"Loading {DATASET_REPO}...")
nepali_ds = load_dataset(DATASET_REPO, split="train")
print(f"Dataset size : {len(nepali_ds)}")
print(f"Columns      : {nepali_ds.column_names}")

cols = nepali_ds.column_names
TEXT_COL = None
for c in ["text", "transcription", "sentence", "nepali_text"]:
    if c in cols:
        TEXT_COL = c
        break
if TEXT_COL is None:
    TEXT_COL = [c for c in cols if "audio" not in c.lower()][0]

AUDIO_COL = "audio" if "audio" in cols else [c for c in cols if "audio" in c.lower()][0]
print(f"Text column  : '{TEXT_COL}'")
print(f"Audio column : '{AUDIO_COL}'")

nepali_ds = nepali_ds.cast_column(AUDIO_COL, Audio(sampling_rate=parler_sample_rate))
nepali_ds_filtered = nepali_ds

split    = nepali_ds_filtered.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
val_ds   = split["test"]
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# =============================================================================
# ## 5. Dataset Class & Collator
# =============================================================================

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from transformers import DacModel

DESCRIPTION_TEMPLATES = [
    "A female speaker delivers clear Nepali speech at a normal pace. The recording is of very high quality.",
    "A female speaker speaks clearly and naturally in Nepali. The audio is clean with no background noise.",
    "A neutral speaker reads Nepali text at a steady pace. The recording quality is excellent.",
    "A female speaker delivers speech slowly and clearly in Nepali. Very high quality audio.",
]

TARGET_SR = 44100  # DAC 44khz


def encode_audio_with_dac(audio_array, sample_rate, dac_model, device):
    """
    Encode a 1D numpy audio array to DAC tokens.
    Returns codes: Tensor of shape [n_codebooks, T]
    """
    # Resample if needed
    if sample_rate != TARGET_SR:
        t = torch.tensor(audio_array).unsqueeze(0).float()
        t = torchaudio.functional.resample(t, sample_rate, TARGET_SR)
        audio_array = t.squeeze().numpy()

    audio_array = audio_array.astype(np.float32)

    # DAC expects (batch, channels, time)
    waveform = torch.tensor(audio_array).unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    waveform = waveform.to(device)

    with torch.no_grad():
        encoded = dac_model.encode(waveform)
        # audio_codes: [B=1, n_codebooks, T]
        codes = encoded.audio_codes[0]  # → [n_codebooks, T]

    return codes.cpu()  # [n_codebooks, T]  — should be [9, T]


class NepaliTTSDataset(Dataset):
    def __init__(self, hf_dataset, text_col, audio_col,
                 tokenizer, desc_tokenizer,
                 dac_model,           # ← renamed from encodec_model
                 device,
                 max_text_len=128,
                 max_audio_tokens=None):
        self.ds                = hf_dataset
        self.text_col          = text_col
        self.audio_col         = audio_col
        self.tokenizer         = tokenizer
        self.desc_tokenizer    = desc_tokenizer
        self.dac_model         = dac_model   # ← renamed
        self.device            = device
        self.max_text_len      = max_text_len
        self.max_audio_tokens  = max_audio_tokens or MAX_AUDIO_TOKENS

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample     = self.ds[idx]
        text       = sample[self.text_col].strip()
        audio_info = sample[self.audio_col]

        prompt_enc = self.tokenizer(
            text, return_tensors="pt",
            max_length=self.max_text_len, truncation=True, padding="max_length",
        )
        desc = "Shristi speaks with a deep, formal Nepali voice. Her speech is clear, steady and authoritative with natural pacing in a quiet noise-free environment."
        desc_enc = self.desc_tokenizer(
            desc, return_tensors="pt",
            max_length=256, truncation=True, padding="max_length",
        )

        # Encode audio with DAC
        audio_array = audio_info["array"]
        sr          = audio_info["sampling_rate"]
        codes = encode_audio_with_dac(
            audio_array, sr,
            self.dac_model,
            self.device
        )  # [9, T]

        # Truncate to max length (leave 1 slot for EOS)
        if codes.shape[1] > self.max_audio_tokens - 1:
            codes = codes[:, :self.max_audio_tokens - 1]
        eos_col = torch.full((codes.shape[0], 1), 1024, dtype=codes.dtype)
        codes   = torch.cat([codes, eos_col], dim=1)
        codes   = codes[:NUM_CODEBOOKS, :]

        return {
            "input_ids":           prompt_enc.input_ids.squeeze(0),
            "attention_mask":      prompt_enc.attention_mask.squeeze(0),
            "desc_input_ids":      desc_enc.input_ids.squeeze(0),
            "desc_attention_mask": desc_enc.attention_mask.squeeze(0),
            "audio_codes":         codes,
        }


print("✅ NepaliTTSDataset defined")
print(f"   NUM_CODEBOOKS    : {NUM_CODEBOOKS}")
print(f"   MAX_AUDIO_TOKENS : {MAX_AUDIO_TOKENS}")
print(f"   TARGET_SR        : {TARGET_SR}")

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # At very top of notebook

import torch.nn.functional as F
from torch.utils.data import DataLoader

def collate_fn(batch):
    max_audio_len = max(b["audio_codes"].shape[1] for b in batch)
    num_cb        = batch[0]["audio_codes"].shape[0]
    padded_codes, decoder_masks = [], []
    for b in batch:
        T   = b["audio_codes"].shape[1]
        pad = max_audio_len - T
        pc  = F.pad(b["audio_codes"], (0, pad), value=1025)
        padded_codes.append(pc)
        decoder_masks.append(F.pad(torch.ones(T, dtype=torch.long), (0, pad), value=0))
    decoder_input_ids = torch.stack(padded_codes)
    dec_mask          = torch.stack(decoder_masks)
    labels = decoder_input_ids.permute(0, 2, 1).clone()
    labels[labels == 1025] = -100
    return {
        "input_ids":              torch.stack([b["input_ids"]          for b in batch]),
        "attention_mask":         torch.stack([b["attention_mask"]      for b in batch]),
        "desc_input_ids":         torch.stack([b["desc_input_ids"]      for b in batch]),
        "desc_attention_mask":    torch.stack([b["desc_attention_mask"] for b in batch]),
        "decoder_input_ids":      decoder_input_ids,
        "decoder_attention_mask": dec_mask,
        "labels":                 labels,
    }


train_dataset = NepaliTTSDataset(train_ds, TEXT_COL, AUDIO_COL,
                                  ft_tokenizer, ft_desc_tokenizer, audio_decoder, device)
val_dataset   = NepaliTTSDataset(val_ds,   TEXT_COL, AUDIO_COL,
                                  ft_tokenizer, ft_desc_tokenizer, audio_decoder, device)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, collate_fn=collate_fn)

print(f"✅ Datasets ready")
print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
print(f"   Val  : {len(val_dataset)} samples,   {len(val_loader)} batches")

# =============================================================================
# ## 5b. Sanity Check
# =============================================================================

print("Running sanity check on one batch...")
sample_batch = next(iter(train_loader))
sample_batch_dev = {k: v.to(device) for k, v in sample_batch.items()}

real_labels = sample_batch_dev["labels"][sample_batch_dev["labels"] != -100]
print(f"  Valid label tokens : {real_labels.numel()}")
print(f"  labels all -100?   : {real_labels.numel() == 0}")

ft_model.eval()
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = ft_model(
            input_ids=              sample_batch_dev["desc_input_ids"],
            attention_mask=         sample_batch_dev["desc_attention_mask"],
            prompt_input_ids=       sample_batch_dev["input_ids"],
            prompt_attention_mask=  sample_batch_dev["attention_mask"],
            decoder_attention_mask= sample_batch_dev["decoder_attention_mask"],
            labels=                 sample_batch_dev["labels"],
        )
print(f"  Loss (bfloat16): {out.loss.item():.4f}")
print("✅ Sanity check PASSED")
ft_model.train()

# =============================================================================
# ## 6. Optimizer & Scheduler
# =============================================================================

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
try:
    import bitsandbytes as bnb
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "-q", "bitsandbytes"], check=True)
    import bitsandbytes as bnb

ft_model = ft_model.to(torch.bfloat16)

optimizer = bnb.optim.AdamW8bit(
    [p for p in ft_model.parameters() if p.requires_grad],
    lr=LEARNING_RATE, weight_decay=0.01, eps=1e-8
)

steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
MAX_TRAIN_STEPS = NUM_EPOCHS * steps_per_epoch
MAX_TRAIN_STEPS = MAX_STEPS if MAX_STEPS is not None else MAX_TRAIN_STEPS

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=MAX_TRAIN_STEPS,
)

print("✅ Optimizer + Scheduler ready (bfloat16)")
print(f"   Peak LR      : {LEARNING_RATE:.1e}")
print(f"   Warmup steps : {WARMUP_STEPS}")
print(f"   Total steps  : {MAX_TRAIN_STEPS}")

# =============================================================================
# ## 7. Resume Training State (optimizer + scheduler + step)
# ── CHANGE: restores optimizer/scheduler/step from local training_state.pt ──
# =============================================================================

global_step   = 0
best_val_loss = float("inf")
train_losses  = []

if RESUME_STATE_PATH is not None and os.path.exists(RESUME_STATE_PATH):
    print(f"\n⏩ Restoring training state from: {RESUME_STATE_PATH}")
    state = torch.load(RESUME_STATE_PATH, map_location=device)

    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    global_step   = state["global_step"]
    best_val_loss = state["best_val_loss"]
    train_losses  = state["train_losses"]

    print(f"  ✅ Restored at step {global_step} | best_val_loss={best_val_loss:.4f}")
    print(f"  ℹ️  Remaining steps: {MAX_TRAIN_STEPS - global_step}")

    # Extend scheduler if remaining steps exceed original schedule
    if global_step >= MAX_TRAIN_STEPS:
        print(f"  ⚠️  global_step ({global_step}) >= MAX_TRAIN_STEPS ({MAX_TRAIN_STEPS})")
        print(f"      Extending MAX_TRAIN_STEPS by {NUM_EPOCHS * steps_per_epoch} steps")
        MAX_TRAIN_STEPS = global_step + NUM_EPOCHS * steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,                  # no warmup for resumed run
            num_training_steps=MAX_TRAIN_STEPS,
        )
        print(f"      New MAX_TRAIN_STEPS: {MAX_TRAIN_STEPS}")
else:
    if RESUME_FROM_HF:
        print(f"\n⚠️  RESUME_FROM_HF=True but no RESUME_STATE_PATH provided.")
        print(f"    Model weights loaded from HF ✅")
        print(f"    Optimizer/scheduler/step starting FRESH (step=0)")
        print(f"    To resume step count too, set RESUME_STATE_PATH to your local training_state.pt")
    else:
        print(f"\n🆕 Fresh training run — no state to restore")

print(f"\n   Starting from step : {global_step}")
print(f"   Training until step: {MAX_TRAIN_STEPS}")

# =============================================================================
# ## 8. Training Loop
# =============================================================================

import time
import zipfile
import shutil

def zip_checkpoint(periodic_ckpt_dir, step):
    zip_name = f"{ckpt_dir}/checkpoint_step_{step}.zip"
    print(f"  🗜️  Zipping checkpoint → {zip_name} ...")
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(periodic_ckpt_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname   = os.path.relpath(file_path, start=os.path.dirname(periodic_ckpt_dir))
                zf.write(file_path, arcname)
    size_mb = os.path.getsize(zip_name) / 1e6
    print(f"  ✅  Saved {zip_name} ({size_mb:.1f} MB)\n")
    return zip_name

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.set_float32_matmul_precision("high")

print("=" * 65)
print("🏋️  STARTING FINETUNING")
print("=" * 65)

ft_model.train()
optimizer.zero_grad()
start_time = time.time()
epoch      = 0

while global_step < MAX_TRAIN_STEPS:
    epoch += 1

    for batch_idx, batch in enumerate(train_loader):

        if global_step >= MAX_TRAIN_STEPS:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = ft_model(
                input_ids=              batch["desc_input_ids"],
                attention_mask=         batch["desc_attention_mask"],
                prompt_input_ids=       batch["input_ids"],
                prompt_attention_mask=  batch["attention_mask"],
                decoder_attention_mask= batch["decoder_attention_mask"],
                labels=                 batch["labels"],
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS

        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in ft_model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            actual_loss = loss.item() * GRAD_ACCUM_STEPS
            train_losses.append(actual_loss)

            if global_step % 10 == 0:
                elapsed       = time.time() - start_time
                steps_per_sec = global_step / max(elapsed, 1e-9)
                remaining     = (MAX_TRAIN_STEPS - global_step) / max(steps_per_sec, 1e-9) / 60
                vram_gb       = torch.cuda.memory_allocated() / 1e9
                print(
                    f"  Step {global_step:4d}/{MAX_TRAIN_STEPS} | "
                    f"Loss: {actual_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"VRAM: {vram_gb:.1f}GB | "
                    f"ETA: {remaining:.1f}min"
                )
                wandb.log({
                    "train/loss":        actual_loss,
                    "train/lr":          scheduler.get_last_lr()[0],
                    "train/vram_gb":     vram_gb,
                    "train/global_step": global_step,
                }, step=global_step)

            if global_step % SAVE_STEPS == 0 and global_step > 0:
                ft_model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            val_out = ft_model(
                                input_ids=              val_batch["desc_input_ids"],
                                attention_mask=         val_batch["desc_attention_mask"],
                                prompt_input_ids=       val_batch["input_ids"],
                                prompt_attention_mask=  val_batch["attention_mask"],
                                decoder_attention_mask= val_batch["decoder_attention_mask"],
                                labels=                 val_batch["labels"],
                            )
                            val_losses.append(val_out.loss.item())

                val_loss = np.mean(val_losses)
                print(f"\n  📊 Step {global_step} | Val Loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss, "val/best_loss": best_val_loss}, step=global_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ft_model.save_pretrained(f"{ckpt_dir}/best_checkpoint")
                    ft_tokenizer.save_pretrained(f"{ckpt_dir}/best_checkpoint")
                    ft_desc_tokenizer.save_pretrained(f"{ckpt_dir}/best_checkpoint/desc_tokenizer")
                    print(f"  💾 New best! val_loss={best_val_loss:.4f} → {ckpt_dir}/best_checkpoint\n")

                ft_model.train()
                torch.cuda.empty_cache()

            if global_step % ZIP_EVERY_STEPS == 0 and global_step > 0:
                periodic_ckpt_dir = f"{ckpt_dir}/checkpoint_step_{global_step}"
                ft_model.eval()
                ft_model.save_pretrained(periodic_ckpt_dir)
                ft_tokenizer.save_pretrained(periodic_ckpt_dir)
                ft_desc_tokenizer.save_pretrained(periodic_ckpt_dir + "/desc_tokenizer")

                torch.save({
                    "epoch":         epoch,
                    "global_step":   global_step,
                    "optimizer":     optimizer.state_dict(),
                    "scheduler":     scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "train_losses":  train_losses,
                }, os.path.join(periodic_ckpt_dir, "training_state.pt"))

                zip_checkpoint(periodic_ckpt_dir, global_step)
                shutil.rmtree(periodic_ckpt_dir)
                ft_model.train()
                torch.cuda.empty_cache()

            if global_step >= MAX_TRAIN_STEPS:
                print(f"\n  🛑 Reached MAX_TRAIN_STEPS={MAX_TRAIN_STEPS}, stopping.")
                break

    if global_step >= MAX_TRAIN_STEPS:
        break

print("\n" + "=" * 65)
print("🎉 FINETUNING COMPLETE")
print(f"   Total epochs : {epoch}")
print(f"   Total steps  : {global_step}")
print(f"   Best val loss: {best_val_loss:.4f}")
print(f"   Time taken   : {(time.time() - start_time)/60:.1f} min")
print("=" * 65)

best_ckpt_path = f"{ckpt_dir}/best_checkpoint"
if not os.path.exists(best_ckpt_path):
    print("⚠️  No best_checkpoint found — saving final model as best...")
    ft_model.eval()
    ft_model.save_pretrained(best_ckpt_path)
    ft_tokenizer.save_pretrained(best_ckpt_path)
    ft_desc_tokenizer.save_pretrained(f"{best_ckpt_path}/desc_tokenizer")
    print(f"✅ Saved to {best_ckpt_path}")


wandb.finish()


# =============================================================================
# ## 9. Push to HuggingFace Hub
# =============================================================================

import gc
from parler_tts import ParlerTTSForConditionalGeneration
from huggingface_hub import HfApi
api       = HfApi()


best_ckpt = f"{ckpt_dir}/best_checkpoint"

# ── Model Card ────────────────────────────────────────────────────────────────
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

# Nepali Parler-TTS — Finetuned

Nepali finetuned version of Indic Parler-TTS.

## Training Configuration

| Parameter | Value |
|---|---|
| Base Model | Indic Parler-TTS |
| Resumed From | {OUTPUT_REPO if RESUME_FROM_HF else 'N/A (fresh run)'} |
| GPU | {GPU} |
| Epochs | {NUM_EPOCHS} |
| Batch Size | {BATCH_SIZE} |
| Gradient Accumulation Steps | {GRAD_ACCUM_STEPS} |
| Effective Batch Size | {BATCH_SIZE * GRAD_ACCUM_STEPS} |
| Learning Rate | {LEARNING_RATE} |
| Warmup Steps | {WARMUP_STEPS} |
| Max Train Steps | {MAX_TRAIN_STEPS} |
| Save Steps | {SAVE_STEPS} |
| Zip Every Steps | {ZIP_EVERY_STEPS} |
| Max Audio Seconds | {MAX_AUDIO_SEC} |
| Max Audio Tokens | {MAX_AUDIO_TOKENS} |
| Precision | bfloat16 |
| Grad Clip Norm | 1.0 |

## Results

| Metric | Value |
|---|---|
| Best Validation Loss | {best_val_loss:.4f} |
| Total Steps Trained | {global_step} |
| Training Time | {(time.time() - start_time)/60:.1f} min |

## W&B Run
[View training run](https://wandb.ai/himalaya-ai-lab/nanochat)
"""

# Write README into the checkpoint folder so it gets pushed
with open(f"{best_ckpt}/README.md", "w") as f:
    f.write(model_card)

training_state = {
    "epoch":         epoch,
    "global_step":   global_step,
    "optimizer":     optimizer.state_dict(),
    "scheduler":     scheduler.state_dict(),
    "best_val_loss": best_val_loss,
    "train_losses":  train_losses,
}
training_state_path = f"{best_ckpt}/training_state.pt"
torch.save(training_state, training_state_path)
print(f"✅ Training state saved → {training_state_path}")


print(f"Loading best checkpoint from: {best_ckpt}")
best_model    = ParlerTTSForConditionalGeneration.from_pretrained(best_ckpt)
best_tok      = AutoTokenizer.from_pretrained(best_ckpt)
best_desc_tok = AutoTokenizer.from_pretrained(f"{best_ckpt}/desc_tokenizer")

print(f"Pushing to: {OUTPUT_REPO}")
best_model.push_to_hub(OUTPUT_REPO,
    commit_message=f"Nepali finetuned Indic Parler — best val_loss={best_val_loss:.4f}")
best_tok.push_to_hub(OUTPUT_REPO,      commit_message="Prompt tokenizer")
best_desc_tok.push_to_hub(OUTPUT_REPO, commit_message="Description tokenizer")

for path_or_fileobj, path_in_repo, msg in [
    (f"{best_ckpt}/README.md",          "README.md",          "Add model card"),
    (training_state_path,               "training_state.pt",  f"Training state — step={global_step}"),
]:
    api.upload_file(
        path_or_fileobj=path_or_fileobj,
        path_in_repo=path_in_repo,
        repo_id=OUTPUT_REPO,
        commit_message=msg,
    )


print(f"\n✅ Pushed → https://huggingface.co/{OUTPUT_REPO}")

del best_model
gc.collect()
torch.cuda.empty_cache()







