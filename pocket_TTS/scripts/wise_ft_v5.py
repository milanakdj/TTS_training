"""Weight-space interpolation (WiSE-FT) between a v5 probe and untouched Kyutai 24L.

    w = A * finetuned + (1 - A) * kyutai

Cheapest test of how much of Kyutai's English is recoverable without training.
Applied to both the raw weights and the EMA shadow (gen_2x2_guided.py loads EMA).
The grafted text embedding is handled row-wise: rows 0..3999 and the padding row
exist in Kyutai and are interpolated; the Nepali rows (4000..n_bins-1) have no
Kyutai counterpart and keep their finetuned values. Tensors Kyutai lacks (the LSD
w_s_t net) are kept as finetuned. Each A becomes its own run dir, loadable by
load_run / gen_2x2_guided.py. Run with repo/.venv from pocket_TTS/:

    SRC=/workspace/v5l_probe ALPHAS=0.25,0.5,0.75 repo/.venv/bin/python3 scripts/wise_ft_v5.py
"""
import os, shutil, sys
from pathlib import Path

import safetensors.torch
import torch

sys.path.insert(0, "/root/tts/TTS_training/pocket_TTS/repo")
from pocket_tts.utils.utils import download_if_necessary
from training.checkpointing import latest_checkpoint
from training.modules.builders import load_model_config

SRC = Path(os.environ.get("SRC", "/workspace/v5l_probe"))
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0.25,0.5,0.75").split(",")]
BASE_CFG = "/root/tts/TTS_training/pocket_TTS/repo/pocket_tts/config/english_2026-04_24l.yaml"
EMBED = "flow_lm.conditioner.embed.weight"

cfg = load_model_config(BASE_CFG, {})
base = safetensors.torch.load_file(download_if_necessary(str(cfg.weights_path)))
base = {k: v.float() for k, v in base.items() if k.startswith("flow_lm.")}
ckpt_path = latest_checkpoint(SRC)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
print(f"finetuned: {ckpt_path} (step {ckpt['step']}) | kyutai: {cfg.weights_path}")


def blend(state: dict, a: float) -> tuple[dict, int, int]:
    out, mixed, kept = {}, 0, 0
    for k, v in state.items():
        b = base.get(k)
        if b is None or not v.is_floating_point():
            out[k], kept = v, kept + 1
            continue
        if k == EMBED:
            inherited = b.shape[0] - 1                 # last row is padding
            w = v.clone()
            w[:inherited] = a * v[:inherited].float() + (1 - a) * b[:inherited]
            w[-1] = a * v[-1].float() + (1 - a) * b[-1]
            out[k] = w.to(v.dtype)
        else:
            assert b.shape == v.shape, (k, b.shape, v.shape)
            out[k] = (a * v.float() + (1 - a) * b).to(v.dtype)
        mixed += 1
    return out, mixed, kept


for a in ALPHAS:
    dst = Path(f"{SRC}_wise{int(round(a * 100)):02d}")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(SRC / "args.yaml", dst / "args.yaml")
    model, m_mixed, m_kept = blend(ckpt["model"], a)
    ema, e_mixed, e_kept = blend(ckpt["ema"], a) if ckpt.get("ema") else (None, 0, 0)
    torch.save({"step": ckpt["step"], "model": model, "ema": ema},
               dst / f"checkpoint_{ckpt['step']:08d}.pt")
    print(f"A={a}: model {m_mixed} blended / {m_kept} kept, ema {e_mixed} / {e_kept} -> {dst}")
print("WISE DONE")
