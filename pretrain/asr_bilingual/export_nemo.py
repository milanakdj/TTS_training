#!/usr/bin/env python
"""Export a Lightning .ckpt to a self-contained .nemo, on CPU.

always_save_nemo writes nemotron_ne_en.nemo for the *latest* checkpoint, not the
best one, so the shipped artifact has to be rebuilt from the best-val_wer .ckpt.
Runs on CPU so it does not contend with the live training run for GPU memory.
"""
import sys, torch
import nemo.collections.asr as nemo_asr

BASE = ("/workspace/hf_cache/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/"
        "snapshots/ea30d66debe3740a08b573244286791d423d6b3e/"
        "nemotron-3.5-asr-streaming-0.6b.nemo")

ckpt, out = sys.argv[1], sys.argv[2]
m = nemo_asr.models.ASRModel.restore_from(BASE, map_location="cpu")
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
sd = {k: v for k, v in sd.items() if not k.startswith(("loss.", "wer."))}
miss, unexp = m.load_state_dict(sd, strict=False)
miss = [k for k in miss if not k.startswith(("loss.", "wer."))]
print(f"[export] {len(sd)} tensors, {len(miss)} missing, {len(unexp)} unexpected")
if len(miss) > 0.02 * len(sd):
    sys.exit("too many missing keys -- architecture mismatch, refusing to export")
m.save_to(out)
print(f"[export] wrote {out}")
