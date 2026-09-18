"""Rebuild a .nemo from a mid-run .ckpt so eval_probe.py can score it.

exp_manager writes a .nemo only for the last state, but the interesting question
is whether the model was BETTER earlier -- run_v5 kept a step-5k checkpoint at
val_wer 0.9978 and never beat it in the next 170,000 steps. The config comes from
the run's own final .nemo (hparams.yaml carries python object tags and will not
load standalone); only the weights are swapped.
"""
import sys, torch
import nemo.collections.asr as nemo_asr

ckpt, base_nemo, out = sys.argv[1], sys.argv[2], sys.argv[3]
m = nemo_asr.models.EncDecCTCModelBPE.restore_from(base_nemo, map_location="cpu")
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
missing, unexpected = m.load_state_dict(sd, strict=False)
print("missing", len(missing), "unexpected", len(unexpected))
m.save_to(out)
print("wrote", out)
