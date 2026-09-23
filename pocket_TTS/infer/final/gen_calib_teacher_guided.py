"""The v3 teacher on the calibration set through the GUIDED training path.

The inference package has no cfg_coef, so TTSModel can only ever run the teacher
unguided -- and this teacher was trained to be sampled at sample_cfg_coef 2.0.
Scoring it unguided measures the deployable path, but it does not tell us whether
the teacher itself is sound, which is what decides whether a re-distill could work.

Output layout matches gen_calib_wavs.py so score_calib_wavs.py reads it unchanged.

    cd repo && .venv/bin/python3 ../infer/final/gen_calib_teacher_guided.py
"""
import json, os, sys, time
import soundfile as sf
import torch

sys.path.insert(0, "/root/tts/TTS_training/pocket_TTS")
from training.eval.librispeech import load_run, load_mono, latents_to_wav

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
RUN = os.environ.get("RUN", "/workspace/nepali_teacher_24l_v3")
CKPT = os.environ.get("CKPT", f"{RUN}/checkpoint_00200000.pt")
NAME = os.environ.get("NAME", "teacher_24l_v3_guided")
CFGS = [float(x) for x in os.environ.get("CFGS", "1.0,2.0").split(",")]
TEMP = float(os.environ.get("TEMP", "0.3"))
DEV = torch.device(os.environ.get("CALIB_DEVICE", "cpu"))
torch.set_num_threads(int(os.environ.get("GEN_THREADS", "8")))

items = json.load(open(f"{F}/calib_v3.json"))
model, mimi, step = load_run(RUN, DEV, use_ema=True, checkpoint=CKPT)
print(f"loaded step {step} from {CKPT} (ema on, temp {TEMP})", flush=True)
tokenize = model.flow_lm.conditioner.tokenizer.sp.encode

# One voice-prompt encode per distinct prompt file, reused across cfg values.
prompts = {}
with torch.no_grad():
    for it in items:
        if it["prompt"] not in prompts:
            prompts[it["prompt"]] = mimi.encode_to_latent(
                load_mono(it["prompt"], mimi.sample_rate)[None, None].to(DEV))[0]

for cfg in CFGS:
    out = f"{F}/wav/calib_{NAME}_th{cfg:g}"   # "th" slot carries cfg here
    os.makedirs(out, exist_ok=True)
    for it in items:                           # purge; stale audio poisons results
        f = f"{out}/{it['id']}.wav"
        if os.path.exists(f):
            os.remove(f)
    t0, empty = time.time(), 0
    for it in items:
        toks = torch.tensor(tokenize(it["text"]), dtype=torch.long)
        torch.manual_seed(1234)
        with torch.no_grad():
            lat = model.generate([toks], [prompts[it["prompt"]]], max_frames=375,
                                 temp=TEMP, n_steps=1, cfg_coef=cfg)[0]
            w = latents_to_wav(mimi, lat, DEV)
        if w is None:
            empty += 1
            continue
        sf.write(f"{out}/{it['id']}.wav", w.float().cpu().numpy(), mimi.sample_rate)
    print(f"  cfg {cfg:>4}: {len(items)-empty}/{len(items)} clips -> {out} "
          f"({empty} empty) [{time.time()-t0:.0f}s]", flush=True)
print("GUIDED GEN DONE")
