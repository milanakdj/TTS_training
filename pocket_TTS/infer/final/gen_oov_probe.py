"""Synthesize the OOV probe set. Run with repo/.venv (it imports pocket_tts).

Raw text by default -- the probes exist to test what the model does with input the
frontend has NOT cleaned. --frontend scores the workaround's ceiling instead.

    cd ../../repo && .venv/bin/python3 ../infer/final/gen_oov_probe.py [--frontend]
"""
import argparse, json, os, sys, time
import scipy.io.wavfile as wav
import torch
from pocket_tts.models.tts_model import TTSModel

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
ap = argparse.ArgumentParser()
ap.add_argument("--config", default=f"{R}/infer/nepali_student_6l.yaml")
ap.add_argument("--prompt", default="/workspace/proc_data_new/ai4bharat___rasa/train/audio/Nepali/NEP_F_CONV_00347.wav")
ap.add_argument("--out")
ap.add_argument("--frontend", action="store_true")
ap.add_argument("--threads", type=int, default=int(os.environ.get("GEN_THREADS", "4")))
# The v3 models need an explicit eos_threshold: out_eos is frozen from the
# teacher while the backbone regressed onto cfg-2.0-combined activations, so the
# head is miscalibrated against the z it now sees and fires within a few frames
# at the shipped default of -4.0. Calibrated on calib_v3.json, not on the probes.
ap.add_argument("--eos-threshold", type=float, default=None)
ap.add_argument("--seed", type=int, default=None, help="probe durations are seed-noisy; vary it")
args = ap.parse_args()

args.out = args.out or f"{F}/wav/oov_probe_{'frontend' if args.frontend else 'raw'}"
torch.set_num_threads(args.threads)
os.makedirs(args.out, exist_ok=True)

prep = (lambda t: t)
if args.frontend:
    sys.path.insert(0, f"{R}/frontend")
    from ne_frontend import normalize, assert_clean
    def prep(t):
        n = normalize(t); assert_clean(n); return n

kw = {} if args.eos_threshold is None else {"eos_threshold": args.eos_threshold}
model = TTSModel.load_model(config=args.config, **kw)
model.to("cpu")
state = model.get_state_for_audio_prompt(args.prompt)

probes = json.load(open(f"{F}/oov_probes.json"))["probes"]
# Stale audio from an earlier checkpoint silently entering results is a known
# way to get a wrong answer here -- purge rather than skip.
for p in probes:
    f = f"{args.out}/{p['id']}.wav"
    if os.path.exists(f):
        os.remove(f)

for p in probes:
    text = prep(p["text"])
    if args.seed is not None:
        torch.manual_seed(args.seed)
    t0 = time.monotonic()
    a = model.generate_audio(state, text, copy_state=True)
    x = a.detach().cpu().numpy().squeeze()
    wav.write(f"{args.out}/{p['id']}.wav", model.config.mimi.sample_rate, x)
    print(f"{p['id']:14s} {len(x)/model.config.mimi.sample_rate:5.2f}s  "
          f"({time.monotonic()-t0:.1f}s)  {text[:50]}")
print(f"\nwrote {len(probes)} clips to {args.out} (frontend={'on' if args.frontend else 'OFF'}, "
      f"eos_threshold={args.eos_threshold if args.eos_threshold is not None else 'default -4.0'}, "
      f"seed={args.seed})")
