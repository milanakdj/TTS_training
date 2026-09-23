"""Synthesize the OOV probe set from a 24-LAYER TEACHER checkpoint, with CFG.

Why this exists instead of gen_oov_probe.py: the pocket_tts inference path
(pocket_tts/models/tts_model.py) has no cfg_coef -- grep it, there is none. The
teacher is only ever sampled with guidance (args.yaml: sample_cfg_coef 2.0), and
depth distillation exists precisely to bake that guidance into a student the
inference path CAN express (configs/nepali_distill_v3.yaml).

Running a teacher through the student harness therefore measures nothing: the
first attempt produced 0.32 s for control_ne, a plain Nepali sentence that v2
renders at 4.72 s, and would have been read as catastrophic content loss. It was
an ungated-teacher artifact, not a model result.

Defaults mirror the training loop's own sampler (temp 0.3, cfg 2.0) so these
clips are comparable to run_dir/samples/. Output layout matches gen_oov_probe.py,
so score_oov_probe.py --wav-dir scores it unchanged.

    cd repo && .venv/bin/python3 ../infer/final/gen_oov_probe_teacher.py \
        --run-dir /workspace/nepali_teacher_24l_v3 --checkpoint <pinned>.pt
"""
import argparse, json, os, sys, time
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.eval.librispeech import load_run, load_mono, latents_to_wav  # noqa: E402

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", default="/workspace/nepali_teacher_24l_v3")
ap.add_argument("--checkpoint", default=None, help="pin one; the run keeps only the last 3")
ap.add_argument("--prompt", default="/workspace/proc_data_new/ai4bharat___rasa/train/audio/Nepali/NEP_F_CONV_00347.wav")
ap.add_argument("--out", required=True)
ap.add_argument("--temp", type=float, default=0.3)
ap.add_argument("--cfg", type=float, default=2.0)
ap.add_argument("--n-steps", type=int, default=1)
ap.add_argument("--eos-threshold", type=float, default=-1.0)
ap.add_argument("--max-sec", type=float, default=30.0)
ap.add_argument("--raw-weights", action="store_true", help="skip EMA (what samples/ uses)")
ap.add_argument("--device", default="cpu", help="cpu: the GPU is held by training")
ap.add_argument("--threads", type=int, default=int(os.environ.get("GEN_THREADS", "4")))
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.set_num_threads(args.threads)
os.makedirs(args.out, exist_ok=True)
device = torch.device(args.device)

model, mimi, step = load_run(args.run_dir, device, use_ema=not args.raw_weights,
                             checkpoint=args.checkpoint)
print(f"loaded step {step} from {args.checkpoint or args.run_dir} "
      f"(ema={'off' if args.raw_weights else 'on'}, cfg={args.cfg}, temp={args.temp})")
torch.manual_seed(args.seed)

tokenize = model.flow_lm.conditioner.tokenizer.sp.encode
max_frames = int(args.max_sec * mimi.frame_rate)

with torch.no_grad():
    voice = mimi.encode_to_latent(load_mono(args.prompt, mimi.sample_rate)[None, None].to(device))[0]

probes = json.load(open(f"{F}/oov_probes.json"))["probes"]
# Stale audio from an earlier checkpoint silently entering results is a known
# way to get a wrong answer here -- purge rather than skip.
for p in probes:
    f = f"{args.out}/{p['id']}.wav"
    if os.path.exists(f):
        os.remove(f)

for p in probes:
    t0 = time.monotonic()
    toks = torch.tensor(tokenize(p["text"]), dtype=torch.long)
    with torch.no_grad():
        latents = model.generate([toks], [voice], max_frames=max_frames, temp=args.temp,
                                 n_steps=args.n_steps, cfg_coef=args.cfg,
                                 eos_threshold=args.eos_threshold)[0]
        wav = latents_to_wav(mimi, latents, device)
    if wav is None:
        print(f"{p['id']:14s}  EMPTY (<8 frames)  {p['text'][:50]}")
        continue
    x = wav.float().cpu().numpy()
    sf.write(f"{args.out}/{p['id']}.wav", x, mimi.sample_rate)
    print(f"{p['id']:14s} {len(x)/mimi.sample_rate:5.2f}s  ({time.monotonic()-t0:.1f}s)  {p['text'][:50]}")
print(f"\nwrote clips to {args.out} (step {step}, cfg {args.cfg})")
