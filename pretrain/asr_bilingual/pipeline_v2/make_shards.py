"""Step 4 of PRETRAINING_FROM_SCRATCH.md: make the data fast.

773k small files on a network filesystem is seek-bound. It barely matters for a
2-day finetune; over 50-150k hours seen it dominates, and the doc calls this
"the real engineering".

Two things happen here, in one pass:
  1. resample to 16 kHz mono once, offline -- the source is 24 kHz and NeMo
     otherwise resamples on CPU every batch of every epoch, forever;
  2. pack into tarred shards -- sequential reads instead of 773k seeks.

NeMo ships the packer, so this drives that rather than reimplementing it.
"""
import argparse, os, subprocess, sys, glob, json

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="/workspace/asr_pretrain_v2/manifests/train_mix.jsonl")
ap.add_argument("--out", default="/workspace/asr_pretrain_v2/tarred")
ap.add_argument("--shards", type=int, default=2048)
ap.add_argument("--max-duration", type=float, default=60.0)
ap.add_argument("--min-duration", type=float, default=0.3)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()

SCRIPT = None
for c in ("/workspace/venvs/nemo/lib/python3.10/site-packages/scripts/"
          "speech_recognition/convert_to_tarred_audio_dataset.py",
          "/workspace/venvs/nemo/share/nemo/scripts/speech_recognition/"
          "convert_to_tarred_audio_dataset.py"):
    if os.path.exists(c): SCRIPT = c; break
if SCRIPT is None:
    hits = glob.glob("/workspace/venvs/nemo/**/convert_to_tarred_audio_dataset.py",
                     recursive=True)
    SCRIPT = hits[0] if hits else None
if SCRIPT is None:
    # The packer ships in the NeMo *repo*, not always in the wheel. Rather than
    # silently training off the slow path, say so and let the caller decide.
    print("convert_to_tarred_audio_dataset.py not found in the venv.\n"
          "Either fetch it from the NeMo repo, or run the pilot untarred and\n"
          "accept the dataloader ceiling that bench_dl.py measured.")
    sys.exit(2)

os.makedirs(a.out, exist_ok=True)
cmd = [sys.executable, SCRIPT,
       f"--manifest_path={a.manifest}", f"--target_dir={a.out}",
       f"--num_shards={a.shards}", f"--max_duration={a.max_duration}",
       f"--min_duration={a.min_duration}", "--shuffle", "--shuffle_seed=42",
       f"--workers={a.workers}"]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
