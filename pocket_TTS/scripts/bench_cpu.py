"""Measure what the post actually claims, for a Nepali Pocket TTS checkpoint.

  TTFB -- wall time from the generate call to the FIRST audio chunk being yielded.
          This is the "~30 ms to first byte" number. It is a STREAMING latency, so
          it must be read off generate_audio_stream(), never off total synthesis.
  RTFx -- audio seconds produced per wall second. >1 is faster than realtime.

Thread count is part of the result: CPU TTS figures are meaningless without it
(the post's Mac number used 2 cores). --quantize enables the INT8 path.
"""
import argparse, os, statistics as st, time

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="pocket_tts model yaml")
ap.add_argument("--checkpoint", default=None, help="training checkpoint_*.pt (EMA weights)")
ap.add_argument("--voice", default=None, help="wav to clone; omitted = model default")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--quantize", action="store_true", help="INT8 CPU path")
ap.add_argument("--text", default="आजको बैठक बिहान दस बजे सभाकक्षमा सुरु हुनेछ। सबै जना भोलि यहाँ भेला हुनेछन्।")
args = ap.parse_args()

os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
import torch
torch.set_num_threads(args.threads)

from pocket_tts import TTSModel  # noqa: E402

print(f"threads={args.threads}  quantize={args.quantize}  device=cpu")
t0 = time.perf_counter()
model = TTSModel.load_model(config=args.config, quantize=args.quantize)
if args.checkpoint:
    model.load_training_checkpoint(args.checkpoint, use_ema=True)
model = model.to("cpu").eval()
print(f"load: {time.perf_counter() - t0:.2f}s   sample_rate={model.sample_rate}")

state = model.get_state_for_audio_prompt(args.voice) if args.voice else model.get_state_for_audio_prompt(None)

ttfbs, rtfs = [], []
for i in range(args.runs):
    t0 = time.perf_counter()
    first, n = None, 0
    with torch.inference_mode():
        for chunk in model.generate_audio_stream(state, args.text):
            if first is None:
                first = time.perf_counter() - t0
            n += chunk.numel()
    wall = time.perf_counter() - t0
    audio_s = n / model.sample_rate
    ttfbs.append(first * 1000); rtfs.append(audio_s / wall)
    print(f"  run {i}: ttfb={first*1000:7.1f} ms  audio={audio_s:5.2f}s  wall={wall:5.2f}s  rtfx={audio_s/wall:5.2f}x")

# run 0 pays for lazy init; report both so the warm number isn't quietly cherry-picked
print(f"\nTTFB  median {st.median(ttfbs):.1f} ms  (warm median {st.median(ttfbs[1:]) if len(ttfbs) > 1 else ttfbs[0]:.1f} ms)")
print(f"RTFx  median {st.median(rtfs):.2f}x realtime")
