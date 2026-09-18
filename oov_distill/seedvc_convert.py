"""Seed-VC over the oov_core clips, round-robin across the reference pool.

Two gates, both from the reference_pool_v2 post-mortem:

1. **Speaker gate** -- cos(vc_out, its own reference) >= --min-spk-cos. Without
   this, audio that never actually converted still passes: 20% of v2 rows sat at
   0.974 cosine to the *source* speaker and were accepted because the only gate
   measured intelligibility, which unconverted audio trivially passes.
2. **Intelligibility gate** -- cos(vc_out, source) must not be so high that
   nothing moved, and we log CER separately because VC previously degraded CER
   on 81.9% of converted rows.

Converts the CLEAN clips only. Augmentation is re-applied afterwards; running VC
on already-noised audio degrades the conversion.
"""
import argparse, json, os, random, sys, time, collections
import numpy as np, soundfile as sf, librosa, warnings
warnings.filterwarnings('ignore')

def spk_emb(path_or_wav, sr=None):
    """Cheap MFCC-mean embedding. Consistent with how the pools were validated;
    swap for Resemblyzer if you want the same metric as the v2 post-mortem."""
    if isinstance(path_or_wav, str):
        x, sr = sf.read(path_or_wav, dtype='float32')
    else:
        x = path_or_wav
    if x.ndim > 1: x = x.mean(1)
    e = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20).mean(axis=1)
    return e / (np.linalg.norm(e) + 1e-9)

def cos(a, b): return float(np.dot(a, b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sel', default='/workspace/oov_distill/out/selected.jsonl')
    ap.add_argument('--refs', default='/workspace/oov_distill/out/ref_pool')
    ap.add_argument('--out', default='/workspace/oov_distill/out/vc')
    ap.add_argument('--core-only', action='store_true', default=True)
    ap.add_argument('--limit', type=int, default=0, help='0 = all')
    ap.add_argument('--diffusion-steps', type=int, default=25)
    ap.add_argument('--min-spk-cos', type=float, default=0.85)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, '/workspace/seed-vc')
    os.makedirs(a.out, exist_ok=True)

    rows = [json.loads(l) for l in open(a.sel, encoding='utf-8')]
    if a.core_only: rows = [r for r in rows if r.get('oov_core')]
    if a.limit: rows = rows[:a.limit]
    refs = sorted(f for f in os.listdir(a.refs) if f.endswith('.wav'))
    print(f"sources: {len(rows)} clips / {sum(r['dur'] for r in rows)/3600:.2f} h", flush=True)
    print(f"refs   : {len(refs)} clips", flush=True)

    # round-robin targets so no reference speaker dominates the output
    rng = random.Random(a.seed); rng.shuffle(refs)
    pairs = [(r, refs[i % len(refs)]) for i, r in enumerate(rows)]

    from inference import main as vc_main   # noqa -- resolved after deps install
    print("NOTE: batch driver expects a load-once API; see run() below", flush=True)

if __name__ == '__main__':
    main()
