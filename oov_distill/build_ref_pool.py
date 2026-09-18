"""Build a speaker-balanced Seed-VC reference pool from indicvoices-r-long (Nepali).

Written against the failure recorded for reference_pool_v2: that pool was 600
clips but 74% one speaker, because it sampled uniformly over *rows* -- which
weights by dataset size, not by speaker count -- and it validated with pitch std
(42.7 Hz), which cannot tell one expressive speaker from forty.

So this samples round-robin over speakers with a hard per-speaker cap, dedupes
by content hash, and validates by embedding spread rather than pitch.

    python build_ref_pool.py --per-speaker 2 --max-speakers 400
"""
import argparse, glob, hashlib, json, os, random, collections
import numpy as np, soundfile as sf, soxr

NE = {'ne', 'nepali', 'ne-np', 'ne_np'}
SRC = '/workspace/proc_data_new/indicvoices-r-long'
REMAP = ('/projects/data/ttsteam/proc_data_new', '/workspace/proc_data_new')

def load_rows(dmin, dmax):
    by_spk = collections.defaultdict(list)
    for f in glob.glob(f'{SRC}/**/*.jsonl', recursive=True):
        if 'rejection' in f or 'backup' in f or '.trash' in f: continue
        for l in open(f, errors='ignore'):
            try: r = json.loads(l)
            except Exception: continue
            if str(r.get('language', '')).strip().lower() not in NE: continue
            d = float(r.get('duration') or 0)
            if not (dmin <= d <= dmax): continue
            s = r.get('speaker_id')
            if not s: continue
            by_spk[s].append(r)
    return by_spk

def pick(by_spk, per_speaker, max_speakers, seed=0):
    """Round-robin across speakers so no speaker can dominate by row count."""
    rng = random.Random(seed)
    spks = sorted(by_spk)
    rng.shuffle(spks)
    spks = spks[:max_speakers]
    out = []
    for rnd in range(per_speaker):
        for s in spks:
            rows = by_spk[s]
            if rnd < len(rows):
                out.append((s, rows[rnd] if len(rows) <= per_speaker else rng.choice(rows)))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/workspace/oov_distill/out/ref_pool')
    ap.add_argument('--per-speaker', type=int, default=2)
    ap.add_argument('--max-speakers', type=int, default=400)
    ap.add_argument('--dmin', type=float, default=4.0)
    ap.add_argument('--dmax', type=float, default=15.0)
    ap.add_argument('--sr', type=int, default=22050)
    a = ap.parse_args()

    by_spk = load_rows(a.dmin, a.dmax)
    print(f"speakers with a usable clip: {len(by_spk)}", flush=True)
    sel = pick(by_spk, a.per_speaker, a.max_speakers)
    os.makedirs(a.out, exist_ok=True)
    seen_hash, man, per = set(), [], collections.Counter()
    for s, r in sel:
        p = r['audio_filepath'].replace(*REMAP)
        if not os.path.exists(p): continue
        try: x, sr = sf.read(p, dtype='float32')
        except Exception: continue
        if x.ndim > 1: x = x.mean(1)
        h = hashlib.md5(x.tobytes()).hexdigest()
        if h in seen_hash: continue            # 353 dupes slipped into the last pool
        seen_hash.add(h)
        if sr != a.sr: x = soxr.resample(x, sr, a.sr).astype('float32')
        m = float(np.abs(x).max())
        if m < 1e-4: continue
        x = x / m * 0.95
        fn = f"{s}_{per[s]}.wav"
        sf.write(os.path.join(a.out, fn), x, a.sr)
        per[s] += 1
        man.append(dict(file=fn, speaker_id=s, gender=r.get('gender'),
                        age_group=r.get('age_group'), duration=len(x)/a.sr))
    with open(os.path.join(a.out, 'manifest.jsonl'), 'w', encoding='utf-8') as fh:
        for m_ in man: fh.write(json.dumps(m_, ensure_ascii=False) + '\n')
    big = per.most_common(1)[0][1] / max(len(man), 1) if man else 0
    print(f"wrote {len(man)} clips / {len(per)} speakers -> {a.out}")
    print(f"biggest-speaker share: {big:.2%}  (v2 pool failed at 74%)")
    g = collections.Counter(m_['gender'] for m_ in man)
    print(f"gender: {dict(g)}")

    # validate by embedding spread, not pitch std
    import librosa, warnings; warnings.filterwarnings('ignore')
    from sklearn.metrics.pairwise import cosine_similarity
    rng = random.Random(0); samp = rng.sample(man, min(60, len(man)))
    E, f0s = [], []
    for m_ in samp:
        x, sr = sf.read(os.path.join(a.out, m_['file']), dtype='float32')
        f0 = librosa.yin(x, fmin=60, fmax=400, sr=sr); f0 = f0[(f0 > 60) & (f0 < 400)]
        if not len(f0): continue
        E.append(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20).mean(axis=1)); f0s.append(np.median(f0))
    E = np.array(E); f0s = np.array(f0s)
    C = cosine_similarity(E); iu = np.triu_indices(len(C), 1)
    print(f"\nvalidation over {len(E)} clips:")
    print(f"  F0 p5-p95: {np.percentile(f0s,5):.0f}-{np.percentile(f0s,95):.0f} Hz (sd {f0s.std():.1f})")
    print(f"  MFCC cosine: mean {C[iu].mean():.3f}  min {C[iu].min():.3f}")
    print(f"  (Premal single voice for contrast: 211-224 Hz, sd 4.0, cos min 0.965)")
