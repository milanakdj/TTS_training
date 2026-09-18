"""Materialise the distilled subset as FLAC parquet shards: clean + augmented.

CPU only -- deliberately. The GPU lever for this corpus is Seed-VC (it is a
single synthetic voice, F0 spread 13 Hz, pairwise MFCC cosine 0.998), and
nothing here pretends otherwise: noise and pitch change channel and F0, not
speaker identity. Speed perturbation is included because resampling moves
formants as well as pitch, which is the closest a CPU augmentation gets to a
different vocal tract.

Every augmented row records its exact parameters so a consumer can filter or
reproduce, and carries `source_id` back to the clean row it came from -- so
clean/augmented pairs never straddle a train/test split.
"""
import os
# One BLAS/OpenMP thread per worker. With 14 processes on 16 cores the default
# per-process thread pools oversubscribe ~2.8x and the whole pool thrashes --
# same failure mode as the CPU bench that looked like "no speedup".
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, glob, io, json, random, hashlib
import numpy as np, soundfile as sf, soxr, librosa
import pyarrow as pa, pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor

SR = 16000
NOISE = sorted(glob.glob('/workspace/proc_data_new/AudioSet/audios/unbal_train/*.wav'))

def _noise(n, rng):
    """n samples of AudioSet noise at SR, tiled if the clip is short."""
    for _ in range(4):
        try:
            p = NOISE[rng.randrange(len(NOISE))]
            d, sr = sf.read(p, dtype='float32')
            if d.ndim > 1: d = d.mean(1)
            if sr != SR: d = soxr.resample(d, sr, SR)
            if len(d) < 100: continue
            if len(d) < n: d = np.tile(d, int(np.ceil(n/len(d))))
            o = rng.randrange(0, max(1, len(d)-n))
            return d[o:o+n]
        except Exception: continue
    return np.zeros(n, dtype='float32')

def augment(x, rng):
    p = {}
    # 1. speed perturbation -- moves pitch AND formants (vocal-tract proxy)
    sp = rng.choice([0.9, 0.95, 1.0, 1.05, 1.1])
    if sp != 1.0:
        x = soxr.resample(x, SR, int(SR/sp)).astype('float32')
    p['speed'] = sp
    # 2. pitch shift, upward-biased as requested
    st = rng.choice([0, 1, 2, 3, -1, -2])
    if st != 0:
        x = librosa.effects.pitch_shift(y=x, sr=SR, n_steps=st, res_type='soxr_hq').astype('float32')
    p['semitones'] = st
    # 3. additive AudioSet noise at a target SNR
    snr = rng.choice([5, 8, 12, 16, 20, 25])
    nz = _noise(len(x), rng)
    ps, pn = float(np.mean(x**2)), float(np.mean(nz**2))
    if pn > 1e-10 and ps > 1e-10:
        x = x + nz * np.sqrt(ps/(pn*10**(snr/10)))
    p['snr_db'] = snr
    # 4. telephone-band simulation on a minority of rows
    if rng.random() < 0.25:
        x = soxr.resample(soxr.resample(x, SR, 8000), 8000, SR).astype('float32')
        p['band'] = 8000
    # 5. gain, then guard against clipping
    g = rng.uniform(0.6, 1.0); x = x * g; p['gain'] = round(g, 3)
    m = float(np.abs(x).max())
    if m > 0.999: x = x / m * 0.999
    return x.astype('float32'), p

def enc(x):
    b = io.BytesIO(); sf.write(b, x, SR, format='FLAC'); return b.getvalue()

def do_shard(job):
    shard_id, recs, n_aug, outdir = job
    rows = []
    for rec in recs:
        path = f"/workspace/oov_distill/stage/{rec['shard']}/t.parquet"
        try:
            df = pq.ParquetFile(path).read_row_group(rec['rg']).to_pandas()
            a = df.iloc[rec['idx']]['audio']
            x, sr = sf.read(io.BytesIO(a['bytes']), dtype='float32')
        except Exception:
            continue
        if x.ndim > 1: x = x.mean(1)
        if sr != SR: x = soxr.resample(x, sr, SR).astype('float32')
        base = dict(id=rec['id'], text=rec['text'], source_id=rec['id'],
                    duration=len(x)/SR, oov_hits=json.dumps(rec['hits'], ensure_ascii=False),
                    n_oov=len(rec['hits']),
                    # tier flags: oov_core marks the 8.8 h subset carrying every
                    # learnable term at >=5 occurrences. Coverage-by-presence is
                    # misleading here -- 21% of "covered" terms occur exactly once.
                    oov_core=bool(rec.get('oov_core', False)),
                    n_oov_learnable=int(rec.get('n_oov_learnable', 0)),
                    variant='clean', aug_params='{}')
        rows.append({**base, 'audio': {'bytes': enc(x), 'path': rec['id']+'.flac'}})
        rng = random.Random(int(hashlib.md5(rec['id'].encode()).hexdigest()[:8], 16))
        for k in range(n_aug):
            y, p = augment(x.copy(), rng)
            rows.append({**base, 'id': f"{rec['id']}_aug{k}", 'variant': f'aug{k}',
                         'duration': len(y)/SR, 'aug_params': json.dumps(p),
                         'audio': {'bytes': enc(y), 'path': f"{rec['id']}_aug{k}.flac"}})
    if not rows: return shard_id, 0, 0.0
    os.makedirs(f"{outdir}/shard_{shard_id:04d}", exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows),
                   f"{outdir}/shard_{shard_id:04d}/train-00000-of-00001.parquet",
                   compression='zstd')
    return shard_id, len(rows), sum(r['duration'] for r in rows)/3600

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sel', default='/workspace/oov_distill/out/selected.jsonl')
    ap.add_argument('--outdir', default='/workspace/oov_distill/out/hf')
    ap.add_argument('--n-aug', type=int, default=1)
    ap.add_argument('--per-shard', type=int, default=250)
    ap.add_argument('--workers', type=int, default=14)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.sel, encoding='utf-8')]
    print(f"{len(recs)} selected rows, {sum(r['dur'] for r in recs)/3600:.1f} h clean, "
          f"x{1+a.n_aug} variants", flush=True)
    jobs = [(i, recs[o:o+a.per_shard], a.n_aug, a.outdir)
            for i, o in enumerate(range(0, len(recs), a.per_shard))]
    tot_r = tot_h = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for sid, n, h in ex.map(do_shard, jobs):
            tot_r += n; tot_h += h
            print(f"  shard {sid:04d}: {n} rows {h:.2f} h  (running {tot_r} rows, {tot_h:.1f} h)", flush=True)
    print(f"DONE {tot_r} rows, {tot_h:.1f} h -> {a.outdir}", flush=True)
