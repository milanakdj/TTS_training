"""Package the Seed-VC output as extra shards on the existing dataset.

Appends shard_0105+ so the original 105 shards are untouched. Each converted
clip ships twice: `vc` (converted, clean) and `vc_aug` (converted, then the same
CPU augmentation chain). source_id still points at the original clean row, so
every variant of one utterance stays on the same side of a split.
"""
import os, json, io, glob, hashlib, random
import numpy as np, soundfile as sf, soxr, pyarrow as pa, pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.insert(0, '/workspace/oov_distill')
from augment import augment, enc, SR

VC   = os.environ.get('VC_DIR', '/workspace/oov_distill/out/vc')
OUT  = '/workspace/oov_distill/out/hf'
START_SHARD = 105

def build(job):
    sid, recs, with_aug = job
    rows = []
    for m in recs:
        p = f"{VC}/{m['id']}_vc.wav"
        if not os.path.exists(p): continue
        x, sr = sf.read(p, dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        if sr != SR: x = soxr.resample(x, sr, SR).astype('float32')
        base = dict(id=f"{m['id']}_vc", text=m['text'], source_id=m['id'],
                    duration=len(x)/SR,
                    oov_hits=json.dumps([], ensure_ascii=False),
                    n_oov=m.get('n_oov') or 0, oov_core=True,
                    n_oov_learnable=m.get('n_oov_learnable') or 0,
                    variant='vc',
                    aug_params=json.dumps({'vc_ref': m['ref'],
                                           'cos_out_ref': m['cos_out_ref'],
                                           'vc_gate_passed': m['passed']}))
        rows.append({**base, 'audio': {'bytes': enc(x), 'path': f"{m['id']}_vc.flac"}})
        if not with_aug: continue
        rng = random.Random(int(hashlib.md5((m['id']+'vc').encode()).hexdigest()[:8], 16))
        y, prm = augment(x.copy(), rng)
        prm.update(vc_ref=m['ref'], cos_out_ref=m['cos_out_ref'], vc_gate_passed=m['passed'])
        rows.append({**base, 'id': f"{m['id']}_vc_aug0", 'variant': 'vc_aug0',
                     'duration': len(y)/SR, 'aug_params': json.dumps(prm),
                     'audio': {'bytes': enc(y), 'path': f"{m['id']}_vc_aug0.flac"}})
    if not rows: return sid, 0, 0.0
    d = f"{OUT}/shard_{sid:04d}"; os.makedirs(d, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), f"{d}/train-00000-of-00001.parquet", compression='zstd')
    return sid, len(rows), sum(r['duration'] for r in rows)/3600

if __name__ == '__main__':
    man = [json.loads(l) for l in open(f'{VC}/vc_manifest.jsonl', encoding='utf-8')]
    # keep only clips that cleared the speaker gate: below it the timbre did not
    # move enough to be worth shipping as a distinct speaker
    kept = [m for m in man if m['passed']]
    print(f"converted {len(man)} | clearing 0.85 gate {len(kept)} ({len(kept)/len(man):.0%})", flush=True)
    per = 200
    with_aug = os.environ.get('VC_WITH_AUG', '0') == '1'
    print(f"shipping variants: {'vc + vc_aug0' if with_aug else 'vc only'}", flush=True)
    start = int(os.environ.get('VC_START_SHARD', START_SHARD))
    jobs = [(start + i, kept[o:o+per], with_aug) for i, o in enumerate(range(0, len(kept), per))]
    tr = th = 0
    with ProcessPoolExecutor(8) as ex:
        for sid, n, h in ex.map(build, jobs):
            tr += n; th += h
            print(f"  shard {sid:04d}: {n} rows {h:.2f} h", flush=True)
    print(f"DONE +{tr} rows, +{th:.2f} h appended to {OUT}")
