"""Distil Premal-12/c9nepali-audio-dataset2 down to a target-hours OOV-dense subset.

Selection is greedy set-cover over the 695 *lexical* OOV terms (the 3,498 digit
strings are a verbalizer problem, not a data problem -- 1,052 distinct phone
numbers teach nothing that the ten digits do not), then a diversity fill.

Text is normalised for whitespace ONLY. The TTS voice reads the article
furniture aloud -- section names and clock timestamps are audible in the audio
-- so stripping headlines from the text would *create* the misalignment it
looks like it is fixing. Verified on shard_0000 row 3.
"""
import argparse, io, json, os, re, sys, unicodedata, hashlib
import numpy as np, pandas as pd, pyarrow.parquet as pq, soundfile as sf

DEV = re.compile(r'[ऀ-ॿ]'); LAT = re.compile(r'[A-Za-z]')
NUM = re.compile(r'(?:[०-९]|[.])+|[0-9.]+')

def load_oov(p):
    toks = [l.strip() for l in open(p, encoding='utf-8') if l.strip()]
    lex = [t for t in toks if not re.fullmatch(NUM, t)]
    return toks, lex

def clean(s):
    s = unicodedata.normalize('NFC', str(s)).replace('\xa0', ' ')
    s = re.sub(r'[ \t]*\n[ \t]*', ' ', s)       # furniture is spoken: keep it
    return re.sub(r'\s+', ' ', s).strip()

def devfrac(s):
    a, b = len(DEV.findall(s)), len(LAT.findall(s))
    return a / max(a + b, 1)

def scan_shard(path, lex_set):
    """-> list of row dicts (no audio bytes), cheap enough to hold 100k of."""
    out = []
    f = pq.ParquetFile(path)
    for rg in range(f.metadata.num_row_groups):
        df = f.read_row_group(rg).to_pandas()
        for i in range(len(df)):
            r = df.iloc[i]
            a = r['audio']
            if a is None: continue
            try: info = sf.info(io.BytesIO(a['bytes']))
            except Exception: continue
            t = clean(r['text'])
            if not t: continue
            hits = frozenset(o for o in lex_set if o in t)
            out.append(dict(id=str(r['id']), shard=os.path.basename(os.path.dirname(path)),
                            rg=rg, idx=i, dur=float(info.duration), sr=info.samplerate,
                            devfrac=devfrac(t), nchar=len(t), text=t, hits=hits))
    return out

def select(rows, lex, target_h, min_dev, dmin, dmax):
    ok = [r for r in rows if r['devfrac'] >= min_dev and dmin <= r['dur'] <= dmax]
    dropped_en = sum(1 for r in rows if r['devfrac'] < min_dev)
    print(f"  candidates {len(ok)}/{len(rows)}  (dropped {dropped_en} english-ish, "
          f"{len(rows)-len(ok)-dropped_en} out-of-duration)", flush=True)
    budget = target_h * 3600
    chosen, got, covered = [], 0.0, set()
    # Lazy greedy (CELF): a row's gain can only shrink as coverage grows, so a
    # stale top-of-heap that still beats the runner-up after re-scoring is
    # exactly the true argmax. This searches all candidates -- the earlier
    # top-4000 window silently capped coverage at 386 of an available 601.
    import heapq
    heap = [(-len(r['hits'])/max(r['dur'], 1.0), i) for i, r in enumerate(ok) if r['hits']]
    heapq.heapify(heap)
    while heap and got < budget:
        neg, i = heapq.heappop(heap)
        r = ok[i]
        new_terms = r['hits'] - covered
        if not new_terms:
            continue
        g = len(new_terms)/max(r['dur'], 1.0)
        if heap and -g > heap[0][0]:
            heapq.heappush(heap, (-g, i))      # stale: re-score and retry
            continue
        r['_taken'] = True; chosen.append(r); got += r['dur']; covered |= new_terms
    print(f"  phase1 set-cover: {got/3600:.1f} h, {len(covered)}/{len(lex)} OOV terms covered", flush=True)
    # phase 2: fill remaining budget, OOV-dense first, hash-shuffled for speaker/topic spread
    rest = [r for r in ok if not r.get('_taken')]
    rest.sort(key=lambda r: (-len(r['hits']),
                             hashlib.md5(r['id'].encode()).hexdigest()))
    for r in rest:
        if got >= budget: break
        r['_taken'] = True; chosen.append(r); got += r['dur']; covered |= r['hits']
    print(f"  phase2 fill    : {got/3600:.1f} h, {len(covered)}/{len(lex)} OOV terms covered", flush=True)
    return chosen, covered

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--shards', default='/workspace/oov_distill/stage')
    p.add_argument('--oov', default='/root/tts/TTS_training/oov_words.txt')
    p.add_argument('--target-h', type=float, default=300)
    p.add_argument('--min-dev', type=float, default=0.7)
    p.add_argument('--dmin', type=float, default=5.0)
    p.add_argument('--dmax', type=float, default=60.0)
    p.add_argument('--cache', default='/workspace/oov_distill/out/scan_cache.jsonl')
    p.add_argument('--out', default='/workspace/oov_distill/out/selected.jsonl')
    a = p.parse_args()
    toks, lex = load_oov(a.oov)
    print(f"OOV: {len(toks)} total -> {len(lex)} lexical", flush=True)
    import glob
    paths = sorted(glob.glob(os.path.join(a.shards, 'shard_*', '*.parquet')))
    print(f"scanning {len(paths)} shards", flush=True)
    cache = a.cache
    if cache and os.path.exists(cache):
        print(f"loading scan cache {cache}", flush=True)
        rows = []
        for l in open(cache, encoding='utf-8'):
            r = json.loads(l); r['hits'] = frozenset(r['hits']); rows.append(r)
    else:
        rows = []
        for n, pth in enumerate(paths):
            rows += scan_shard(pth, lex)
            if (n+1) % 25 == 0:
                print(f"  scanned {n+1}/{len(paths)} shards, {len(rows)} rows, "
                      f"{sum(r['dur'] for r in rows)/3600:.0f} h", flush=True)
        if cache:
            with open(cache, 'w', encoding='utf-8') as fh:
                for r in rows:
                    d = dict(r); d['hits'] = sorted(d['hits'])
                    fh.write(json.dumps(d, ensure_ascii=False) + '\n')
            print(f"cached scan -> {cache}", flush=True)
    print(f"scanned {len(rows)} rows / {sum(r['dur'] for r in rows)/3600:.0f} h", flush=True)
    chosen, covered = select(rows, lex, a.target_h, a.min_dev, a.dmin, a.dmax)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        for r in chosen:
            r = dict(r); r['hits'] = sorted(r.pop('hits')); r.pop('_taken', None)
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    miss = [t for t in lex if t not in covered]
    json.dump({'target_h': a.target_h, 'rows': len(chosen),
               'hours': sum(r['dur'] for r in chosen)/3600,
               'oov_lexical': len(lex), 'oov_covered': len(covered),
               'oov_missing': miss},
              open(a.out.replace('.jsonl', '_report.json'), 'w'), ensure_ascii=False, indent=2)
    print(f"WROTE {a.out}: {len(chosen)} rows, "
          f"{sum(r['dur'] for r in chosen)/3600:.1f} h, "
          f"OOV {len(covered)}/{len(lex)} covered, {len(miss)} missing", flush=True)
