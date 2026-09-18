"""Did voice conversion cost intelligibility? Measures CER delta, pre vs post VC.

Compares flex(original) against flex(converted) rather than against the dataset
text, because the Premal text is NOT verbatim -- digits are written as digits but
spoken verbalised ("२१" is read "एक्काइस"), so scoring against it would charge VC
for a mismatch that was there before conversion. The pre/post delta isolates what
VC actually changed, and matches how the production pipeline scored
cer_source/cer_final.

flex trains at max_duration 30 and degrades on long audio, so both sides are
decoded on the same leading 25 s window.
"""
import sys, os, json, random, re, unicodedata, argparse
sys.path.insert(0, '/workspace/models/indic-transcribe-flex')
import numpy as np, soundfile as sf, librosa, warnings
warnings.filterwarnings('ignore')

def norm(s):
    s = unicodedata.normalize('NFC', str(s)).replace('\xa0', ' ')
    s = re.sub(r'[^\w\sऀ-ॿ]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    if not r: return None
    d = np.arange(len(h) + 1)
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j-1] + 1, prev + (rc != hc))
            prev = cur
    return d[len(h)] / len(r)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--secs', type=float, default=25.0)
    a = ap.parse_args()
    from indic_transcribe import IndicTranscribe
    asr = IndicTranscribe.from_pretrained('/workspace/models/indic-transcribe-flex')
    man = [json.loads(l) for l in open('/workspace/oov_distill/out/vc/vc_manifest.jsonl', encoding='utf-8')]
    random.Random(0).shuffle(man)
    man = man[:a.n]
    rows = []
    for k, m in enumerate(man):
        so = f"/workspace/oov_distill/out/vc_src/{m['id']}.wav"
        vc = f"/workspace/oov_distill/out/vc/{m['id']}_vc.wav"
        if not (os.path.exists(so) and os.path.exists(vc)): continue
        try:
            hs, hv = [], []
            for p in (so, vc):
                x, sr = librosa.load(p, sr=16000)
                sf.write('/tmp/_cer.wav', x[:int(a.secs*16000)], 16000)
                (hs if p == so else hv).append(str(asr('/tmp/_cer.wav', lang='ne')))
            c = cer(hs[0], hv[0])          # original transcript as the reference
        except Exception as e:
            print(f"  {m['id']} fail {type(e).__name__}", flush=True); continue
        if c is None: continue
        rows.append(dict(id=m['id'], cer_delta=round(c, 4), cos_ref=m['cos_out_ref'], passed=m['passed']))
        if (k+1) % 20 == 0: print(f"  {k+1}/{len(man)}", flush=True)
    v = np.array([r['cer_delta'] for r in rows])
    print(f"\nn={len(v)}  CER(converted vs original transcript)")
    print(f"  mean {v.mean():.4f}  p50 {np.median(v):.4f}  p90 {np.percentile(v,90):.4f}")
    print(f"  clips where VC changed >10% of characters: {(v>0.10).mean():.0%}")
    pa = np.array([r['cer_delta'] for r in rows if r['passed']])
    fa = np.array([r['cer_delta'] for r in rows if not r['passed']])
    if len(pa) and len(fa):
        print(f"  gate PASS rows: mean CER {pa.mean():.4f} (n={len(pa)})")
        print(f"  gate FAIL rows: mean CER {fa.mean():.4f} (n={len(fa)})")
    json.dump(rows, open('/workspace/oov_distill/out/vc/cer_check.json','w'), indent=2)
