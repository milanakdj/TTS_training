import json, io, os, soundfile as sf, pyarrow.parquet as pq
SRC='/workspace/oov_distill/out/vc_src'; os.makedirs(SRC, exist_ok=True)
rows=[json.loads(l) for l in open('/workspace/oov_distill/out/selected.jsonl',encoding='utf-8')]
rows=[r for r in rows if not r.get('oov_core')]
rows.sort(key=lambda r:(r['shard'], r['rg'], r['idx']))
print(f"extracting {len(rows)} clips ({sum(r['dur'] for r in rows)/3600:.1f} h)", flush=True)
cache={}; n=0; skipped=0
for r in rows:
    p=f"{SRC}/{r['id']}.wav"
    if os.path.exists(p): skipped+=1; continue
    key=(r['shard'],r['rg'])
    if key not in cache:
        cache.clear()
        cache[key]=pq.ParquetFile(f"/workspace/oov_distill/stage/{r['shard']}/t.parquet").read_row_group(r['rg']).to_pandas()
    b=cache[key].iloc[r['idx']]['audio']['bytes']
    x,sr=sf.read(io.BytesIO(b),dtype='float32')
    sf.write(p,x,sr); n+=1
    if n%2000==0: print(f"  {n} written / {skipped} skipped", flush=True)
print(f"DONE: {n} written, {skipped} already present, total {n+skipped}")
