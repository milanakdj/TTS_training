"""Fetch ~2,000 h of MLS English (CC-BY-4.0) for the English replay/retrain mix.

44.5k h lives in 1,416 parquet shards at ~31 h / 0.5 GB each, so 64 shards is
~2,000 h for ~32 GB. Downloading shards (not streaming) because the tarred-manifest
build reads them several times.
"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

REPO = "parler-tts/mls_eng"
OUT = "/root/tts/TTS_training/pretrain/asr_bilingual/data/mls_eng"
N = int(os.environ.get("N_SHARDS", 64))
os.makedirs(OUT, exist_ok=True)

files = [f"data/train-{i:05d}-of-01416.parquet" for i in range(N)]
files += ["data/dev-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]

def get(f):
    for attempt in range(4):
        try:
            p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=OUT)
            print(f"ok {f}", flush=True); return p
        except Exception as e:
            print(f"[retry {attempt}] {f}: {type(e).__name__} {str(e)[:120]}", flush=True)
    print(f"FAILED {f}", flush=True); return None

with ThreadPoolExecutor(6) as pool:
    got = list(pool.map(get, files))
print(f"[done] {sum(1 for g in got if g)}/{len(files)} shards", flush=True)
