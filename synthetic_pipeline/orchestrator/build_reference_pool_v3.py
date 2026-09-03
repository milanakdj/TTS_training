"""
Reference-voice pool v3 -- replaces v2, which passed a pitch-std check while
being 74% one speaker (443/600 clips) with 353 duplicate files, all from FLEURS.

Two changes from v2:
  1. Sample by SPEAKER, not by row. v2 drew 600 uniformly from 1,562 candidates,
     so FLEURS's row count (1,355) bought it 89% of the pool while lilgoose777 --
     224 clips, 0 duplicates, no voice above 9% -- got 68 slots. v3 embeds every
     candidate, clusters, dedupes by content hash, and takes a per-speaker cap.
  2. Validate with speaker embeddings, not pitch statistics. Pitch std cannot
     tell one expressive speaker from forty; it read 42.7 Hz on the broken pool.
"""
import hashlib, io, os, sys, warnings
import numpy as np
import soundfile as sf
warnings.filterwarnings("ignore")

OUT_DIR   = "/root/tts/TTS_training/synthetic_pipeline/manifests/reference_pool_v3"
INDEX_PATH= "/root/tts/TTS_training/synthetic_pipeline/manifests/reference_pool_v3_index.parquet"
TARGET    = 600
PER_SPEAKER_CAP = 10      # no voice may exceed this many clips
CLUSTER_COS     = 0.85
DUR = (2.0, 15.0)

SOURCES = [
    ("lilgoose777", "lilgoose777/nepali-tts-massive-combined",
     ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]),
    ("fleurs_ne_np", "himalaya-ai/nep-voice-tts-compilation",
     [f"data/finetune/google_fleurs_ne_np/ne_np/train/google_fleurs_ne_np-ne_np-train-{i:05d}.parquet"
      for i in range(10)]),
]

def collect():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    cands, seen = [], set()
    for name, repo, files in SOURCES:
        kept = 0
        for f in files:
            t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"), columns=["audio"])
            for r in t.column("audio").to_pylist():
                b = r["bytes"]
                try:
                    info = sf.info(io.BytesIO(b))
                    d = info.frames / info.samplerate
                except Exception:
                    continue
                if not (DUR[0] <= d <= DUR[1]):
                    continue
                h = hashlib.md5(b).hexdigest()
                if h in seen:            # v2 shipped 353 duplicate files; drop them here
                    continue
                seen.add(h)
                cands.append({"bytes": b, "source": name})
                kept += 1
        print(f"  {name}: {kept} unique candidates", flush=True)
    return cands


CAND_DIR = "/root/tts/TTS_training/synthetic_pipeline/manifests/pool_v3_candidates"

def stage1():
    """orchestrator/.venv -- has huggingface_hub + pyarrow, no resemblyzer."""
    import json
    os.makedirs(CAND_DIR, exist_ok=True)
    cands = collect()
    meta = []
    for i, c in enumerate(cands):
        data, sr = sf.read(io.BytesIO(c["bytes"]), dtype="int16")
        p = os.path.join(CAND_DIR, f"cand_{i:05d}.wav")
        sf.write(p, data, sr, subtype="PCM_16")
        meta.append({"path": p, "source": c["source"]})
    json.dump(meta, open(os.path.join(CAND_DIR, "candidates.json"), "w"))
    print(f"stage1: {len(meta)} unique candidates -> {CAND_DIR}")

def stage2():
    """vc_service/.venv -- has resemblyzer + scipy, no pyarrow."""
    import json
    from concurrent.futures import ThreadPoolExecutor
    cands = json.load(open(os.path.join(CAND_DIR, "candidates.json")))
    print(f"[1/3] {len(cands)} candidates from stage1", flush=True)

    print("[2/3] embedding...", flush=True)
    from resemblyzer import VoiceEncoder, preprocess_wav
    enc = VoiceEncoder(device="cuda", verbose=False)
    with ThreadPoolExecutor(8) as ex:
        E = np.array(list(ex.map(lambda c: enc.embed_utterance(preprocess_wav(c["path"])), cands)))

    print("[3/3] clustering + per-speaker sampling...", flush=True)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = np.clip(1 - E @ E.T, 0, None); np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), "average"),
                   t=1 - CLUSTER_COS, criterion="distance")
    rng = np.random.RandomState(42)
    picked = []
    # round-robin across speakers so the pool stays flat even if TARGET is small
    by_spk = {s: list(rng.permutation(np.where(lab == s)[0])) for s in np.unique(lab)}
    order = sorted(by_spk, key=lambda s: -len(by_spk[s]))
    for depth in range(PER_SPEAKER_CAP):
        for s in order:
            if depth < len(by_spk[s]) and len(picked) < TARGET:
                picked.append(by_spk[s][depth])
    picked = np.array(picked)

    os.makedirs(OUT_DIR, exist_ok=True)
    import shutil
    rows = []
    for i, j in enumerate(picked):
        p = os.path.join(OUT_DIR, f"ref_{i:04d}.wav")
        shutil.copy(cands[j]["path"], p)
        rows.append({"ref_id": i, "path": p, "source": cands[j]["source"],
                     "speaker_cluster": int(lab[j])})
    json.dump(rows, open(INDEX_PATH.replace(".parquet", ".json"), "w"))

    Ep = E[picked]
    sizes = np.bincount(lab[picked]); sizes = sizes[sizes > 0]
    pair = (Ep @ Ep.T)[np.triu_indices(len(Ep), 1)]
    print(f"\nwrote {len(picked)} clips to {OUT_DIR}")
    print(f"  speakers: {len(sizes)} | biggest {sizes.max()} clips ({100*sizes.max()/len(picked):.1f}%)")
    print(f"  mean pairwise cos {pair.mean():.3f}   (v2 was 0.796, biggest speaker 74%)")
    import collections
    print("  by source:", dict(collections.Counter(r["source"] for r in rows)))
    if sizes.max() / len(picked) > 0.15:
        print("  WARNING: a single voice still exceeds 15% of the pool", file=sys.stderr)

def stage3():
    """orchestrator/.venv -- JSON index -> parquet, which run_production.py reads."""
    import json, pyarrow as pa, pyarrow.parquet as pq
    rows = json.load(open(INDEX_PATH.replace(".parquet", ".json")))
    pq.write_table(pa.Table.from_pylist(rows), INDEX_PATH)
    print(f"stage3: wrote {INDEX_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    {"1": stage1, "2": stage2, "3": stage3}[sys.argv[1]]()
