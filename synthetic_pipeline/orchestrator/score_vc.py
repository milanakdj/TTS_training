"""Per-row score for every kept=="vc" row: did the conversion actually land on a
real reference speaker, or just blur the Edge TTS voice?

Emits id, vc_shift, ref_sim, ref_cluster, edge_like, vc_worked.
"""
import json, os, glob, warnings, time
import numpy as np
import multiprocessing as mp
warnings.filterwarnings("ignore")

SP  = "/root/tts/TTS_training/synthetic_pipeline/manifests"
MAN = "/root/tts/TTS_training/synthetic_pipeline/manifests/production_manifest.jsonl"
REFD= "/root/tts/TTS_training/synthetic_pipeline/manifests/reference_pool_v2"
NPROC = 14
_enc = None

def _init():
    global _enc
    from resemblyzer import VoiceEncoder
    import torch
    torch.set_num_threads(1)
    _enc = VoiceEncoder(device="cuda", verbose=False)

def _embed(path):
    from resemblyzer import preprocess_wav
    try:
        return _enc.embed_utterance(preprocess_wav(path)).astype(np.float32)
    except Exception:
        return np.zeros(256, dtype=np.float32)

def run(paths, tag):
    t0 = time.time()
    with mp.get_context("spawn").Pool(NPROC, initializer=_init) as pool:
        out = []
        for i, e in enumerate(pool.imap(_embed, paths, chunksize=32), 1):
            out.append(e)
            if i % 20000 == 0:
                r = i / (time.time() - t0)
                print(f"  {tag} {i}/{len(paths)}  {r:.0f}/s  eta {(len(paths)-i)/r/60:.0f}min", flush=True)
    return np.vstack(out)

def main():
    rows = []
    for l in open(MAN):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("kept") in ("vc", "raw_fallback"):
            rows.append(r)
    vc  = [r for r in rows if r["kept"] == "vc"]
    raw = [r for r in rows if r["kept"] == "raw_fallback"]
    print(f"{len(vc)} vc rows to score", flush=True)

    # source-voice centroids from pure Edge TTS rows
    import random; random.seed(0); random.shuffle(raw)
    cents = {}
    for v in ("ne-NP-HemkalaNeural", "ne-NP-SagarNeural"):
        sub = [r["audio_path"] for r in raw if r["voice"] == v][:600]
        E = run(sub, f"centroid {v}")
        c = E.mean(0); cents[v] = c / np.linalg.norm(c)
    print("centroids done", flush=True)

    E_ref = run(sorted(glob.glob(REFD + "/*.wav")), "references")
    # collapse the reference pool into speaker clusters (it has 102 duplicate groups)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = np.clip(1 - E_ref @ E_ref.T, 0, None); np.fill_diagonal(D, 0)
    rlab = fcluster(linkage(squareform(D, checks=False), "average"), t=0.15, criterion="distance")
    rsize = np.bincount(rlab)
    dominant = int(rsize.argmax())
    print(f"reference pool -> {rlab.max()} speaker clusters, dominant #{dominant} = {rsize[dominant]}/600 clips", flush=True)

    E = run([r["audio_path"] for r in vc], "vc rows")
    np.save(f"{SP}/vc_emb.npy", {"ids": [r["sentence_id"] for r in vc], "E": E,
                                 "E_ref": E_ref, "rlab": rlab, "cents": cents}, allow_pickle=True)

    src = np.vstack([cents[r["voice"]] for r in vc])
    vc_shift = 1 - np.einsum("ij,ij->i", E, src)
    S = E @ E_ref.T
    ref_sim = S.max(1)
    ref_cluster = rlab[S.argmax(1)]
    edge_like = vc_shift < 0.07                      # cos >= 0.93 to its own Edge voice
    worked = (~edge_like) & (ref_sim >= 0.88)

    np.savez(f"{SP}/vc_scores.npz", ids=np.array([str(r["sentence_id"]) for r in vc]),
             voice=np.array([r["voice"] for r in vc]), vc_shift=vc_shift.astype("float32"),
             ref_sim=ref_sim.astype("float32"), ref_cluster=ref_cluster.astype("int16"),
             dominant=dominant, edge_like=edge_like, vc_worked=worked)
    print(f"\nwrote vc_scores.npz ({len(vc)} rows) -- run stage 'parquet' to convert")
    print(f"  edge_like (VC never took) : {edge_like.sum():6d}  {100*edge_like.mean():.1f}%")
    print(f"  ref_sim >= 0.88           : {(ref_sim>=0.88).sum():6d}  {100*(ref_sim>=0.88).mean():.1f}%")
    print(f"  vc_worked                 : {worked.sum():6d}  {100*worked.mean():.1f}%")
    print(f"  of those, on a NON-dominant reference speaker: {(worked & (ref_cluster!=dominant)).sum()}")


def to_parquet():
    """orchestrator/.venv -- has pyarrow; vc_service/.venv (which runs main) does not."""
    import pyarrow as pa, pyarrow.parquet as pq
    d = np.load(f"{SP}/vc_scores.npz", allow_pickle=True)
    dominant = int(d["dominant"])
    t = pa.table({
        "id": pa.array(d["ids"].tolist()),
        "voice": pa.array(d["voice"].tolist()),
        "vc_shift": pa.array(d["vc_shift"]),
        "ref_sim": pa.array(d["ref_sim"]),
        "ref_cluster": pa.array(d["ref_cluster"]),
        "ref_cluster_is_dominant": pa.array(d["ref_cluster"] == dominant),
        "edge_like": pa.array(d["edge_like"]),
        "vc_worked": pa.array(d["vc_worked"]),
    })
    pq.write_table(t, f"{SP}/vc_speaker_scores.parquet")
    print(f"wrote {SP}/vc_speaker_scores.parquet ({t.num_rows} rows)")


if __name__ == "__main__":
    import sys
    (to_parquet if len(sys.argv) > 1 and sys.argv[1] == "parquet" else main)()
