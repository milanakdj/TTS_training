"""
Pack the finished synthetic Nepali TTS rows into HF parquet shards and push them
to himalaya-ai/nepali-tts-synthetic-v2.

Audio is embedded as-is: the original 24 kHz mono 16-bit WAV bytes, no re-encode.

Disk is the binding constraint here (~189 GB free vs ~141 GB of audio), so this
builds ONE shard at a time, uploads it, and deletes it before starting the next.
Peak local footprint is a single shard (~3 GB) rather than the whole dataset.

Fully resumable: shards already present in the repo are skipped, so re-running
after an interruption picks up where it left off.
"""
import json, os, sys, time
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

ROOT = "/root/tts/TTS_training/synthetic_pipeline"
MANIFEST = f"{ROOT}/manifests/production_manifest.jsonl"
REPO_ID = os.environ.get("REPO_ID", "milanakdj/nepali-tts-synthetic-v2")
STAGE = os.environ.get("STAGE", "/tmp/claude-0/-root/5f540b2f-fd6c-46e6-8acb-60a198436033/scratchpad/tts_shards")  # tmpfs: fast, and only ever holds one shard
ROWS_PER_SHARD = 8000              # ~2.9 GB at the observed ~368 KB/row
KEPT = ("vc", "raw_fallback")

# HF's Audio feature expects a struct of the raw file bytes plus a filename.
SCHEMA = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("voice", pa.string()),
    ("kept", pa.string()),
    ("synth_cer", pa.float32()),
    ("vc_cer", pa.float32()),
    ("synth_pass", pa.bool_()),
    ("vc_pass", pa.bool_()),
    ("ref_pool_version", pa.string()),
])


def load_rows():
    """Kept rows only, sorted by sentence_id so sharding is deterministic."""
    rows, missing = [], 0
    with open(MANIFEST) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("kept") not in KEPT:
                continue
            if not os.path.exists(r["audio_path"]):
                missing += 1
                continue
            rows.append(r)
    rows.sort(key=lambda r: r["sentence_id"])
    return rows, missing


def build(rows, path):
    cols = {k: [] for k in SCHEMA.names}
    for r in rows:
        with open(r["audio_path"], "rb") as fh:
            raw = fh.read()
        cols["id"].append(str(r["sentence_id"]))
        cols["text"].append(r["text"])
        cols["audio"].append({"bytes": raw, "path": os.path.basename(r["audio_path"])})
        cols["voice"].append(r.get("voice"))
        cols["kept"].append(r.get("kept"))
        cols["synth_cer"].append(r.get("synth_cer"))
        cols["vc_cer"].append(r.get("vc_cer"))
        cols["synth_pass"].append(r.get("synth_pass"))
        cols["vc_pass"].append(r.get("vc_pass"))
        cols["ref_pool_version"].append(r.get("ref_pool_version"))
    # WAV bytes are incompressible, so skip compression and save the CPU.
    pq.write_table(pa.table(cols, schema=SCHEMA), path, compression="none")


def main():
    # HF_UPLOAD_TOKEN lets this push as a different account than the ambient
    # HF_TOKEN (which belongs to Sunil8bodhan / bodhan-ai).
    token = os.environ.get("HF_UPLOAD_TOKEN") or os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    print("pushing as:", api.whoami()["name"], "->", REPO_ID, flush=True)

    # Public repo; the card's extra_gated_* metadata is what turns on gating,
    # so the card goes up before any audio does.
    api.create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    card = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                    repo_id=REPO_ID, repo_type="dataset",
                    commit_message="dataset card (gated access)")
    print("card uploaded, gating active", flush=True)

    print("reading manifest...", flush=True)
    rows, missing = load_rows()
    total = len(rows)
    nshards = (total + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD
    print(f"kept rows with audio present: {total} (skipped {missing} missing files)", flush=True)
    print(f"-> {nshards} shards x {ROWS_PER_SHARD} rows", flush=True)

    done = {f for f in api.list_repo_files(REPO_ID, repo_type="dataset") if f.endswith(".parquet")}
    print(f"already in repo: {len(done)} shards", flush=True)
    os.makedirs(STAGE, exist_ok=True)
    t0 = time.time()

    for i in range(nshards):
        name = f"data/train-{i:05d}-of-{nshards:05d}.parquet"
        if name in done:
            print(f"[{i+1}/{nshards}] skip (already uploaded)", flush=True)
            continue
        chunk = rows[i * ROWS_PER_SHARD:(i + 1) * ROWS_PER_SHARD]
        local = f"{STAGE}/shard_{i:05d}.parquet"
        b = time.time()
        build(chunk, local)
        gb = os.path.getsize(local) / 1e9
        print(f"[{i+1}/{nshards}] built {gb:.2f} GB in {time.time()-b:.0f}s, uploading...", flush=True)
        u = time.time()
        api.upload_file(path_or_fileobj=local, path_in_repo=name, repo_id=REPO_ID,
                        repo_type="dataset", commit_message=f"shard {i+1}/{nshards}")
        os.remove(local)
        el = (time.time() - t0) / 60
        pct = (i + 1) / nshards
        print(f"[{i+1}/{nshards}] uploaded in {time.time()-u:.0f}s "
              f"({gb/max(time.time()-u,1)*1000:.0f} MB/s) | elapsed {el:.1f}min | "
              f"ETA {el/pct-el:.1f}min", flush=True)

    print("ALL SHARDS UPLOADED", flush=True)


if __name__ == "__main__":
    main()
