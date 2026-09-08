"""Off-site backup of the Nepali training corpus to a gated HF dataset repo.

Backup only -- not a release. Nothing on local disk is modified or renamed: the
WAVs are read in place and the renaming happens only inside the uploaded shards.

What is deliberately NOT uploaded, because a gated repo still exposes its column
schema and file list to anyone:

  * original file paths, which carry YouTube channel and video ids
  * `original_audio_filepath`, which carries internal pipeline paths
  * source dataset names -- mapped to opaque codes s1..sN
  * diarization speaker labels -- hashed

The id -> path mapping needed to restore is written to a LOCAL sidecar
(id_map.jsonl) and is not uploaded here. Keep it with your own backups, or put it
in a small private repo; without it the archive still restores audio + text, just
not the original filenames.

    ORCH=/root/tts/TTS_training/synthetic_pipeline/orchestrator/.venv/bin/python3
    $ORCH backup_corpus.py --dry-run          # plan + sizes, uploads nothing
    $ORCH backup_corpus.py                    # run; resumable, skips done shards

Resumable: shards already present in the repo are skipped, so re-running after an
interruption picks up where it stopped.
"""
import argparse, hashlib, io, json, os, sys, time

MANIFEST = "/root/tts/TTS_training/pocket_TTS/manifests/train_v2_aligned.jsonl"
TOKFILE = "/root/.hf_milanakdj"
# Already safely on the Hub as AI4Bharat releases, so backing them up again is
# wasted storage. indicvoices-r-long is NOT among them and is kept.
EXCLUDE = {"rasa", "indicvoices-r"}

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="milanakdj/nepali-speech-archive")
ap.add_argument("--shards", type=int, default=180)      # ~1 GB each after FLAC
ap.add_argument("--tmp", default="/workspace/hf_release/_shards")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="package only N clips, for a pilot")
args = ap.parse_args()

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi

# ---------------------------------------------------------------- plan
rows = []
for line in open(MANIFEST):
    r = json.loads(line)
    if r["source"] in EXCLUDE:
        continue
    rows.append(r)
rows.sort(key=lambda r: r["path"])          # deterministic sharding
if args.limit:
    rows = rows[: args.limit]

sources = sorted({r["source"] for r in rows})
SRC_CODE = {s: f"s{i+1}" for i, s in enumerate(sources)}   # opaque, order-stable
hours = sum(r["duration"] for r in rows) / 3600
print(f"{len(rows):,} clips, {hours:,.1f} h")
print(f"source codes (kept LOCAL, not uploaded): {SRC_CODE}")
print(f"~{hours * 48004 * 3600 / 1e9:,.0f} GB as WAV, ~{hours * 48004 * 3600 / 1e9 * 0.503:,.0f} GB as FLAC")

n_shards = max(1, min(args.shards, len(rows)))
shards = [rows[i::n_shards] for i in range(n_shards)]      # round-robin: even sizes
print(f"{n_shards} shards, ~{len(shards[0]):,} clips each")

if args.dry_run:
    print("\n--- dry run, nothing uploaded ---")
    sys.exit(0)

# ---------------------------------------------------------------- setup
tok = os.environ.get("HF_UPLOAD_TOKEN") or open(TOKFILE).read().strip()
api = HfApi(token=tok)
api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)
try:
    done = {f for f in api.list_repo_files(args.repo, repo_type="dataset")}
except Exception:
    done = set()
os.makedirs(args.tmp, exist_ok=True)

# The card says as little as possible while still being true. Anyone can read this
# and the column names, so neither mentions provenance.
CARD = """---
language:
- ne
gated: manual
extra_gated_heading: Restricted archive
extra_gated_prompt: >-
  Internal archival copy. Access is not granted for external use, redistribution,
  or publication. Request access only if you are working with the owning team.
extra_gated_fields:
  Name: text
  Affiliation: text
  Reason for access: text
extra_gated_button_content: Request access
---

# Nepali speech archive

Internal archival copy of a Nepali speech corpus, stored for safekeeping. Not a
public dataset release. Approximately 2,000 hours, 24 kHz mono, FLAC.

Access is restricted and is not granted for redistribution or publication.
"""
api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                repo_id=args.repo, repo_type="dataset",
                commit_message="archive card")
# Turning gating ON is a settings-API call; the YAML above does not do it.
api.update_repo_settings(repo_id=args.repo, repo_type="dataset", gated="manual")
print(f"repo ready, gated=manual: https://huggingface.co/datasets/{args.repo}")

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("text", pa.string()),
    ("duration", pa.float32()),
    ("spk", pa.string()),
    ("src", pa.string()),
    ("words", pa.string()),      # alignment, JSON-encoded, needed to restore
])

idmap = open("/workspace/hf_release/id_map.jsonl", "a")
t_start = time.time()
for si, shard in enumerate(shards):
    name = f"data/train-{si:05d}-of-{n_shards:05d}.parquet"
    if name in done:
        print(f"[{si+1}/{n_shards}] skip, already uploaded")
        continue
    ids, audio, texts, durs, spks, srcs, words = [], [], [], [], [], [], []
    for r in shard:
        p = r["path"]
        try:
            data, sr = sf.read(p, dtype="int16", always_2d=False)
        except Exception as e:
            print(f"   unreadable, skipped: {e}", flush=True)
            continue
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="FLAC")
        uid = hashlib.sha256(p.encode()).hexdigest()[:16]
        ids.append(uid)
        audio.append({"bytes": buf.getvalue(), "path": f"{uid}.flac"})
        texts.append(r.get("transcript") or "")
        durs.append(r["duration"])
        # Diarization labels are per-video and meaningless outside their source,
        # but they still group speakers, so hash them with the directory.
        spks.append(hashlib.sha256(
            (os.path.dirname(p) + "|" + str(r.get("speaker"))).encode()).hexdigest()[:12])
        srcs.append(SRC_CODE[r["source"]])
        words.append(json.dumps(r.get("words") or [], ensure_ascii=False))
        idmap.write(json.dumps({"id": uid, "path": p}, ensure_ascii=False) + "\n")
    idmap.flush()

    local = f"{args.tmp}/shard.parquet"
    pq.write_table(pa.table({"id": ids, "audio": audio, "text": texts, "duration": durs,
                             "spk": spks, "src": srcs, "words": words}, schema=SCHEMA),
                   local, compression="zstd")
    mb = os.path.getsize(local) / 1e6
    api.upload_file(path_or_fileobj=local, path_in_repo=name,
                    repo_id=args.repo, repo_type="dataset",
                    commit_message=f"shard {si+1}/{n_shards}")
    os.remove(local)
    el = time.time() - t_start
    print(f"[{si+1}/{n_shards}] {len(ids):,} clips, {mb:,.0f} MB uploaded "
          f"| {el/60:.0f} min elapsed, ~{el/(si+1)*(n_shards-si-1)/60:.0f} min left", flush=True)

idmap.close()
print("\nDONE. id -> path mapping is at /workspace/hf_release/id_map.jsonl (NOT uploaded).")
