#!/usr/bin/env python
"""Upload the Seed-VC shards (0109+) to milanakdj/nepali-oov-distilled.

Only the new shards are sent; 0000-0108 already exist upstream and are left alone.
Refuses to run under any account other than milanakdj.
"""
import os, sys
from huggingface_hub import HfApi, whoami

REPO = "milanakdj/nepali-oov-distilled"
ROOT = "/workspace/oov_distill/out/hf"

name = whoami().get("name")
if name != "milanakdj":
    sys.exit(f"REFUSING: token authenticates as {name!r}, expected 'milanakdj'")

new = [d for d in sorted(os.listdir(ROOT))
       if d.startswith("shard_") and int(d.split("_")[1]) >= 109]
if len(new) != 104:
    sys.exit(f"expected 104 new shards, found {len(new)}")
print(f"[auth] {name} | uploading {len(new)} shards {new[0]}..{new[-1]}", flush=True)

HfApi().upload_folder(
    folder_path=ROOT, repo_id=REPO, repo_type="dataset",
    allow_patterns=[f"{d}/*" for d in new],
    commit_message="Add 104 Seed-VC shards (0109-0212): 20,663 clips / 236.5 h, "
                   "speaker-gate passed, clean vc only")
print("[done] https://huggingface.co/datasets/" + REPO, flush=True)
