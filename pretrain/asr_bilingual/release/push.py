#!/usr/bin/env python
"""Create milanakdj/nemotron-asr-nepali-0.6b (PRIVATE) and upload the release dir.

Refuses to run unless the active token resolves to the expected account, so a
stale HF_TOKEN cannot silently publish under the wrong namespace.
"""
import os, sys
from huggingface_hub import HfApi, whoami

EXPECT_USER = os.environ.get("EXPECT_USER", "milanakdj")
REPO = os.environ.get("REPO_ID", "milanakdj/nemotron-asr-nepali-0.6b")
SRC = "/workspace/asr_bilingual/release"

who = whoami()
name = who.get("name")
if name != EXPECT_USER:
    sys.exit(f"REFUSING: token authenticates as {name!r}, expected {EXPECT_USER!r}. "
             "Export a milanakdj write token and re-run.")
print(f"[auth] {name}")

need = [f for f in ("nemotron_ne_nepali_best.nemo", "README.md")
        if not os.path.exists(os.path.join(SRC, f))]
if need:
    sys.exit(f"missing from {SRC}: {need}")

api = HfApi()
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
print(f"[repo] {REPO} (private)")
api.upload_folder(folder_path=SRC, repo_id=REPO, repo_type="model",
                  commit_message="Nepali ASR snapshot @ ~147.7k steps (WER 0.195 / CER 0.075)")
print(f"[done] https://huggingface.co/{REPO}")
