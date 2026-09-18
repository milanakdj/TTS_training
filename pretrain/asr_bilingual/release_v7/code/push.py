"""Publish the run_v7 release folder to the Hub, refusing to run outside milanakdj.

Guard first, upload second. The box's ambient HF_TOKEN authenticates as a
different account, so a push that "just works" is the failure mode this exists
to prevent.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami

REPO = "milanakdj/parakeet-ctc-nepali-110m"
HERE = Path(__file__).resolve().parent.parent

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("REFUSING: set HF_TOKEN explicitly for this invocation")

name = whoami(token=token).get("name")
if name != "milanakdj":
    sys.exit(f"REFUSING: token authenticates as {name!r}, expected 'milanakdj'")

api = HfApi(token=token)
api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)

# Everything except this script's own directory traversal quirks; the .nemo is
# the payload and the rest is the record of how it was made.
api.upload_folder(
    repo_id=REPO,
    repo_type="model",
    folder_path=str(HERE),
    ignore_patterns=["code/push.py", "__pycache__/*", "*.pyc"],
    commit_message="run_v7_encinit: 110M parakeet-CTC Nepali, 50k steps, human WER 0.176",
)

print(f"pushed -> https://huggingface.co/{REPO}")
