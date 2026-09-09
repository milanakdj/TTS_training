"""Publish the Nepali Pocket-TTS student, and update the Indic-Parler card.

The token is read from /root/.hf_milanakdj (the same file scripts/train_*.sh read)
and is never printed. The ambient HF_TOKEN belongs to a different account and is
deliberately not used -- see docs/ADR.md ADR-012.

    python3 push_to_hf.py check
    python3 push_to_hf.py model  <repo_id> [--gated manual|auto]
    python3 push_to_hf.py parler <repo_id>
"""
import os, sys
from huggingface_hub import HfApi

TOKFILE = "/root/.hf_milanakdj"
TOK = os.environ.get("HF_UPLOAD_TOKEN") or (
    open(TOKFILE).read().strip() if os.path.exists(TOKFILE) else None)
if not TOK:
    sys.exit(f"no token: set HF_UPLOAD_TOKEN or provide {TOKFILE}")
api = HfApi(token=TOK)
cmd = sys.argv[1] if len(sys.argv) > 1 else "check"


def check():
    me = api.whoami()
    tok = (me.get("auth") or {}).get("accessToken") or {}
    print(f"account   : {me.get('name')}")
    print(f"token role: {tok.get('role')}  ({tok.get('displayName')})")
    orgs = [o.get("name") for o in me.get("orgs", [])]
    print(f"orgs      : {orgs or 'none'}")
    # A fine-grained token can be scoped per-namespace, so "can I write" is only
    # answerable per target. Report the namespaces that are even candidates.
    print(f"\ncandidate namespaces: {[me.get('name')] + orgs}")


def push_model(repo, gated):
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=repo, repo_type="model", folder_path="/workspace/hf_release/staged",
        commit_message="Nepali Pocket-TTS 6L distilled student (step 200000)")
    print(f"uploaded -> https://huggingface.co/{repo}")
    # extra_gated_* card metadata does NOT enable gating on its own, and neither
    # does a `gated:` key in the YAML. Only the settings API does.
    api.update_repo_settings(repo_id=repo, repo_type="model", gated=gated)
    info = api.model_info(repo, expand=["gated"])   # reads false without expand
    print(f"gated = {info.gated}")


def push_parler(repo):
    for name in ("README.md", "CAPTIONS.txt"):
        api.upload_file(
            path_or_fileobj=f"/workspace/hf_release/staged_parler/{name}",
            path_in_repo=name, repo_id=repo, repo_type="model",
            commit_message="Record the negative result and the best base-model captions")
        print(f"uploaded {name}")
    print(f"-> https://huggingface.co/{repo}")


if cmd == "check":
    check()
elif cmd == "model":
    g = sys.argv[sys.argv.index("--gated") + 1] if "--gated" in sys.argv else "manual"
    push_model(sys.argv[2], g)
elif cmd == "parler":
    push_parler(sys.argv[2])
else:
    sys.exit(__doc__)
