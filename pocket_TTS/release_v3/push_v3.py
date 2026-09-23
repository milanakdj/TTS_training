"""Stage and push the v3 teacher + student to the Hub, private, under milanakdj.

The ambient HF_TOKEN in this environment belongs to a different account that
other processes here use, so this reads its own token from a 0600 file and
refuses to run if it does not resolve to milanakdj.

    repo/.venv/bin/python3 release_v3/push_v3.py --stage-only
    repo/.venv/bin/python3 release_v3/push_v3.py --push student
    repo/.venv/bin/python3 release_v3/push_v3.py --push teacher
"""
import argparse, json, os, re, shutil, sys
from pathlib import Path

from huggingface_hub import HfApi

R = Path("/root/tts/TTS_training/pocket_TTS")
STAGE = R / "release_v3" / "stage"
TOKFILE = "/root/.hf_milanakdj_token"
OWNER = "milanakdj"

MODELS = {
    "student": {
        "repo": f"{OWNER}/pocket-tts-nepali-6l-v3-failed",
        "src_cfg": R / "infer/nepali_student_6l_v3.yaml",
        "weights": Path("/workspace/nepali_student_6l_v3/model_final_200k.safetensors"),
        "run_dir": Path("/workspace/nepali_student_6l_v3"),
        "ckpt": None,
    },
    "teacher": {
        "repo": f"{OWNER}/pocket-tts-nepali-24l-teacher-v3-failed",
        "src_cfg": R / "infer/nepali_teacher_24l_v3.yaml",
        "weights": Path("/workspace/nepali_teacher_24l_v3/model_final_200k.safetensors"),
        "run_dir": Path("/workspace/nepali_teacher_24l_v3"),
        # Distilling a new student needs the TRAINING checkpoint: builders.py does a
        # plain torch.load and reads payload["ema"], which the safetensors export has
        # already merged in and cannot be recovered from. v2 shipped this for exactly
        # that reason; keep doing it.
        "ckpt": Path("/workspace/nepali_teacher_24l_v3/checkpoint_00200000.pt"),
    },
}


def token():
    if not os.path.exists(TOKFILE):
        sys.exit(f"missing {TOKFILE}")
    return open(TOKFILE).read().strip()


def check_identity(api):
    who = api.whoami()
    if who.get("name") != OWNER:
        sys.exit(f"token resolves to {who.get('name')!r}, refusing to push (want {OWNER!r})")
    print(f"authenticated as {who['name']}")


def stage(kind):
    m = MODELS[kind]
    d = STAGE / kind
    if d.exists():
        shutil.rmtree(d)
    (d / "tokenizer").mkdir(parents=True)

    # config.yaml: rewrite every local absolute path to an hf:// path in this repo,
    # so the config is loadable by anyone with read access and not just on this box.
    cfg = m["src_cfg"].read_text()
    cfg = re.sub(r"^weights_path: .*$",
                 f"weights_path: hf://{m['repo']}/model.safetensors", cfg, flags=re.M)
    cfg = re.sub(r"^(\s*)tokenizer_path: .*$",
                 rf"\1tokenizer_path: hf://{m['repo']}/tokenizer/ne_en_9682.model",
                 cfg, flags=re.M)
    assert "/root/" not in cfg and "/workspace/" not in cfg, "local path left in config.yaml"
    (d / "config.yaml").write_text(cfg)

    for name in ("ne_en_9682.model", "ne_en_9682.vocab"):
        shutil.copy2(R / "tokenizer_v3" / name, d / "tokenizer" / name)
    shutil.copy2(R / "frontend/ne_frontend.py", d / "ne_frontend.py")
    shutil.copy2(m["run_dir"] / "args.yaml", d / "training_args.yaml")
    shutil.copy2(m["weights"], d / "model.safetensors")

    card = R / "release_v3" / f"README_{kind}.md"
    if card.exists():
        shutil.copy2(card, d / "README.md")
    ev = R / "release_v3" / "eval_results_v3.json"
    if ev.exists():
        shutil.copy2(ev, d / "eval_results.json")
    inf = R / "release_v3" / f"inference_{kind}.py"
    if inf.exists():
        shutil.copy2(inf, d / "inference.py")
    if m["ckpt"]:
        (d / "training").mkdir()
        os.symlink(m["ckpt"], d / "training" / m["ckpt"].name)

    total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    print(f"staged {kind} -> {d}  ({total/1e9:.2f} GB)")
    for f in sorted(d.rglob("*")):
        if f.is_file() or f.is_symlink():
            print(f"   {f.relative_to(d)}")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--push", choices=["student", "teacher"])
    a = ap.parse_args()
    kinds = [a.push] if a.push else list(MODELS)
    for k in kinds:
        d = stage(k)
        if a.stage_only:
            continue
        api = HfApi(token=token())
        check_identity(api)
        repo = MODELS[k]["repo"]
        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        print(f"uploading {d} -> {repo} (private)")
        api.upload_folder(repo_id=repo, folder_path=str(d), repo_type="model",
                          commit_message="v3: grafted ne+en tokenizer, 200k steps")
        info = api.model_info(repo)
        print(f"done: https://huggingface.co/{repo}  private={info.private}")
