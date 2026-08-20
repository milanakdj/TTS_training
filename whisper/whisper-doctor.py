#!/usr/bin/env python3
"""Diagnose and repair a Whisper checkpoint that decodes garbage.

The failure this targets: Whisper's output head `proj_out` is weight-tied to the
decoder token embeddings, so it is NOT stored in the checkpoint -- that is the
"missing keys in the checkpoint model loaded: ['proj_out.weight']" line in the
training log. Loading re-creates it by tying, but ONLY if config.json says
tie_word_embeddings: true. If it says false, the head is left randomly
initialised and the model emits endless nonsense (WER in the hundreds) while
every training metric looked perfect, because training never used from_pretrained.

The fix is one boolean in config.json. No re-saving 6GB of weights.

Runs on CPU. No dataset download. The probe feeds 30 seconds of silence: a
healthy Whisper returns nothing or one short phrase, a random head returns a full
225 tokens of junk. That distinction needs no reference transcript.

Usage:
    python whisper-doctor.py --ckpt ~/whisper-output/large-v3/checkpoint-26550
    python whisper-doctor.py --ckpt ... --probe            # + silence decode (slow, CPU)
    python whisper-doctor.py --ckpt ... --fix-config       # patch config.json in place
    python whisper-doctor.py --ckpt ... --fix-config \
        --push-config milanakdj/whisper-large-v3-nepali-final-largev3_1
    python whisper-doctor.py --self-check
"""
import argparse
import glob
import json
import os
import shutil


def read_config(ckpt):
    path = os.path.join(ckpt, "config.json")
    if not os.path.isfile(path):
        raise SystemExit(f"No config.json in {ckpt}")
    with open(path) as f:
        return path, json.load(f)


def weight_keys(ckpt):
    """Names stored in the checkpoint. Sharded saves list them in the index file;
    a single-file save needs the safetensors header read."""
    index = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return set(json.load(f)["weight_map"])
    single = os.path.join(ckpt, "model.safetensors")
    if os.path.isfile(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as f:
            return set(f.keys())
    if glob.glob(os.path.join(ckpt, "pytorch_model*.bin")):
        return None  # .bin: reading keys means unpickling the whole file, skip it
    raise SystemExit(f"No model weights found in {ckpt}")


def diagnose(tie_flag, proj_out_stored):
    """Pure decision table. tie_flag: config's tie_word_embeddings (may be None if
    absent -- HF then falls back to the class default, which is True for Whisper).
    proj_out_stored: is proj_out.weight in the checkpoint (None = unknown)."""
    tie = True if tie_flag is None else bool(tie_flag)
    if tie:
        return "ok", (
            "tie_word_embeddings is true, so from_pretrained rebuilds proj_out from "
            "the decoder embeddings. This is NOT the cause of the garbage output."
        )
    if proj_out_stored:
        return "ok", (
            "tie_word_embeddings is false, but proj_out.weight IS stored in the "
            "checkpoint, so the real trained head gets loaded. Not the cause."
        )
    return "broken", (
        "tie_word_embeddings is false AND proj_out.weight is absent from the "
        "checkpoint. The output head is randomly initialised on every load. This "
        "fully explains WER above 100%. Fix with --fix-config."
    )


def probe_silence(ckpt):
    """Decode 30s of silence on CPU. Healthy: few tokens. Random head: 225."""
    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    _, cfg = read_config(ckpt)
    d, mel = cfg.get("d_model"), cfg.get("num_mel_bins")
    variant = {384: "tiny", 512: "base", 768: "small", 1024: "medium"}.get(d)
    if d == 1280:
        variant = "large-v3" if mel == 128 else "large-v2"
    if variant is None:
        raise SystemExit(f"Unrecognised Whisper size: d_model={d}, num_mel_bins={mel}")
    base = f"openai/whisper-{variant}"
    print(f"[probe] {base} (d_model={d}, mel_bins={mel}); loading on CPU, "
          f"this takes a minute for large models", flush=True)

    processor = WhisperProcessor.from_pretrained(base, language="nepali", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        ckpt, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    tied = torch.equal(model.proj_out.weight, model.model.decoder.embed_tokens.weight)
    print(f"[probe] proj_out_tied={tied}", flush=True)

    feats = processor.feature_extractor(
        np.zeros(16000 * 30, dtype="float32"), sampling_rate=16000, return_tensors="pt"
    ).input_features
    with torch.no_grad():
        ids = model.generate(feats, max_new_tokens=225)
    n_tok = ids.shape[-1]
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"[probe] silence -> {n_tok} tokens: {text[:300]!r}", flush=True)
    if n_tok >= 220:
        print("[probe] VERDICT: the model ran to the token limit on silence. That is "
              "a broken output head or a broken decode, not a weak model.", flush=True)
    else:
        print("[probe] VERDICT: sane token count on silence. The output head works; "
              "look elsewhere for the garbage.", flush=True)
    return tied, n_tok, text


def fix_config(ckpt, push_repo=None):
    path, cfg = read_config(ckpt)
    if cfg.get("tie_word_embeddings") is True:
        print(f"[fix] {path} already has tie_word_embeddings: true -- nothing to do")
    else:
        shutil.copy2(path, path + ".bak")
        cfg["tie_word_embeddings"] = True
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"[fix] set tie_word_embeddings: true in {path} (backup: {path}.bak)")

    if push_repo:
        from huggingface_hub import HfApi
        HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
            path_or_fileobj=path,
            path_in_repo="config.json",
            repo_id=push_repo,
            commit_message="Fix tie_word_embeddings: proj_out was loading randomly",
        )
        print(f"[fix] pushed config.json to https://huggingface.co/{push_repo}")


def self_check():
    assert diagnose(False, False)[0] == "broken"
    assert diagnose(False, None)[0] == "broken"      # unknown = assume absent
    assert diagnose(False, True)[0] == "ok"          # real head is stored
    assert diagnose(True, False)[0] == "ok"          # tying rebuilds it
    assert diagnose(None, False)[0] == "ok"          # absent key -> Whisper default true
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="local checkpoint directory")
    ap.add_argument("--probe", action="store_true",
                    help="decode 30s of silence on CPU as an end-to-end sanity test")
    ap.add_argument("--fix-config", action="store_true",
                    help="set tie_word_embeddings: true in the checkpoint's config.json")
    ap.add_argument("--push-config", default=None, metavar="REPO_ID",
                    help="also upload the fixed config.json to this Hub repo")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        return self_check()
    if not args.ckpt:
        raise SystemExit("--ckpt is required (or use --self-check)")
    if not os.path.isdir(args.ckpt):
        raise SystemExit(f"{args.ckpt} is not a local directory. Download the repo "
                         f"first: hf download <repo> --local-dir <dir>")

    path, cfg = read_config(args.ckpt)
    keys = weight_keys(args.ckpt)
    proj_out_stored = None if keys is None else ("proj_out.weight" in keys)
    tie_flag = cfg.get("tie_word_embeddings")

    print(f"[doctor] {args.ckpt}")
    print(f"  size                  : d_model={cfg.get('d_model')} "
          f"num_mel_bins={cfg.get('num_mel_bins')} vocab={cfg.get('vocab_size')}")
    print(f"  tie_word_embeddings   : {tie_flag}"
          f"{' (key absent -> HF default true)' if tie_flag is None else ''}")
    print(f"  proj_out.weight stored: {proj_out_stored}"
          f"{' (unknown: .bin checkpoint)' if proj_out_stored is None else ''}")

    verdict, why = diagnose(tie_flag, proj_out_stored)
    print(f"  VERDICT [{verdict}]      : {why}")

    if args.fix_config:
        fix_config(args.ckpt, args.push_config)
    elif verdict == "broken":
        print("\nRe-run with --fix-config to patch it.")

    if args.probe:
        probe_silence(args.ckpt)


if __name__ == "__main__":
    main()
