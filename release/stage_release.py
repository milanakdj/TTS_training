"""Stage the Nepali pocket-TTS student for a Hugging Face push.

Repo id is a parameter, not a constant: config.yaml, the card and the example all
have to reference the repo by name via hf:// paths, so changing the destination
means regenerating these three files rather than editing them by hand.

    python3 stage_release.py <repo_id>
"""
import json, os, shutil, sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "himalaya-ai/pocket-tts-nepali-6l"
# --brief summarises the training-data section instead of listing per-source hours.
BRIEF = "--brief" in sys.argv
OUT_SUFFIX = "_brief" if BRIEF else ""
SRC = "/root/tts/TTS_training/pocket_TTS"
OUT = "/root/tts/TTS_training/release/staged" + ("_brief" if "--brief" in sys.argv else "")
WEIGHTS = "/workspace/nepali_student_6l/model_final_200k.safetensors"

os.makedirs(f"{OUT}/tokenizer", exist_ok=True)
for src, dst in [(WEIGHTS, f"{OUT}/model.safetensors"),
                 (f"{SRC}/tokenizer/nepali_bpe4000.model", f"{OUT}/tokenizer/nepali_bpe4000.model"),
                 (f"{SRC}/tokenizer/nepali_bpe4000.vocab", f"{OUT}/tokenizer/nepali_bpe4000.vocab")]:
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copy(src, dst)
        print(f"copied {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.1f} MB)")

# --- config.yaml: identical architecture to the local infer config, but every path
# points into this repo so `load_model(config="hf://<repo>/config.yaml")` self-resolves.
cfg = open(f"{SRC}/infer/nepali_student_6l.yaml").read()
cfg = cfg.replace("# Inference config for the Nepali 6-LAYER STUDENT (stage 2, depth-distilled).\n"
                  "# Weights are the final step-200000 export (EMA), snapshotted after the\n"
                  "# training process exited.\n",
                  "# Nepali pocket-TTS, 6-layer depth-distilled student (step 200000, EMA).\n"
                  "# Every path is an hf:// self-reference, so this config can be loaded\n"
                  "# directly and it will fetch its own weights and tokenizer.\n")
cfg = cfg.replace(f"weights_path: {WEIGHTS}", f"weights_path: hf://{REPO}/model.safetensors")
cfg = cfg.replace(f"tokenizer_path: {SRC}/tokenizer/nepali_bpe4000.model",
                  f"tokenizer_path: hf://{REPO}/tokenizer/nepali_bpe4000.model")
cfg = cfg.replace("# Human evals preferred this model at 0.3 (equal WER/similarity/speech rate).\n", "")
assert "hf://" in cfg and WEIGHTS not in cfg and f"{SRC}/tokenizer" not in cfg, "path rewrite failed"
open(f"{OUT}/config.yaml", "w").write(cfg)
print("wrote config.yaml")

# --- eval_results.json: the measured numbers, machine-readable.
F = f"{SRC}/infer/final"


def agg(path, key="wer"):
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    out = {}
    for sysname, rows in d.items():
        good = [r for r in rows if r.get("dev_ratio", 1) >= 0.5] if key != "sim" else rows
        if good:
            out[sysname] = {"n": len(good),
                            **{m: round(sum(r[m] for r in good) / len(good), 4)
                               for m in (["wer", "cer"] if key != "sim" else ["sim"])}}
    return out


ev = {
    "eval_set": {"n_utterances": 100, "n_identities": 77,
                 # The brief variant reports speech types rather than dataset names.
                 "sources": (["spontaneous_snr40_50", "spontaneous_snr50",
                              "read_speech", "read_speech_long",
                              "spontaneous_curated", "acted_studio"] if BRIEF else
                             ["ans_snr40-50", "ans_snr50", "indicvoices-r",
                              "indicvoices-r-long", "mahadhwani", "rasa"]),
                 "protocol": ("held-out target utterance; voice prompt is a different clip "
                              "of the same speaker identity taken from the train split")},
    "asr_finetuned_nepali_whisper": agg(f"{F}/wer_ft.json"),
    "asr_base_whisper_large_v3_turbo": agg(f"{F}/wer.json"),
    "speaker_similarity_resemblyzer": agg(f"{F}/sim.json", "sim"),
    "cpu_real_time_factor": {k: round(sum(r["rtf"] for r in v) / len(v), 2)
                             for k, v in (json.load(open(f"{F}/bench.json")).items()
                                          if os.path.exists(f"{F}/bench.json") else [])},
    "notes": ("real_human rows are the ceiling: the genuine recording of the same "
              "utterance scored through the identical pipeline. Absolute values are "
              "not comparable across the two ASR rows."),
}
json.dump(ev, open(f"{OUT}/eval_results.json", "w"), indent=1)
print("wrote eval_results.json")
# --- card + example: templated so the repo id can never drift between the config's
# hf:// paths and the documentation that tells people to use them.
card = "README.brief.md.tmpl" if BRIEF else "README.md.tmpl"
for tmpl, dst in [(card, "README.md"), ("inference.py.tmpl", "inference.py")]:
    body = open(f"/root/tts/TTS_training/release/templates/{tmpl}").read()
    assert "__REPO_ID__" in body, tmpl
    open(f"{OUT}/{dst}", "w").write(body.replace("__REPO_ID__", REPO))
    print(f"wrote {dst}")

print(f"\nstaged for {REPO} -> {OUT}")
print("\n".join(f"  {f}" for f in sorted(os.listdir(OUT))))
