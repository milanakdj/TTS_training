"""Evaluate a finetuned nemotron-3.5-asr-streaming-0.6b checkpoint on ne/en gold test.

Reuses baseline_eval.py's measurement (jiwer wer/cer on RAW manifest text, no
normalisation) so the numbers are directly comparable to the pre-training
baseline of WER 1.440 / CER 0.878 on the 600-clip Nepali gold test.

Two things baseline_eval.py got wrong that this fixes, both reported explicitly:

1. PROMPT MODE.  transcribe() takes a `target_lang` kwarg, but it is a *fallback*
   only: `_transcribe_forward` uses the per-sample prompt indices the dataloader
   emits (batch[4]) whenever they exist, and with a manifest they always exist.
   Those indices come from LhotseSpeechToTextBpeDatasetWithPromptIndex, whose
   `default_prompt_mode` is "unified" -> each cut independently takes the
   language-agnostic `auto` prompt (index 101) with probability 0.5, else the
   real language id (ne-NP=46, en-US=0).  So the baseline was measured on a
   random ~50/50 mix of auto and ne-NP prompts, and was not reproducible run to
   run.  The prompt mode is read from cut.custom["prompt_mode"], i.e. a manifest
   field, so this script rewrites the manifest with prompt_mode=langID
   (--prompt-mode to change) and the language pinned in `lang`/`target_lang`.

2. --limit used to truncate the references but not the hypotheses: the old
   script sliced `rows` to N yet handed the *whole* manifest to transcribe().
   Here the temp manifest is truncated, so both sides shrink together.

Usage:
    python eval_ckpt.py <ckpt.nemo|ckpt.ckpt|ckpt_dir> [--manifest ...] [--limit N]
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import unicodedata

BASE_NEMO = ("/workspace/hf_cache/hub/models--nvidia--nemotron-3.5-asr-streaming-0.6b/"
             "snapshots/ea30d66debe3740a08b573244286791d423d6b3e/"
             "nemotron-3.5-asr-streaming-0.6b.nemo")
MANIFESTS = "/root/tts/TTS_training/pretrain/asr_bilingual/manifests"
LOGS = "/root/tts/TTS_training/pretrain/asr_bilingual/logs"
CKPT_DIR = "/root/tts/TTS_training/pretrain/asr_bilingual/ckpt/nemotron_ne_en"

# Pre-training baseline, raw (unnormalised) jiwer on manifests/ne_test.jsonl, n=600.
# Reproduced exactly from logs/baseline_ne.json.
BASELINE = {"ne-NP": {"wer": 1.440, "cer": 0.878, "n": 600}}


# ---------------------------------------------------------------- text handling
_LANGTAG = re.compile(r"<[a-z]{2}-[A-Z]{2}>")
# Python's \w does NOT match Devanagari combining marks (matras are category
# Mn/Mc), so a plain [^\w\s] strip would delete every vowel sign.  Keep the whole
# Devanagari block U+0900-U+097F, then take back its punctuation: danda,
# double danda, abbreviation sign, high spacing dot.
_PUNCT = re.compile(r"[^\w\sऀ-ॿ]|[।॥॰ॱ]", re.UNICODE)


def normalize(s):
    """Diagnostic normalisation only -- NOT what the baseline used.

    Training text is raw manifest text (danda, latin parentheticals and all), so
    the headline number stays raw.  This strips the <xx-YY> language tags the
    model emits, punctuation and case, to separate "wrong words" from "right
    words, wrong surface form".
    """
    s = unicodedata.normalize("NFC", s)
    s = _LANGTAG.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def score(refs, hyps, norm=False):
    from jiwer import cer, wer
    if norm:
        pairs = [(normalize(r), normalize(h)) for r, h in zip(refs, hyps)]
        pairs = [(r, h) for r, h in pairs if r]  # jiwer dies on empty references
        if not pairs:
            return float("nan"), float("nan"), 0
        refs, hyps = [list(x) for x in zip(*pairs)]
    return wer(refs, hyps), cer(refs, hyps), len(refs)


# ---------------------------------------------------------------- checkpoint io
def resolve_ckpt(path):
    """Accept a .nemo, a .ckpt, or a directory; return a concrete file path."""
    if os.path.isdir(path):
        cands = (sorted(glob.glob(os.path.join(path, "**", "*.nemo"), recursive=True)) +
                 sorted(glob.glob(os.path.join(path, "**", "*.ckpt"), recursive=True)))
        cands = [c for c in cands if "-last" not in os.path.basename(c)] or cands
        if not cands:
            sys.exit(f"no .nemo/.ckpt found under {path}")
        # newest wins; .nemo preferred at equal recency because it restores cleanly
        path = max(cands, key=lambda p: (os.path.getmtime(p), p.endswith(".nemo")))
        print(f"[ckpt] resolved directory -> {path}")
    if not os.path.exists(path):
        sys.exit(f"checkpoint not found: {path}")
    return path


def load_model(path, base_nemo, device):
    import torch
    import nemo.collections.asr as nemo_asr

    if path.endswith(".nemo"):
        m = nemo_asr.models.ASRModel.restore_from(path, map_location=device)
    elif path.endswith(".ckpt"):
        # A Lightning .ckpt carries weights but not the tokenizer/prompt_dictionary
        # artifacts, and EncDecRNNTBPEModelWithPrompt cannot be built from the
        # checkpoint config alone (see its restore_from docstring).  Restore the
        # architecture from the pretrained .nemo and overwrite the weights.
        print(f"[ckpt] .ckpt -> restoring architecture from {base_nemo}")
        m = nemo_asr.models.ASRModel.restore_from(base_nemo, map_location=device)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if not k.startswith(("loss.", "wer."))}
        miss, unexp = m.load_state_dict(sd, strict=False)
        miss = [k for k in miss if not k.startswith(("loss.", "wer."))]
        print(f"[ckpt] loaded state_dict: {len(sd)} tensors, "
              f"{len(miss)} missing, {len(unexp)} unexpected")
        for k in miss[:8]:
            print(f"       MISSING  {k}")
        for k in unexp[:8]:
            print(f"       UNEXPECT {k}")
        if len(miss) > 0.02 * len(sd):
            sys.exit("too many missing keys -- this is not the same architecture")
    else:
        sys.exit(f"unknown checkpoint extension: {path}")
    m.eval()
    return m


# ---------------------------------------------------------------- manifest prep
def prep_manifest(src, lang, prompt_mode, limit, out_path):
    """Copy the test manifest, pinning language and prompt mode per utterance.

    `lang`/`target_lang` feed the lhotse supervision language (lang_field defaults
    to "lang" at transcribe time); `prompt_mode` lands in cut.custom and makes the
    prompt index deterministic.
    """
    rows = []
    with open(src) as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            r = json.loads(line)
            r["lang"] = lang
            r["target_lang"] = lang
            r["prompt_mode"] = prompt_mode
            rows.append(r)
    if not rows:
        sys.exit(f"empty manifest: {src}")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def infer_lang(src):
    with open(src) as f:
        r = json.loads(f.readline())
    return r.get("target_lang") or r.get("lang") or "en-US"


# ---------------------------------------------------------------- report
def report(tag, refs, hyps, elapsed, lang):
    print(f"\n== {tag}  [{lang}]  n={len(refs)}   [{elapsed:.0f}s]")
    w, c, _ = score(refs, hyps, norm=False)
    print(f"   RAW        WER {w:.3f}   CER {c:.3f}    <- comparable to baseline")
    nw, nc, nn = score(refs, hyps, norm=True)
    print(f"   NORMALISED WER {nw:.3f}   CER {nc:.3f}   (n={nn}, diagnostic only)")

    empty = sum(1 for h in hyps if not h.strip())
    tagged = sum(1 for h in hyps if _LANGTAG.search(h))
    unk = sum(1 for h in hyps if "⁇" in h)
    print(f"   empty hyps {empty}/{len(hyps)}   <xx-YY> tag {tagged}   ⁇ {unk}")

    b = BASELINE.get(lang)
    if b:
        dw, dc = w - b["wer"], c - b["cer"]
        rw = (1 - w / b["wer"]) * 100 if b["wer"] else float("nan")
        print(f"   baseline   WER {b['wer']:.3f}   CER {b['cer']:.3f}   (n={b['n']}, raw, "
              f"pre-training)")
        print(f"   delta      WER {dw:+.3f}   CER {dc:+.3f}   ({rw:+.1f}% relative WER)")
        if b["n"] != len(refs):
            print(f"   NOTE: baseline n={b['n']} but this run n={len(refs)}; "
                  "subset scores are indicative only")
    else:
        print(f"   no pre-training baseline recorded for {lang}")
    return {"wer": w, "cer": c, "wer_norm": nw, "cer_norm": nc,
            "n": len(refs), "empty": empty, "seconds": elapsed}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", nargs="?", default=None,
                   help=".nemo, .ckpt, or a directory to search "
                        f"(e.g. {CKPT_DIR}); optional only with --resume --out")
    p.add_argument("--manifest", default=f"{MANIFESTS}/ne_test.jsonl")
    p.add_argument("--lang", default=None,
                   help="ne-NP or en-US; default: read target_lang from the manifest")
    p.add_argument("--limit", type=int, default=0,
                   help="evaluate only the first N clips (0 = all)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0,
                   help="0 keeps dataloader order identical to the manifest")
    p.add_argument("--prompt-mode", default="langID", choices=["langID", "auto", "unified"],
                   help="langID forces the real language prompt (default and correct); "
                        "unified reproduces the baseline's random 50%% auto mix")
    p.add_argument("--base", default=BASE_NEMO, help="pretrained .nemo, for .ckpt loads")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="write per-utterance refs/hyps here")
    p.add_argument("--resume", action="store_true",
                   help="if --out already holds the right number of hypotheses, "
                        "re-score them instead of running the model (no GPU)")
    a = p.parse_args()

    lang = a.lang or infer_lang(a.manifest)

    refs_src = [json.loads(l) for l in open(a.manifest)]
    if a.limit:
        refs_src = refs_src[:a.limit]
    refs = [r["text"] for r in refs_src]

    # Resolving the checkpoint is skipped when --resume can answer from cache, so
    # a re-score works before (or after) any checkpoint exists.
    cached_only = a.resume and a.out and os.path.exists(a.out)
    if not cached_only and not a.ckpt:
        sys.exit("a checkpoint path is required (unless --resume with an existing --out)")
    ckpt = None if cached_only else resolve_ckpt(a.ckpt)
    tag = os.path.basename(ckpt) if ckpt else "cache"
    out = a.out or os.path.join(
        LOGS, "eval_%s_%s%s.json" % (lang.split("-")[0], os.path.splitext(tag)[0],
                                     f"_n{a.limit}" if a.limit else ""))

    if a.resume and os.path.exists(out):
        cached = json.load(open(out))
        rows = cached["rows"] if isinstance(cached, dict) else cached
        if len(rows) == len(refs):
            print(f"[resume] re-scoring {len(rows)} cached hypotheses from {out}")
            report(tag + " (cached)", [r["ref"] for r in rows], [r["hyp"] for r in rows],
                   0.0, lang)
            return
        print(f"[resume] {out} has {len(rows)} rows, need {len(refs)} -- re-running")

    tmp = os.path.join(LOGS, f".eval_tmp_{lang}_{os.getpid()}.jsonl")
    prep_manifest(a.manifest, lang, a.prompt_mode, a.limit, tmp)
    print(f"[prep] {len(refs)} clips, lang={lang}, prompt_mode={a.prompt_mode} -> {tmp}")

    try:
        model = load_model(ckpt, a.base, a.device)
        pd = model.cfg.model_defaults.get("prompt_dictionary", {}) or {}
        if lang not in pd:
            sys.exit(f"'{lang}' is not in the model's prompt_dictionary "
                     f"(have e.g. {list(pd)[:6]}) -- results would be garbage")
        print(f"[prompt] {lang} -> index {pd[lang]}   (auto={pd.get('auto')})")

        t = time.time()
        hyps = model.transcribe(tmp, batch_size=a.batch_size,
                                num_workers=a.num_workers, verbose=False,
                                target_lang=lang)
        el = time.time() - t
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    if hyps and isinstance(hyps[0], list):  # some decoders return [best, nbest]
        hyps = hyps[0]
    hyps = [h.text if hasattr(h, "text") else str(h) for h in hyps]
    if len(hyps) != len(refs):
        sys.exit(f"length mismatch: {len(hyps)} hypotheses for {len(refs)} references "
                 "-- refusing to score misaligned pairs")

    summary = report(tag, refs, hyps, el, lang)
    for i in range(min(3, len(refs))):
        print(f"  REF: {refs[i][:90]}")
        print(f"  HYP: {hyps[i][:90]}")

    summary.update(ckpt=ckpt, manifest=a.manifest, lang=lang,
                   prompt_mode=a.prompt_mode, batch_size=a.batch_size)
    json.dump({"summary": summary,
               "rows": [{"id": i, "ref": r, "hyp": h}
                        for i, (r, h) in enumerate(zip(refs, hyps))]},
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
