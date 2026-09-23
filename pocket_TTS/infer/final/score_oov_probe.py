"""Score the OOV probe set: does non-Devanagari input survive synthesis at all?

WER is the wrong primary metric here. The v2 failure was *deletion* -- 'मैले 45
रुपैयाँ तिरें।' produced 0.8 s of audio reading only 'मैले' -- and an averaged WER
buries that in the same range as ordinary mispronunciation. So the headline number
is TAIL RETENTION: did the last content word of the sentence get spoken at all.

Probes are synthesized RAW, without ne_frontend. That is the point: v3's claim is
that the model no longer loses content when the frontend is bypassed. Pass --frontend
to score the normalized path instead (which v2 also passes -- that is the workaround,
not the fix).

    $ASR_VENV score_oov_probe.py            # raw, the real test
    $ASR_VENV score_oov_probe.py --frontend # normalized, the workaround's ceiling
"""
import argparse, json, os, re, sys, unicodedata
import numpy as np, torch, librosa, jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
ap = argparse.ArgumentParser()
ap.add_argument("--wav-dir")
ap.add_argument("--out", default=f"{F}/oov_probe.json")
ap.add_argument("--frontend", action="store_true")
args = ap.parse_args()
# Separate directories per mode: one run's audio silently entering the other's
# results is the classic way to get a wrong answer out of this harness.
args.wav_dir = args.wav_dir or f"{F}/wav/oov_probe_{'frontend' if args.frontend else 'raw'}"

MODEL_ID = os.environ.get("FT_ASR", "himalaya-ai/whisper-large-v3-nepali-final")
_TOKFILE = "/root/.hf_whisper_ne_token"
HF_TOK = (open(_TOKFILE).read().strip() if os.path.exists(_TOKFILE)
          else os.environ.get("HF_TOKEN"))
# OOV_DEVICE=cpu forces CPU: during a training run the GPU is held by the trainer
# and a large-v3 encoder alongside it is an OOM risk to the run, not just to us.
# fp16 is a CUDA-only choice here -- half matmuls on CPU are unsupported/slow.
DEVICE = os.environ.get("OOV_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_PUNCT = str.maketrans("", "", "।॥‘’“”–—…" + "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def norm(t):
    """Fold the orthographic variation the ASR is free to pick between.

    Anusvara and candrabindu (ं / ँ) are written interchangeably in Nepali, so
    'तिरें' and 'तिरेँ' are the same word. Not folding them scored three correct
    utterances as content loss on the first run.
    """
    t = unicodedata.normalize("NFC", t or "").translate(_PUNCT).lower()
    t = t.replace("\u0901", "\u0902")           # ँ -> ं
    return re.sub(r"\s+", " ", t).strip()


processor = WhisperProcessor.from_pretrained(MODEL_ID, language="nepali",
                                             task="transcribe", token=HF_TOK)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, token=HF_TOK).to(DEVICE).eval()
tok = processor.tokenizer
tid = lambda t: tok.convert_tokens_to_ids(t)
EOS = tid("<|endoftext|>")
# Measured 2026-09-18 on this release: sot x1 and sot x2 give byte-identical
# output. The card's doubled-sot warning applied to intermediate checkpoints.
PREFIX = [tid("<|startoftranscript|>"), tid("<|ne|>"), tid("<|transcribe|>"),
          tid("<|notimestamps|>")]


def transcribe(batch, max_new_tokens=225):
    feats = processor.feature_extractor(batch, sampling_rate=16000,
                                        return_tensors="pt").input_features.to(DEVICE, dtype=model.dtype)
    cur = torch.tensor([PREFIX] * len(batch), device=DEVICE)
    out, past = cur, None
    done = torch.zeros(len(batch), dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        enc = model.get_encoder()(feats)
        for _ in range(max_new_tokens):
            res = model(encoder_outputs=enc, decoder_input_ids=cur,
                        past_key_values=past, use_cache=True)
            past = res.past_key_values
            nxt = res.logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            out = torch.cat([out, nxt[:, None]], dim=1)
            done |= nxt == EOS
            if bool(done.all()):
                break
            cur = nxt[:, None]
    return processor.batch_decode(out, skip_special_tokens=True)


probes = json.load(open(f"{F}/oov_probes.json"))["probes"]
rows, missing = [], []
for p in probes:
    w = f"{args.wav_dir}/{p['id']}.wav"
    if not os.path.exists(w):
        missing.append(p["id"]); continue
    rows.append((p, w))
if missing:
    print(f"missing {len(missing)} wavs (run gen_oov_probe.py first): {missing[:5]}")
if not rows:
    sys.exit("nothing to score")

hyps = []
for i in range(0, len(rows), 8):
    chunk = rows[i:i + 8]
    hyps += transcribe([librosa.load(w, sr=16000)[0] for _, w in chunk])

out, by_kind = [], {}
for (p, w), hyp in zip(rows, hyps):
    dur = len(librosa.load(w, sr=16000)[0]) / 16000
    kept = norm(p["tail_anchor"]) in norm(hyp)
    rec = {"id": p["id"], "kind": p["kind"], "text": p["text"], "hyp": hyp.strip(),
           "audio_sec": round(dur, 2), "tail_kept": kept,
           "cer": jiwer.cer(norm(p["text"]), norm(hyp)) if norm(p["text"]) else None}
    out.append(rec)
    by_kind.setdefault(p["kind"], []).append(kept)

json.dump({"frontend": args.frontend, "rows": out}, open(args.out, "w"),
          ensure_ascii=False, indent=1)

kept = sum(r["tail_kept"] for r in out)
print(f"\n{'kind':20s} {'tail kept':>10s}")
for k in sorted(by_kind):
    v = by_kind[k]
    print(f"{k:20s} {sum(v):>5d}/{len(v):<4d}")
print(f"\nTAIL RETENTION {kept}/{len(out)} ({100*kept/len(out):.0f}%)"
      f"   frontend={'on' if args.frontend else 'OFF (raw input)'}")
for r in out:
    if not r["tail_kept"]:
        print(f"  LOST  [{r['kind']}] {r['text'][:46]}  ->  {r['hyp'][:46]!r} ({r['audio_sec']}s)")
