"""WER/CER with the Nepali-FINETUNED Whisper, not base large-v3.

Base large-v3-turbo scores the REAL human held-out recordings at WER 1.15 -- the
instrument is blind on Nepali, so it cannot rank two TTS systems. This uses
himalaya-ai/whisper-large-v3-nepali-final (10.98% WER / 3.43% CER on its own clean
test split). TTS_training/whisper/CLAUDE.md still calls it
milanakdj/whisper-large-v3-nepali-final-largev3_1 -- that name 404s; the repo moved
to the himalaya-ai org and is private.

The token comes from /root/.hf_whisper_ne_token (0600) and is passed only to these
two from_pretrained calls. It is deliberately NOT exported as HF_TOKEN: the shared
HF_TOKEN in this environment belongs to another account that other processes here
use, and overwriting it would break them.

That checkpoint was fine-tuned with a DOUBLED <|startoftranscript|> in the decoder
prefix. Decoded with the standard single-sot prefix it emits endless nonsense
(WER > 400%). The decode loop below is the canonical one from
TTS_training/whisper/whisper-eval.py and must not be replaced with pipeline(),
faster-whisper, or a plain model.generate() -- none of them can express the prefix.

Its training corpus is clean single-speaker studio audio, so it is NOT trusted a
priori on the noisy YouTube sources here. That is what the real_human row is for:
if the human ceiling for a source is bad, the model numbers for that source are
unreadable and are reported as such rather than believed.
"""
import json, os, re, string, unicodedata
import numpy as np
import torch, librosa, jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
MODEL_ID = os.environ.get("FT_ASR", "himalaya-ai/whisper-large-v3-nepali-final")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TOKFILE = "/root/.hf_whisper_ne_token"
HF_TOK = (open(_TOKFILE).read().strip() if os.path.exists(_TOKFILE)
          else os.environ.get("HF_TOKEN"))

_EXTRA_PUNCT = "।॥‘’“”–—…"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + _EXTRA_PUNCT)
_PUNCT_CHARS = frozenset(string.punctuation + _EXTRA_PUNCT)
_DEV = ((0x0900, 0x097F),)


def normalize_text(t):
    t = unicodedata.normalize("NFC", t or "").translate(_PUNCT_TABLE).lower()
    return re.sub(r"\s+", " ", t).strip()


def _is_punct(c):
    return c in _PUNCT_CHARS or unicodedata.category(c)[0] in ("P", "S")


def dev_ratio(t):
    chars = [c for c in unicodedata.normalize("NFC", t or "")
             if not c.isspace() and not _is_punct(c)]
    if not chars:
        return 0.0
    return sum(1 for c in chars if any(lo <= ord(c) <= hi for lo, hi in _DEV)) / len(chars)


processor = WhisperProcessor.from_pretrained(
    MODEL_ID, language="nepali", task="transcribe", token=HF_TOK)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, token=HF_TOK).to(DEVICE).eval()
tok = processor.tokenizer
tid = lambda t: tok.convert_tokens_to_ids(t)
# The doubled <|startoftranscript|> is the whole point -- one sot decodes garbage.
PREFIX = [tid("<|startoftranscript|>"), tid("<|startoftranscript|>"),
          tid("<|ne|>"), tid("<|transcribe|>"), tid("<|notimestamps|>")]
EOS = tid("<|endoftext|>")


def transcribe(batch, max_new_tokens=225):
    feats = processor.feature_extractor(
        batch, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE, dtype=model.dtype)
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


pairs = json.load(open(f"{F}/pairs.json"))
_lim = int(os.environ.get("EVAL_LIMIT", "0"))
if _lim:
    pairs = pairs[:_lim]
BS = int(os.environ.get("ASR_BS", "8"))
SYS = {"real_human": None,
       "teacher_24l": f"{F}/wav/teacher_24l",
       "student_6l": f"{F}/wav/student_6l"}
rows = {}
for name, d in SYS.items():
    items = [(p, p["real_audio"] if d is None else f"{d}/{p['id']}.wav") for p in pairs]
    items = [(p, path) for p, path in items if os.path.exists(path)]
    out = []
    for i in range(0, len(items), BS):
        chunk = items[i:i + BS]
        audio = [librosa.load(path, sr=16000)[0] for _, path in chunk]
        hyps = transcribe(audio)
        for (p, _), hyp in zip(chunk, hyps):
            ref_n, hyp_n = normalize_text(p["text"]), normalize_text(hyp)
            if not ref_n:
                continue
            out.append({"id": p["id"], "source": p["source"], "hyp": hyp,
                        "wer": jiwer.wer(ref_n, hyp_n), "cer": jiwer.cer(ref_n, hyp_n),
                        "dev_ratio": dev_ratio(hyp)})
    rows[name] = out
    good = [r for r in out if r["dev_ratio"] >= 0.5]
    if good:
        print(f"== {name}: n={len(good)} (+{len(out)-len(good)} discarded as romanized) "
              f"WER {sum(r['wer'] for r in good)/len(good):.3f}  "
              f"CER {sum(r['cer'] for r in good)/len(good):.3f}", flush=True)
json.dump(rows, open(f"{F}/wer_ft.json", "w"), ensure_ascii=False, indent=1)
print("WER-FT DONE")
