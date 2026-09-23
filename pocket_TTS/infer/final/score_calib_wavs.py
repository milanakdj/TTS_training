"""CER of each calibration threshold, Nepali-finetuned Whisper. Picks the winner.

Reuses score_wer_ft.py's decode loop and normalization so the number is on the
same scale as the headline eval. The romanization filter applies here too: a
clip Whisper answers in Latin script is an unreadable measurement, not a bad
clip.
"""
import json, os, sys, glob, re, string, unicodedata
import torch, librosa, jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
name = sys.argv[1]
MODEL_ID = "himalaya-ai/whisper-large-v3-nepali-final"
HF_TOK = open("/root/.hf_whisper_ne_token").read().strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EXTRA = "।॥‘’“”–—…"
_TABLE = str.maketrans("", "", string.punctuation + _EXTRA)
_PUNCT = frozenset(string.punctuation + _EXTRA)

def norm(t):
    t = unicodedata.normalize("NFC", t or "").translate(_TABLE).lower()
    return re.sub(r"\s+", " ", t).strip()

def dev_ratio(t):
    ch = [c for c in unicodedata.normalize("NFC", t or "")
          if not c.isspace() and c not in _PUNCT and unicodedata.category(c)[0] not in ("P", "S")]
    return sum(1 for c in ch if 0x900 <= ord(c) <= 0x97F) / len(ch) if ch else 0.0

proc = WhisperProcessor.from_pretrained(MODEL_ID, language="nepali", task="transcribe", token=HF_TOK)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, token=HF_TOK).to(DEVICE).eval()
tok = proc.tokenizer
tid = lambda t: tok.convert_tokens_to_ids(t)
PREFIX = [tid("<|startoftranscript|>"), tid("<|startoftranscript|>"),
          tid("<|ne|>"), tid("<|transcribe|>"), tid("<|notimestamps|>")]
EOS = tid("<|endoftext|>")

def transcribe(batch, max_new_tokens=225):
    feats = proc.feature_extractor(batch, sampling_rate=16000,
                                   return_tensors="pt").input_features.to(DEVICE, dtype=model.dtype)
    cur = torch.tensor([PREFIX] * len(batch), device=DEVICE)
    out, past = cur, None
    done = torch.zeros(len(batch), dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        enc = model.get_encoder()(feats)
        for _ in range(max_new_tokens):
            res = model(encoder_outputs=enc, decoder_input_ids=cur, past_key_values=past, use_cache=True)
            past = res.past_key_values
            nxt = res.logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
            out = torch.cat([out, nxt[:, None]], dim=1)
            done |= nxt == EOS
            if bool(done.all()):
                break
            cur = nxt[:, None]
    return proc.batch_decode(out, skip_special_tokens=True)

items = json.load(open(f"{F}/calib_v3.json"))
res = {}
# real_human is the ceiling and the check on the instrument: this project has
# twice lost weeks to a metric that could not see what it was measuring, so a
# model CER is only readable next to the human CER on the same clips.
dirs = sorted(glob.glob(f"{F}/wav/calib_{name}_th*"))
if os.environ.get("WITH_HUMAN"):
    dirs = ["REAL_HUMAN"] + dirs
for d in dirs:
    th = "real_human" if d == "REAL_HUMAN" else d.split("_th")[-1]
    rows = []
    for i in range(0, len(items), 8):
        chunk = items[i:i + 8]
        paths = [it["real_audio"] if d == "REAL_HUMAN" else f"{d}/{it['id']}.wav"
                 for it in chunk]
        chunk = [(it, p) for it, p in zip(chunk, paths) if os.path.exists(p)]
        if not chunk:
            continue
        hyps = transcribe([librosa.load(p, sr=16000)[0] for _, p in chunk])
        for (it, _), h in zip(chunk, hyps):
            r, hh = norm(it["text"]), norm(h)
            if r:
                rows.append({"wer": jiwer.wer(r, hh), "cer": jiwer.cer(r, hh), "dev": dev_ratio(h)})
    good = [r for r in rows if r["dev"] >= 0.5]
    if good:
        res[th] = {"n": len(good), "discarded": len(rows) - len(good),
                   "wer": sum(r["wer"] for r in good) / len(good),
                   "cer": sum(r["cer"] for r in good) / len(good)}
        print(f"  th {th:>5}: n={res[th]['n']:2d} (+{res[th]['discarded']} romanized)  "
              f"WER {res[th]['wer']:.3f}  CER {res[th]['cer']:.3f}", flush=True)
best = min((k for k in res if k != "real_human"), key=lambda k: res[k]["cer"])
print(f"\nBEST eos_threshold for {name} by CER: {best}  (CER {res[best]['cer']:.3f})")
json.dump({"model": name, "criterion": "CER on calib_v3.json (40 utts, disjoint from pairs.json)",
           "results": res, "best": float(best)},
          open(f"{F}/calib_content_{name}.json", "w"), indent=1)
