"""WER/CER for both models AND for the real human audio of the same utterances.

The human row is the point of this script. Whisper large-v3-turbo is mediocre on
Nepali, so an absolute TTS WER means nothing on its own -- the same lesson that
made every emotion2vec number in this project unusable. Scoring the real held-out
human recording through the identical pipeline gives the floor that the model
numbers have to be read against.

normalize_text / script_ratios are copied from
synthetic_pipeline/asr_qc_service/server.py (importing it would load FastAPI and a
second copy of the model at import time). script_ratios catches Whisper's habit of
romanizing Nepali audio, which yields CER ~1.0 on audio that is actually fine.
"""
import json, os, re, string, unicodedata, sys
import jiwer
from faster_whisper import WhisperModel

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
_EXTRA_PUNCT = "।॥‘’“”–—…"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + _EXTRA_PUNCT)
_PUNCT_CHARS = frozenset(string.punctuation + _EXTRA_PUNCT)
_DEV = ((0x0900, 0x097F),)
_LAT = ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF))


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


pairs = json.load(open(f"{F}/pairs.json"))
_lim = int(os.environ.get("EVAL_LIMIT", "0"))
if _lim:
    pairs = pairs[:_lim]
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def transcribe(path):
    segs, _ = model.transcribe(path, language="ne", beam_size=5)
    return " ".join(s.text for s in segs).strip()


SYS = {"real_human": None,
       "teacher_24l": f"{F}/wav/teacher_24l",
       "student_6l": f"{F}/wav/student_6l"}
rows = {}
for name, d in SYS.items():
    out = []
    for p in pairs:
        path = p["real_audio"] if d is None else f"{d}/{p['id']}.wav"
        if not os.path.exists(path):
            continue
        hyp = transcribe(path)
        ref_n, hyp_n = normalize_text(p["text"]), normalize_text(hyp)
        if not ref_n:
            continue
        out.append({"id": p["id"], "source": p["source"], "hyp": hyp,
                    "wer": jiwer.wer(ref_n, hyp_n),
                    "cer": jiwer.cer(ref_n, hyp_n),
                    "dev_ratio": dev_ratio(hyp)})
    rows[name] = out
    # A romanized Whisper output is a broken measurement, not a broken clip.
    good = [r for r in out if r["dev_ratio"] >= 0.5]
    n_rom = len(out) - len(good)
    if good:
        print(f"== {name}: n={len(good)} (+{n_rom} discarded as romanized) "
              f"WER {sum(r['wer'] for r in good)/len(good):.3f}  "
              f"CER {sum(r['cer'] for r in good)/len(good):.3f}", flush=True)
json.dump(rows, open(f"{F}/wer.json", "w"), ensure_ascii=False, indent=1)
print("WER DONE")
