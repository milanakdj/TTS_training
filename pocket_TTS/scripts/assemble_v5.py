"""Build the v5 training clips + manifest from QC-passed Kyutai English.

  plain   Kyutai clip as-is
  concat  [real Nepali clip][0.25 s][Kyutai English in that voice]
  insert  [real Nepali ..word k][0.1 s][Kyutai phrase][0.1 s][real Nepali word k..]

Real audio is resampled to 24 kHz and the Kyutai part is RMS-matched to it so
the join is not a loudness step. Word times from the source manifest and from
qc_align_v5.py are shifted onto the joined timeline; training reads its target
text from `words`, so `transcript` is just their join. Run with repo/.venv:

    RAW=/workspace/v5_synth/audio/raw python3 assemble_v5.py
"""
import json, math, os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RAW = os.environ.get("RAW", "/workspace/v5_synth/audio/raw")
BASE = os.path.dirname(RAW)
QC = os.environ.get("QC_OUT", f"{BASE}/qc.jsonl")
OUT_WAV = f"{BASE}/clips"
MANIFEST = os.environ.get("MANIFEST", f"{BASE}/train_v5_synth.jsonl")
SR = 24000
os.makedirs(OUT_WAV, exist_ok=True)


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-8)


def shift(words, dt):
    return [{"word": w["word"], "start": round(w["start"] + dt, 3), "end": round(w["end"] + dt, 3)}
            for w in words]


def sil(sec):
    return np.zeros(int(sec * SR), dtype=np.float32)


jobs = {j["id"]: j for j in map(json.loads, open("/workspace/v5_synth/jobs.jsonl"))}
qc = [r for r in map(json.loads, open(QC)) if r["ok"]]
n_by = {"plain": 0, "concat": 0, "insert": 0}
with open(MANIFEST, "w") as man:
    for r in qc:
        j = jobs[r["id"]]
        en = load(f"{RAW}/{r['id']}.wav")
        enw = r["words"]
        real = j["real"]
        if j["kind"] == "plain":
            audio, words, lang = en, enw, "en"
        else:
            ra = load(real["path"])
            en = en * (rms(ra) / rms(en))
            if j["kind"] == "concat":
                gap = sil(0.25)
                audio = np.concatenate([ra, gap, en])
                words = real["words"] + shift(enw, len(ra) / SR + 0.25)
            else:
                k = j["split"]
                hw = real["words"]
                t = (hw[k - 1]["end"] + hw[k]["start"]) / 2
                a = max(0.0, enw[0]["start"] - 0.05)
                b = min(len(en) / SR, enw[-1]["end"] + 0.1)
                piece = en[int(a * SR):int(b * SR)]
                cut = int(t * SR)
                ins = 0.1 + len(piece) / SR + 0.1
                audio = np.concatenate([ra[:cut], sil(0.1), piece, sil(0.1), ra[cut:]])
                words = (hw[:k] + shift(enw, t + 0.1 - a) + shift(hw[k:], ins))
            lang = "mix"
        audio = np.clip(audio, -1.0, 1.0)
        out = f"{OUT_WAV}/{r['id']}.wav"
        sf.write(out, audio, SR)
        man.write(json.dumps({
            "path": out, "duration": round(len(audio) / SR, 3),
            "transcript": " ".join(w["word"] for w in words),
            "speaker": f"v5synth/{real['speaker']}", "source": f"v5_{j['kind']}",
            "language": lang, "style": None, "words": words,
        }, ensure_ascii=False) + "\n")
        n_by[j["kind"]] += 1
print(f"assembled {sum(n_by.values()):,} clips {n_by} -> {MANIFEST}")
