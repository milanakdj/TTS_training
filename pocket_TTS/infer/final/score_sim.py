"""Speaker similarity: cosine(synthesized, voice prompt), with the human ceiling.

The ceiling row is cosine(real held-out human clip, prompt) -- two genuine
recordings of the same speaker. That is the highest score any model could earn
here, and it is well below 1.0, so raw similarity numbers are only readable
against it.
"""
import json, os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
pairs = json.load(open(f"{F}/pairs.json"))
_lim = int(os.environ.get("EVAL_LIMIT", "0"))
if _lim:
    pairs = pairs[:_lim]
enc = VoiceEncoder(device="cuda", verbose=False)
cache = {}


def emb(path):
    if path not in cache:
        try:
            cache[path] = enc.embed_utterance(preprocess_wav(path)).astype(np.float32)
        except Exception:
            cache[path] = None
    return cache[path]


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
        a, b = emb(path), emb(p["prompt"])
        if a is None or b is None:
            continue
        sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        out.append({"id": p["id"], "source": p["source"], "sim": sim})
    rows[name] = out
    if out:
        print(f"== {name}: n={len(out)} SIM {sum(r['sim'] for r in out)/len(out):.3f}", flush=True)
json.dump(rows, open(f"{F}/sim.json", "w"), ensure_ascii=False, indent=1)
print("SIM DONE")
