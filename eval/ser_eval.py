"""Score TTS output with a speech-emotion-recognition model, not with F0/MFCC spread.

MFCC-centroid / F0 separation only measures whether clips differ ACOUSTICALLY.
"louder and higher-pitched" scores as separated while sounding nothing like anger.
This asks a model trained on emotional speech what emotion it actually hears.

Layout expected:  <root>/<emotion>/*.wav|flac   (emotion dir name = intended label)

Reports per-intended-emotion mean probability of each SER class, the confusion
matrix, and top-1 accuracy. Accuracy is the number that tracks "does it sound angry".
"""
import argparse, glob, os, sys, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--model", default="iic/emotion2vec_plus_large")
ap.add_argument("--max-per", type=int, default=64)
ap.add_argument("--json-out", default=None)
args = ap.parse_args()

# funasr is noisy on import/inference; keep stdout readable
import logging
logging.disable(logging.INFO)
from funasr import AutoModel

CANON = {"生气/angry": "angry", "厌恶/disgusted": "disgusted", "恐惧/fearful": "fearful",
         "开心/happy": "happy", "中立/neutral": "neutral", "其他/other": "other",
         "难过/sad": "sad", "吃惊/surprised": "surprised", "<unk>": "unk"}

model = AutoModel(model=args.model, hub="ms", device="cuda:0", disable_update=True)

# Directory names must match emotion2vec's own class names or top-1 silently
# returns nan and the row is never scored -- "fear"/"surprise" did exactly that.
ALIAS = {"fear": "fearful", "surprise": "surprised", "disgust": "disgusted",
         "anger": "angry", "sadness": "sad", "happiness": "happy"}
emos = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
res, order = {}, None
for e in emos:
    files = sorted(glob.glob(os.path.join(args.root, e, "*.wav")) +
                   glob.glob(os.path.join(args.root, e, "*.flac")))[:args.max_per]
    if not files:
        continue
    probs = []
    for f in files:
        r = model.generate(f, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        order = [CANON.get(l, l) for l in r["labels"]]
        probs.append(np.asarray(r["scores"], dtype=float))
    res[e] = (np.stack(probs), files)

if not res:
    sys.exit(f"no audio under {args.root}/<emotion>/")

keep = [i for i, c in enumerate(order) if c not in ("other", "unk")]
cls = [order[i] for i in keep]

print(f"\nSER model: {args.model}")
print(f"root     : {args.root}\n")
w = max(len(c) for c in cls) + 1
print("intended    n  " + "".join(f"{c[:8]:>10s}" for c in cls) + "   top1")
rows = {}
for e, (P, files) in res.items():
    m = P[:, keep].mean(0)
    tgt = ALIAS.get(e, e)
    top1 = np.mean([cls[int(np.argmax(p[keep]))] == tgt for p in P]) if tgt in cls else float("nan")
    rows[e] = (m, top1, len(files))
    print(f"{e:10s} {len(files):3d}  " + "".join(f"{v:10.3f}" for v in m) + f"   {top1*100:5.1f}%")

acc = [r[1] for e, r in rows.items() if not np.isnan(r[1])]
if acc:
    print(f"\nmean top-1 accuracy over intended emotions: {np.mean(acc)*100:.1f}%"
          f"   (chance = {100/len(cls):.1f}%)")
print("\nreading: a row should peak on its OWN column. A row that peaks on 'neutral'\n"
      "means the SER model does not hear that emotion at all.")

if args.json_out:
    with open(args.json_out, "w") as fh:
        json.dump({e: {"n": int(r[2]), "top1": float(r[1]),
                       "mean_probs": dict(zip(cls, [float(x) for x in r[0]]))}
                   for e, r in rows.items()}, fh, indent=2)
    print(f"\nwrote {args.json_out}")
