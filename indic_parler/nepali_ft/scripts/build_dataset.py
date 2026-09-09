"""Build a Parler-TTS finetuning dataset from the Nepali half of ai4bharat___rasa.

Rasa ships a `style` label per clip (ANGER/HAPPY/SAD/FEAR/DISGUST/SURPRISE plus
neutral registers WIKI/CONV/NEWS/BOOK/...). Those are exactly the emotion names
the indic-parler model card lists, so we turn each into a natural-language caption
of the kind the model was conditioned on, rather than inventing a new vocabulary.

Captions deliberately carry speaker + emotion + pace + recording quality, and NOT a
measured pitch word: pitch is the thing the emotion label should be teaching, and
naming it in the caption lets the model shortcut to the pitch word at inference.

Reads audio in place from /workspace (never moves the originals) and writes parquet
shards with the audio embedded, which is what `load_dataset(<dir>)` wants.
"""
import argparse, json, os, io, random, sys
import numpy as np, soundfile as sf
import pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--rasa", default="/workspace/proc_data_new/ai4bharat___rasa")
ap.add_argument("--out", default="/workspace/milan_nepali_parler_ft/data")
ap.add_argument("--language", default="Nepali")
ap.add_argument("--min-s", type=float, default=0.6)
ap.add_argument("--max-s", type=float, default=25.0)
ap.add_argument("--rows-per-shard", type=int, default=2000)
args = ap.parse_args()

SRC_PREFIX = "/projects/data/ttsteam/proc_data_new"
DST_PREFIX = "/workspace/proc_data_new"

# Rasa style -> the emotion/register wording used in indic-parler captions
STYLE = {
    "ANGER":    ["speaks in an angry tone, forceful and tense",
                 "sounds angry and irritated, with hard emphasis",
                 "delivers the line angrily, sharp and clipped"],
    "HAPPY":    ["speaks in a happy, cheerful tone",
                 "sounds happy and pleased, warm and bright",
                 "delivers the line happily, light and upbeat"],
    "SAD":      ["speaks in a sad, sorrowful tone",
                 "sounds sad and downcast, subdued and heavy",
                 "delivers the line sadly, quiet and weary"],
    "FEAR":     ["speaks in a fearful, anxious tone",
                 "sounds frightened and uneasy, tight and hesitant",
                 "delivers the line fearfully, tense and unsteady"],
    "DISGUST":  ["speaks in a disgusted tone",
                 "sounds disgusted and repelled",
                 "delivers the line with clear distaste"],
    "SURPRISE": ["speaks in a surprised tone",
                 "sounds surprised and taken aback",
                 "delivers the line with sudden surprise"],
    "NEWS":     ["reads the line in a clear news-reading style",
                 "delivers the line like a news anchor, even and articulate",
                 "speaks in a measured news-reading register"],
    "BOOK":     ["reads the line in a calm narration style",
                 "narrates the line steadily, as if reading aloud from a book",
                 "speaks in an even narration register"],
    "WIKI":     ["reads the line in a neutral, informative style",
                 "speaks in a plain, even reading register",
                 "delivers the line neutrally and clearly"],
    "CONV":     ["speaks in a natural conversational tone",
                 "sounds relaxed and conversational",
                 "delivers the line as ordinary conversation"],
    "PROPER NOUN": ["pronounces the name clearly and distinctly",
                    "articulates the proper noun carefully",
                    "says the name with careful enunciation"],
    "ALEXA":    ["speaks in a helpful voice-assistant tone",
                 "sounds like a friendly voice assistant",
                 "delivers the line as a voice assistant would"],
    "BB":       ["speaks in a clear command-giving tone",
                 "delivers a short, direct instruction",
                 "gives the instruction plainly and directly"],
    "DIGI":     ["speaks in a clear command-giving tone",
                 "delivers a short, direct instruction",
                 "gives the instruction plainly and directly"],
    "UMANG":    ["speaks in a helpful voice-assistant tone",
                 "sounds like a friendly voice assistant",
                 "delivers the line as a voice assistant would"],
}
QUALITY = [" The recording is very high quality, with her voice sounding clear and very close up.",
           " The recording is very high quality, with his voice sounding clear and very close up."]


def pace_word(chars, dur, med):
    r = chars / max(dur, 1e-3)
    if r < med * 0.85: return " at a slow pace"
    if r > med * 1.15: return " at a fast pace"
    return " at a moderate pace"


rows = {"train": [], "validation": []}
rates = []
raw = {"train": [], "validation": []}
for split, key in (("train", "train"), ("val", "validation")):
    mf = os.path.join(args.rasa, split, "manifest.jsonl")
    for line in open(mf):
        r = json.loads(line)
        if r.get("language") != args.language:
            continue
        d = r.get("duration", 0.0)
        t = (r.get("text") or "").strip()
        if not t or not (args.min_s <= d <= args.max_s):
            continue
        st = r.get("style")
        if st not in STYLE:
            continue
        raw[key].append(r)
        rates.append(len(t) / d)
med = float(np.median(rates))
print(f"median speaking rate: {med:.2f} chars/s", flush=True)

rng = random.Random(20260905)
os.makedirs(args.out, exist_ok=True)
schema = pa.schema([
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("text", pa.string()), ("description", pa.string()),
    ("speaker", pa.string()), ("style", pa.string()),
    ("emotion", pa.string()), ("duration", pa.float32()), ("id", pa.string()),
])
EMOSET = {"ANGER", "HAPPY", "SAD", "FEAR", "DISGUST", "SURPRISE"}

for key in ("train", "validation"):
    d = os.path.join(args.out, key); os.makedirs(d, exist_ok=True)
    buf, shard, n_ok, n_bad = [], 0, 0, 0
    for r in raw[key]:
        p = r["audio_filepath"].replace(SRC_PREFIX, DST_PREFIX)
        try:
            y, sr = sf.read(p, dtype="int16")
        except Exception:
            n_bad += 1; continue
        b = io.BytesIO(); sf.write(b, y, sr, format="WAV", subtype="PCM_16")
        spk = r["speaker_id"]; st = r["style"]; t = r["text"].strip()
        v = rng.randrange(3)
        desc = (f"{spk} {STYLE[st][v]}" + pace_word(len(t), r["duration"], med) + "."
                + QUALITY[0 if r.get("gender") == "Female" else 1])
        buf.append({"audio": {"bytes": b.getvalue(), "path": os.path.basename(p)},
                    "text": t, "description": desc, "speaker": spk, "style": st,
                    "emotion": st.lower() if st in EMOSET else "neutral",
                    "duration": float(r["duration"]), "id": r.get("filename", "")})
        n_ok += 1
        if len(buf) >= args.rows_per_shard:
            pq.write_table(pa.Table.from_pylist(buf, schema=schema),
                           os.path.join(d, f"part-{shard:04d}.parquet"))
            buf = []; shard += 1
            print(f"  {key}: {n_ok} rows, {shard} shards", flush=True)
    if buf:
        pq.write_table(pa.Table.from_pylist(buf, schema=schema),
                       os.path.join(d, f"part-{shard:04d}.parquet"))
        shard += 1
    print(f"{key}: {n_ok} rows in {shard} shards ({n_bad} unreadable)", flush=True)
print("DONE ->", args.out)
