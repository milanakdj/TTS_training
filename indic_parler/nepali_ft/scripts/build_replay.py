"""Build the REPLAY dataset the three failed finetunes were missing.

The official Parler expressive recipe (parler-tts-mini-expresso) mixes Jenny TTS +
LibriTTS-R into the expressive finetune "to preserve generic voice capabilities".
I trained on Nepali-only expressive data with no replay all three times, which is
textbook catastrophic forgetting -- and is why the user heard "very bad quality".

Replay here = Rasa Nepali NEUTRAL registers (WIKI/CONV/NEWS/BOOK/PROPER NOUN/ALEXA/
BB/DIGI/UMANG). Deliberately chosen because indic-parler was ALREADY trained on Rasa,
so this is the model's own prior distribution -- exactly what replay should remind it
of. The expressive half comes from data_gem (native Gemini Nepali), which is new.
"""
import argparse, io, json, os, random
import numpy as np, soundfile as sf
import pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/workspace/milan_nepali_parler_ft/data_replay")
ap.add_argument("--hours", type=float, default=14.0)
args = ap.parse_args()
R = "/workspace/proc_data_new/ai4bharat___rasa"
SRC, DST = "/projects/data/ttsteam/proc_data_new", "/workspace/proc_data_new"
EMO = {"ANGER", "HAPPY", "SAD", "FEAR", "DISGUST", "SURPRISE"}
STYLE = {"WIKI": "reads the line in a neutral, informative style",
         "CONV": "speaks in a natural conversational tone",
         "NEWS": "reads the line in a clear news-reading style",
         "BOOK": "reads the line in a calm narration style",
         "PROPER NOUN": "pronounces the name clearly and distinctly",
         "ALEXA": "speaks in a helpful voice-assistant tone",
         "UMANG": "speaks in a helpful voice-assistant tone",
         "BB": "speaks in a clear command-giving tone",
         "DIGI": "speaks in a clear command-giving tone"}

rows = []
for split in ("train", "val"):
    for line in open(f"{R}/{split}/manifest.jsonl"):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("language") != "Nepali" or r.get("style") in EMO: continue
        if r.get("style") not in STYLE: continue
        d = float(r.get("duration") or 0)
        if not (0.8 <= d <= 20.0) or not (r.get("text") or "").strip(): continue
        rows.append(r)
rng = random.Random(20260906); rng.shuffle(rows)

schema = pa.schema([("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                    ("text", pa.string()), ("description", pa.string()), ("speaker", pa.string()),
                    ("role", pa.string()), ("gender", pa.string()),
                    ("duration", pa.float32()), ("id", pa.string())])
QF = " The recording is very high quality, with her voice sounding clear and very close up."
QM = " The recording is very high quality, with his voice sounding clear and very close up."

budget, acc, buf, shard, n = args.hours * 3600, 0.0, [], 0, 0
d_out = os.path.join(args.out, "train"); os.makedirs(d_out, exist_ok=True)
for r in rows:
    if acc >= budget: break
    p = r["audio_filepath"].replace(SRC, DST)
    try: y, sr = sf.read(p, dtype="int16")
    except Exception: continue
    b = io.BytesIO(); sf.write(b, y, sr, format="WAV", subtype="PCM_16")
    spk = r["speaker_id"]
    desc = (f"{spk} {STYLE[r['style']]} at a moderate pace."
            + (QF if r.get("gender") == "Female" else QM))
    buf.append({"audio": {"bytes": b.getvalue(), "path": os.path.basename(p)},
                "text": r["text"].strip(), "description": desc, "speaker": spk,
                "role": "", "gender": (r.get("gender") or "").lower(),
                "duration": float(r["duration"]), "id": r.get("filename", "")})
    acc += float(r["duration"]); n += 1
    if len(buf) >= 1000:
        pq.write_table(pa.Table.from_pylist(buf, schema=schema),
                       os.path.join(d_out, f"part-{shard:04d}.parquet")); buf = []; shard += 1
if buf:
    pq.write_table(pa.Table.from_pylist(buf, schema=schema),
                   os.path.join(d_out, f"part-{shard:04d}.parquet"))
print(f"replay: {n} clips, {acc/3600:.2f} h -> {args.out}")
