"""Build pocket-tts manifests for Nepali from the corpora already on /workspace.

pocket-tts wants one jsonl line per utterance:
    {"path": ..., "duration": ..., "transcript": ...}
plus optional "words": [{"word","start","end"}] from forced alignment, which the
loader uses to cut utterances at word boundaries and to place EOS consistently.

Audio is referenced IN PLACE -- nothing is copied or moved.

Sources (both already 24 kHz, which is Mimi's rate):
  indicvoices-r  168 h / 908 speakers  restored-for-TTS spontaneous speech
  ai4bharat_rasa  55 h /   2 speakers  studio, carries the 6 emotion registers
Raw IndicVoices is deliberately excluded: it is the same underlying audio as
indicvoices-r but unrestored, so it would duplicate content at worse quality --
and the README is explicit that the model mimics the acoustic quality it is fed.
"""
import argparse, json, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/root/tts/TTS_training/pocket_TTS/manifests")
ap.add_argument("--val-frac", type=float, default=0.005)
ap.add_argument("--min-s", type=float, default=1.0)
ap.add_argument("--max-s", type=float, default=30.0)
args = ap.parse_args()

SRC, DST = "/projects/data/ttsteam/proc_data_new", "/workspace/proc_data_new"
SOURCES = [
    ("indicvoices-r", "/workspace/proc_data_new/indicvoices-r/train/manifest.jsonl"),
    ("rasa",          "/workspace/proc_data_new/ai4bharat___rasa/train/manifest.jsonl"),
    ("rasa",          "/workspace/proc_data_new/ai4bharat___rasa/val/manifest.jsonl"),
]

rows, seen, stats = [], set(), {}
for tag, mf in SOURCES:
    if not os.path.exists(mf):
        print("MISSING", mf); continue
    n = 0; hrs = 0.0
    for line in open(mf):
        try: r = json.loads(line)
        except Exception: continue
        if str(r.get("language", "")).lower() not in ("ne", "nepali"): continue
        d = float(r.get("duration") or 0)
        t = (r.get("text") or "").strip()
        if not t or not (args.min_s <= d <= args.max_s): continue
        p = (r.get("audio_filepath") or "").replace(SRC, DST)
        if not p or p in seen: continue
        seen.add(p)
        rows.append({"path": p, "duration": round(d, 3), "transcript": t,
                     "speaker": r.get("speaker_id"), "source": tag,
                     "style": r.get("style"), "gender": r.get("gender")})
        n += 1; hrs += d
    stats[tag] = stats.get(tag, [0, 0.0])
    stats[tag][0] += n; stats[tag][1] += hrs
    print(f"{tag}: +{n} clips, +{hrs/3600:.1f} h  <- {mf}", flush=True)

rng = random.Random(20260905); rng.shuffle(rows)
nval = max(200, int(len(rows) * args.val_frac))
val, train = rows[:nval], rows[nval:]
os.makedirs(args.out, exist_ok=True)
for name, part in (("train", train), ("valid", val)):
    fp = os.path.join(args.out, f"{name}.jsonl")
    with open(fp, "w") as f:
        for r in part: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(part)} rows, {sum(x['duration'] for x in part)/3600:.1f} h -> {fp}")
print("\nby source:", {k: (v[0], round(v[1]/3600, 1)) for k, v in stats.items()})
spk = len({r["speaker"] for r in rows})
print(f"total: {len(rows)} clips, {sum(r['duration'] for r in rows)/3600:.1f} h, {spk} speakers")
