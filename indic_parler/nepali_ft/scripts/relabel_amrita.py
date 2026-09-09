"""Rewrite training captions so the speaker name is "Amrita" -- the one Nepali
voice that actually exists in indic-parler-tts.

Rationale: at inference we always say "Amrita". The gem+replay run trained
expressive control keyed to names the model has no voice for (Srijana is a Rasa
speaker, out-of-vocabulary here; Kore/Aoede/Gacrux are Gemini TTS voices). So the
caption->prosody mapping it learned was attached to tokens we never use. This is a
train/inference mismatch, and closing it is the point of this run.

Only FEMALE clips are relabelled -- mapping male audio onto a female speaker name
would teach the wrong timbre. Male clips keep their original names and act as
generic replay.

Audio bytes are copied through untouched; only the description/speaker columns change.
"""
import pyarrow.parquet as pq, pyarrow as pa, glob, os, sys, collections

SRC_DST = [("data_gem", "data_gem_a"), ("data_replay", "data_replay_a")]
NAME = "Amrita"
stats = collections.Counter()

for src, dst in SRC_DST:
    for split in sorted(os.listdir(src)):
        sd, dd = f"{src}/{split}", f"{dst}/{split}"
        os.makedirs(dd, exist_ok=True)
        for f in sorted(glob.glob(sd + "/*.parquet")):
            t = pq.read_table(f)
            rows = t.to_pylist()
            for r in rows:
                if r["gender"] != "female":
                    stats["male_kept"] += 1
                    continue
                old = r["speaker"]
                d = r["description"]
                # every builder writes the name as the leading token of the caption
                if old and d.startswith(old):
                    r["description"] = NAME + d[len(old):]
                    stats["relabelled"] += 1
                else:
                    stats["prefix_miss"] += 1
                r["speaker"] = NAME
            pq.write_table(pa.Table.from_pylist(rows, schema=t.schema),
                           os.path.join(dd, os.path.basename(f)))
        print(f"{sd} -> {dd}", flush=True)

print(dict(stats))
if stats["prefix_miss"]:
    sys.exit(f"ERROR: {stats['prefix_miss']} female captions did not start with the "
             f"speaker name; relabelling would have been silent. Inspect before training.")
print("OK")
