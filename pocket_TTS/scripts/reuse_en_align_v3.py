"""Seed the English alignments from the previous v3 run instead of redoing them.

keep_latin only changed the Nepali branch of build_manifest_v3.py, so every
English transcript is byte-identical to the previous build and its word timings
are still valid. seed_align_v3.py cannot know that -- it seeds from v2, which had
no English at all -- so without this the aligner would spend ~1 h recomputing
223,983 rows that did not move.

Guarded on transcript equality, not just (path, start): that is the exact key
align_data --resume trusts, and the reason 1.8% of Nepali rows had to be redone
in the first place.
"""
import json, os

M = "/root/tts/TTS_training/pocket_TTS/manifests"
OLD = f"{M}/v3_pre_codeswitch/train_v3_aligned.jsonl"

old_en = {}
for line in open(OLD):
    d = json.loads(line)
    if d.get("language") == "en" and d.get("words"):
        old_en[(d["path"], round(float(d.get("start", 0.0)), 3))] = d

kept, stale, missing = 0, 0, 0
remaining = []
with open(f"{M}/train_v3_aligned.jsonl", "a") as out:
    for line in open(f"{M}/train_v3_en.jsonl"):
        d = json.loads(line)
        k = (d["path"], round(float(d.get("start", 0.0)), 3))
        prev = old_en.get(k)
        if prev is None:
            missing += 1; remaining.append(line); continue
        if prev["transcript"] != d["transcript"]:
            stale += 1; remaining.append(line); continue
        out.write(json.dumps(prev) + "\n")
        kept += 1

with open(f"{M}/train_v3_en.jsonl", "w") as f:
    f.writelines(remaining)

print(f"reused {kept:,} English alignments from the previous v3 run")
print(f"  {stale:,} transcript changed (must realign), {missing:,} not previously aligned")
print(f"  train_v3_en.jsonl now holds {len(remaining):,} rows to align")
