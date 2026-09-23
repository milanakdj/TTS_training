"""Calibration set for the v3 EOS threshold -- DISJOINT from pairs.json.

The v3 student inherits a frozen out_eos head from the teacher, but its
backbone regressed onto the teacher's cfg-2.0-combined activations, which live
at a different scale. The head is therefore miscalibrated against the student's
own z and fires within a few frames at the shipped default (-4.0).

eos_threshold is a documented inference knob, so recalibrating it is legitimate
-- but picking it on the 100 eval utterances would be fitting the test set.
These 40 utterances come from the same valid_v3 split with the eval ids removed.
"""
import json, os, random, collections

R = "/root/tts/TTS_training/pocket_TTS"
N = 40
key = lambda r: (os.path.dirname(r["path"]), r["speaker"])

eval_paths = {p["real_audio"] for p in json.load(open(f"{R}/infer/final/pairs.json"))}
val = [json.loads(l) for l in open(f"{R}/manifests/valid_v3_aligned.jsonl")]
prompts = collections.defaultdict(list)
for l in open(f"{R}/manifests/train_v3_aligned.jsonl"):
    r = json.loads(l)
    if 3.0 <= r["duration"] <= 5.0:
        prompts[key(r)].append(r["path"])

cands = [r for r in val if 3.0 <= r["duration"] <= 10.0
         and r["path"] not in eval_paths and prompts.get(key(r))]
by_src = collections.defaultdict(list)
for r in cands:
    by_src[r["source"]].append(r)
rng = random.Random(7)
for v in by_src.values():
    rng.shuffle(v)
quota = max(1, N // len(by_src))
picked = [r for src in sorted(by_src) for r in by_src[src][:quota]]
pool = [r for src in sorted(by_src) for r in by_src[src][quota:]]
rng.shuffle(pool)
picked = (picked + pool)[:N]

out = [{"id": f"c{i:03d}", "source": r["source"], "text": r["transcript"],
        "real_audio": r["path"], "duration": r["duration"],
        "prompt": rng.choice(sorted(prompts[key(r)]))}
       for i, r in enumerate(sorted(picked, key=lambda x: x["path"]))]
assert not ({r["real_audio"] for r in out} & eval_paths), "calibration set leaks into pairs.json"
json.dump(out, open(f"{R}/infer/final/calib_v3.json", "w"), ensure_ascii=False, indent=1)
print(f"{len(out)} calibration utterances, sources: "
      f"{dict(collections.Counter(r['source'] for r in out))}")
print(f"mean reference duration {sum(r['duration'] for r in out)/len(out):.2f}s")
