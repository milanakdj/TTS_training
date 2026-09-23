"""Bilingual calibration set for the v4 student EOS threshold -- DISJOINT from
pairs_2x2.json. Mirrors build_calib_v3.py but pulls both languages (v3's
valid_v3_aligned.jsonl carries language=ne|en) since v4's English EOS head is
the one observed to fire early, not Nepali's -- see
pocket-tts-v3-eos-miscalibrated and the "empty generation, skipped" flag noted
during v4 teacher training.
"""
import json, os, random, collections

R = "/root/tts/TTS_training/pocket_TTS"
N_PER_LANG = 20
key = lambda r: (os.path.dirname(r["path"]), r["speaker"])

eval_paths = {p["real_audio"] for p in json.load(open(f"{R}/infer/final/pairs_2x2.json"))}
val = [json.loads(l) for l in open(f"{R}/manifests/valid_v3_aligned.jsonl")]
# English val speakers are disjoint from English train speakers by construction
# (build_pairs_2x2.py's comment) -- same-identity lookup always misses for en,
# so mirror that script's fallback: any train clip in the target language.
prompts, any_prompt = collections.defaultdict(list), {"ne": [], "en": []}
for l in open(f"{R}/manifests/train_v3_aligned.jsonl"):
    r = json.loads(l)
    if 3.0 <= r["duration"] <= 5.0:
        lang = r.get("language", "ne")
        prompts[key(r)].append(r["path"])
        any_prompt[lang].append(r["path"])

rng = random.Random(7)
for v in any_prompt.values():
    rng.shuffle(v)
out = []
for lang in ("ne", "en"):
    cands = [r for r in val if r.get("language") == lang and 3.0 <= r["duration"] <= 12.0
             and r["path"] not in eval_paths]
    by_src = collections.defaultdict(list)
    for r in cands:
        by_src[r["source"]].append(r)
    for v in by_src.values():
        rng.shuffle(v)
    quota = max(1, N_PER_LANG // max(1, len(by_src)))
    picked = [r for src in sorted(by_src) for r in by_src[src][:quota]]
    pool = [r for src in sorted(by_src) for r in by_src[src][quota:]]
    rng.shuffle(pool)
    picked = (picked + pool)[:N_PER_LANG]
    for i, r in enumerate(sorted(picked, key=lambda x: x["path"])):
        same = prompts.get(key(r))
        prompt = rng.choice(sorted(same)) if same else any_prompt[lang][i % len(any_prompt[lang])]
        out.append({"id": f"c{lang}{len(out):03d}", "lang": lang, "source": r["source"],
                     "text": r["transcript"], "real_audio": r["path"], "duration": r["duration"],
                     "prompt": prompt})

assert not ({r["real_audio"] for r in out} & eval_paths), "calibration set leaks into pairs_2x2.json"
json.dump(out, open(f"{R}/infer/final/calib_v4.json", "w"), ensure_ascii=False, indent=1)
print(f"{len(out)} calibration utterances: "
      f"{dict(collections.Counter(r['lang'] for r in out))}")
