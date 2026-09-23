"""The bilingual 2x2 eval set: {Nepali, English} text x {Nepali, English} voice.

v3 was judged on Nepali text with Nepali voices and nothing else, so "renders
English perfectly, Nepali unintelligible" stayed invisible for 200k steps. The
CROSS cells are what decide whether a model is actually bilingual rather than two
monolingual models sharing weights -- a Nepali speaker reading an English brand
name is the product requirement.

Two ceilings, and they are NOT the same thing:

  intelligibility ceiling  always available. `real_audio` is the genuine recording
                           of the target text; its CER through the same ASR says
                           how much of the score is the instrument, not the model.

  speaker-sim ceiling      only when the prompt is the SAME identity as the target
                           (`sim_ceiling: true`). The English valid speakers are
                           disjoint from train by construction, so English and
                           cross cells get an unrelated-speaker prompt and their
                           sim numbers have a cross-speaker FLOOR, not a ceiling.
                           Do not compare a sim number across that boundary.
"""
import json, os, random, collections

R = "/root/tts/TTS_training/pocket_TTS"
PER_CELL = 50
key = lambda r: (os.path.dirname(r["path"]), r.get("speaker"))
is_en = lambda r: r.get("source", "").startswith("en-in")

same_id, any_prompt = collections.defaultdict(list), {"ne": [], "en": []}
for l in open(f"{R}/manifests/train_v3_aligned.jsonl"):
    r = json.loads(l)
    if not (3.0 <= r["duration"] <= 5.0):
        continue
    lang = "en" if is_en(r) else "ne"
    same_id[(lang, key(r))].append(r["path"])
    any_prompt[lang].append(r["path"])

val = [json.loads(l) for l in open(f"{R}/manifests/valid_v3_aligned.jsonl")]
targets = {"ne": [], "en": []}
for r in val:
    if 3.0 <= r["duration"] <= 10.0:
        targets["en" if is_en(r) else "ne"].append(r)

rng = random.Random(11)
for v in targets.values():
    rng.shuffle(v)
for v in any_prompt.values():
    rng.shuffle(v)
print("targets available:", {k: len(v) for k, v in targets.items()})

items = []
for t_lang in ("ne", "en"):
    for v_lang in ("ne", "en"):
        for i, r in enumerate(targets[t_lang][:PER_CELL]):
            pool = same_id.get((v_lang, key(r))) if t_lang == v_lang else None
            items.append({
                "id": f"{t_lang}{v_lang}_{i:03d}",
                "cell": f"{t_lang}_text/{v_lang}_voice",
                "text_lang": t_lang, "voice_lang": v_lang,
                "source": r["source"], "text": r["transcript"],
                "real_audio": r["path"],              # intelligibility ceiling
                "duration": r["duration"],
                "prompt": rng.choice(sorted(pool)) if pool else any_prompt[v_lang][i],
                "sim_ceiling": bool(pool),            # is the prompt the same identity?
            })

json.dump(items, open(f"{R}/infer/final/pairs_2x2.json", "w"), ensure_ascii=False, indent=1)
print(f"wrote {len(items)} items -> infer/final/pairs_2x2.json")
for cell in sorted({i["cell"] for i in items}):
    g = [i for i in items if i["cell"] == cell]
    print(f"  {cell:<22} n={len(g):>3}  sim_ceiling={sum(i['sim_ceiling'] for i in g)}/{len(g)}")
