"""Build the held-out eval set: (voice prompt, target transcript, real human audio).

The valid split is nearly one-clip-per-identity -- only 3 of its 941 speaker
identities have two clips long enough to pair -- so the prompt cannot come from
the valid split. It comes from the TRAIN split instead, from the same identity.
That is sound: the prompt is conditioning, not a target. The utterance being
synthesized (its text, and the real human audio used as the ceiling) is still
strictly held out.

Identity key is (directory, speaker). The `speaker` field alone is a per-video
diarization label -- 'SPEAKER_01' in two different YouTube videos is two
different people -- so keying on it alone would pair unrelated voices.
"""
import json, os, random, collections

R = "/root/tts/TTS_training/pocket_TTS"
N_TOTAL = 100
key = lambda r: (os.path.dirname(r["path"]), r["speaker"])

val = [json.loads(l) for l in open(f"{R}/manifests/valid_v2_aligned.jsonl")]
prompts = collections.defaultdict(list)
for l in open(f"{R}/manifests/train_v2_aligned.jsonl"):
    r = json.loads(l)
    if 3.0 <= r["duration"] <= 5.0:   # max_voice_prompt_sec is 5.0 in training
        prompts[key(r)].append(r["path"])

cands = [r for r in val if 3.0 <= r["duration"] <= 10.0 and prompts.get(key(r))]
by_src = collections.defaultdict(list)
for r in cands:
    by_src[r["source"]].append(r)

# Stratify: even quota per source, then top up from the largest sources. Without
# this the two `ans` YouTube sources would swamp the studio-quality ones.
rng = random.Random(42)
for v in by_src.values():
    rng.shuffle(v)
quota = max(1, N_TOTAL // len(by_src))
picked = []
for src in sorted(by_src):
    picked += by_src[src][:quota]
pool = [r for src in sorted(by_src) for r in by_src[src][quota:]]
rng.shuffle(pool)
picked += pool[: max(0, N_TOTAL - len(picked))]
picked = picked[:N_TOTAL]

pairs = []
for i, r in enumerate(sorted(picked, key=lambda x: x["path"])):
    pool_p = sorted(prompts[key(r)])
    pairs.append({
        "id": f"{i:03d}",
        "source": r["source"],
        "speaker": f"{os.path.basename(key(r)[0])}/{r['speaker']}",
        "text": r["transcript"],
        "real_audio": r["path"],          # held-out human audio = the ceiling
        "duration": r["duration"],
        "prompt": rng.choice(pool_p),     # same identity, from train
    })

os.makedirs(f"{R}/infer/final", exist_ok=True)
json.dump(pairs, open(f"{R}/infer/final/pairs.json", "w"), ensure_ascii=False, indent=1)
print(f"{len(pairs)} pairs, {len({p['speaker'] for p in pairs})} identities, "
      f"by source: {dict(collections.Counter(p['source'] for p in pairs))}")
