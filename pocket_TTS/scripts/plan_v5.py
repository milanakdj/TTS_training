"""Plan the v5 synthetic set: pair each Haiku-written text with a voice.

v4's English replay (en-in YouTube) pulled the model away from Kyutai's own
English: v4 student CER 0.356 / sim 0.71 vs untouched Kyutai 6L 0.009 / 0.85 on
the 2x2 (memory: pocket-tts-v4-english-is-forgetting). v5 replaces it with
English that untouched Kyutai 6L speaks in voices cloned from our corpus.

Training cuts each clip at a word boundary inside its first 5 s and uses the
head as the voice prompt (dataloader/loader.py), so a clip that STARTS with a
real Nepali speaker teaches "this Nepali voice, now speaking English". Three
job kinds, all voiced by Kyutai from the same real clip it is joined to:

  plain   Kyutai English alone (prompt: a 3-5 s real clip, 80% ne / 20% en)
  concat  [real Nepali clip 3-5 s][Kyutai English sentence, same voice]
  insert  [real Nepali words ..k][Kyutai English phrase][real Nepali words k..]

Writes /workspace/v5_synth/jobs.jsonl. Deterministic (seeded).
"""
import collections, glob, json, os, random

R = "/root/tts/TTS_training/pocket_TTS"
OUT = "/workspace/v5_synth"
USES_PER_SPEAKER = 5
rng = random.Random(5)
key = lambda r: (os.path.dirname(r["path"]), r.get("speaker"))


def texts(pattern):
    seen, out = set(), []
    for f in sorted(glob.glob(pattern)):
        rows = []
        for line in open(f):
            try:
                rows.append(json.loads(line)["text"].strip())
            except (json.JSONDecodeError, KeyError):
                continue
        rows = [t for t in rows if t]
        # Some Haiku agents scripted word-list templates instead of writing: those
        # files are ~10% unique with broken grammar. Real writing is >=78% unique (dedup handles the rest).
        uniq = len({t.lower() for t in rows}) / max(len(rows), 1)
        if uniq < 0.7:
            print(f"  REJECT {os.path.basename(f)}: {uniq:.0%} unique (templated)")
            continue
        for t in rows:
            if 1 <= len(t.split()) <= 30 and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
    rng.shuffle(out)
    return out


def aligned(r):
    return r.get("words") and all(w.get("start") is not None and w.get("end") is not None
                                  for w in r["words"])


sentences = texts(f"{OUT}/text/parts/[ab]*.jsonl")
phrases = texts(f"{OUT}/text/parts/p*.jsonl")

prompt_ne, prompt_en, host_ne = collections.defaultdict(list), collections.defaultdict(list), []
for line in open(f"{R}/manifests/train_v3_aligned.jsonl"):
    r = json.loads(line)
    if not aligned(r):
        continue
    en = r.get("language") == "en"
    if 3.0 <= r["duration"] <= 5.0:
        (prompt_en if en else prompt_ne)[key(r)].append(r)
    elif not en and 4.0 <= r["duration"] <= 10.0 and len(r["words"]) >= 5:
        host_ne.append(r)


def speaker_cycle(pool):
    keys = sorted(pool)
    rng.shuffle(keys)
    while True:
        for k in keys:
            for _ in range(USES_PER_SPEAKER):
                yield rng.choice(pool[k])


ne_voice, en_voice = speaker_cycle(prompt_ne), speaker_cycle(prompt_en)
rng.shuffle(host_ne)

jobs = []
for i, t in enumerate(sentences):
    kind = "plain" if i % 2 == 0 else "concat"
    src = next(en_voice) if kind == "plain" and rng.random() < 0.2 else next(ne_voice)
    jobs.append({"id": f"{kind[:2]}{i:06d}", "kind": kind, "text": t, "real": src})
for i, t in enumerate(phrases):
    host = host_ne[i]
    # word boundary strictly inside the utterance, never before the first 2 words
    k = rng.randint(2, len(host["words"]) - 1)
    jobs.append({"id": f"in{i:06d}", "kind": "insert", "text": t, "real": host, "split": k})

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/jobs.jsonl", "w") as f:
    for j in jobs:
        f.write(json.dumps(j, ensure_ascii=False) + "\n")
c = collections.Counter(j["kind"] for j in jobs)
print(f"{len(sentences):,} sentences, {len(phrases):,} phrases -> {len(jobs):,} jobs {dict(c)}")
print(f"voices: {len(prompt_ne):,} ne speakers, {len(prompt_en):,} en speakers, "
      f"{len(host_ne):,} ne host clips")
