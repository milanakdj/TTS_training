"""Blind A/B listening set: 12 utterances, teacher vs student under A/B labels.

Blind because the only instrument that has held up in this project is the user's
ear, and it should not be told which file is the 3x cheaper model. key.json has
the mapping; real_<id>.wav is the human recording of the same sentence.
"""
import json, os, random, shutil, collections

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
OUT = f"{F}/listen"
pairs = {p["id"]: p for p in json.load(open(f"{F}/pairs.json"))}

have = [i for i in pairs
        if os.path.exists(f"{F}/wav/teacher_24l/{i}.wav")
        and os.path.exists(f"{F}/wav/student_6l/{i}.wav")]
by_src = collections.defaultdict(list)
for i in sorted(have):
    by_src[pairs[i]["source"]].append(i)
rng = random.Random(7)
pick = []
for src in sorted(by_src):
    rng.shuffle(by_src[src])
    pick += by_src[src][:2]
pick = sorted(pick)[:12]

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
key, lines = {}, []
for i in pick:
    sysmap = {"teacher_24l": None, "student_6l": None}
    labels = ["A", "B"]
    rng.shuffle(labels)
    for (name, _), lab in zip(sysmap.items(), labels):
        shutil.copy(f"{F}/wav/{name}/{i}.wav", f"{OUT}/{i}_{lab}.wav")
        sysmap[name] = lab
    if os.path.exists(pairs[i]["real_audio"]):
        shutil.copy(pairs[i]["real_audio"], f"{OUT}/{i}_real.wav")
    key[i] = {v: k for k, v in sysmap.items()}
    lines.append(f"{i}  [{pairs[i]['source']}]  {pairs[i]['text']}")

json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)
open(f"{OUT}/TEXTS.txt", "w").write(
    "Blind A/B: which of A/B sounds better? _real is the human recording.\n"
    "Mapping is in key.json -- read it after listening, not before.\n\n" + "\n".join(lines) + "\n")
print(f"{len(pick)} blind A/B triples -> {OUT}")
