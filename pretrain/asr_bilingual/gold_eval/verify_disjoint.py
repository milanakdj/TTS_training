"""Prove the FLEURS gold set is absent from run_v7's training data.

Four independent checks, because "it wasn't one of the 12 sources" is exactly the
kind of assumption that produced hum_heldout. Path identity is the weakest check
(different roots make it vacuous); text identity is the one that matters, since a
re-scrape of the same sentences under a different filename is the realistic way
contamination enters.
"""
import hashlib, json, os, re, sys, unicodedata, collections

MAN = "/workspace/asr_pretrain_v2/manifests"
TRAIN = [f"{MAN}/train_ne_human.jsonl", f"{MAN}/train_ne_pseudo.jsonl",
         f"{MAN}/train_ne_script.jsonl", f"{MAN}/train_en_pseudo.jsonl"]
GOLD = "fleurs_ne_test.raw.jsonl"

def canon(t):
    """Aggressive canonical form: NFC, strip all non-word chars, collapse space.
    Deliberately looser than the training normalizer so near-misses still collide."""
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()

gold = [json.loads(l) for l in open(GOLD)]
g_paths = {os.path.realpath(r["audio_filepath"]) for r in gold}
g_text = {canon(r["text"]): r for r in gold}
g_raw = {canon(r["raw_text"]): r for r in gold}
g_dur = collections.defaultdict(list)
for r in gold: g_dur[round(r["duration"], 1)].append(r)
print(f"gold: {len(gold)} clips, {len(g_text)} distinct canonical texts, "
      f"{round(sum(r['duration'] for r in gold)/3600,3)} h\n")

# 5-gram index over gold, for near-duplicate detection
def grams(s, n=5):
    w = s.split()
    return {" ".join(w[i:i+n]) for i in range(max(0, len(w)-n+1))} if len(w) >= n else set()
g_gram = collections.defaultdict(set)
for ct in g_text:
    for gm in grams(ct): g_gram[gm].add(ct)

hits = {"path": [], "text": [], "raw": [], "near": []}
dur_cand = 0
n = 0
for tf in TRAIN:
    for line in open(tf):
        d = json.loads(line); n += 1
        if os.path.realpath(d["audio_filepath"]) in g_paths: hits["path"].append(d)
        ct = canon(d["text"])
        if ct in g_text: hits["text"].append((d, g_text[ct]))
        elif ct in g_raw: hits["raw"].append((d, g_raw[ct]))
        else:
            ov = collections.Counter()
            for gm in grams(ct):
                for t in g_gram[gm]: ov[t] += 1
            if ov:
                t, c = ov.most_common(1)[0]
                if c >= 3: hits["near"].append((d, t, c))
        if round(d.get("duration", -1), 1) in g_dur: dur_cand += 1

print(f"scanned {n:,} training rows across {len(TRAIN)} manifests\n")
print(f"  exact path overlap        : {len(hits['path'])}")
print(f"  canonical text overlap    : {len(hits['text'])}")
print(f"  raw-transcript overlap    : {len(hits['raw'])}")
print(f"  near-dup (>=3 shared 5-gr): {len(hits['near'])}")
print(f"  rows sharing a duration   : {dur_cand:,}  (expected; duration alone is not evidence)")
for k in ("text", "raw", "near"):
    for item in hits[k][:5]:
        print(f"\n  !! {k}: {json.dumps(item[0], ensure_ascii=False)[:220]}")
json.dump({k: len(v) for k, v in hits.items()}, open("disjoint_report.json", "w"), indent=1)
print("\nVERDICT:", "DISJOINT" if not any(hits[k] for k in ("path","text","raw","near")) else "CONTAMINATED")
