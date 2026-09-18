"""Same disjointness test, against the sibling 0.6b finetune's training data.

A head-to-head is only fair if the gold set is unseen by *both* models. The 0.6b
ships its own manifests, so this is checkable rather than assumed.
"""
import json, os, re, unicodedata, collections

DL = "/workspace/gold_eval_dl/manifests"
TRAIN = [f"{DL}/ne_corpus_train.jsonl", f"{DL}/ne_train.jsonl", f"{DL}/train_mix_langid.jsonl"]

def canon(t):
    t = unicodedata.normalize("NFC", t)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip().lower()

gold = [json.loads(l) for l in open("fleurs_ne_test.raw.jsonl")]
g_text = {canon(r["text"]) for r in gold} | {canon(r["raw_text"]) for r in gold}
g_base = {os.path.basename(r["audio_filepath"]) for r in gold}
def grams(s, n=5):
    w = s.split()
    return {" ".join(w[i:i+n]) for i in range(max(0, len(w)-n+1))} if len(w) >= n else set()
g_gram = collections.defaultdict(set)
for ct in g_text:
    for gm in grams(ct): g_gram[gm].add(ct)

tot = txt = base = near = 0
fleurs_str = 0
for tf in TRAIN:
    if not os.path.exists(tf): print("MISSING", tf); continue
    for line in open(tf):
        tot += 1
        if "fleurs" in line.lower(): fleurs_str += 1
        d = json.loads(line)
        p = d.get("audio_filepath", "")
        if os.path.basename(p) in g_base: base += 1
        ct = canon(d.get("text", ""))
        if ct in g_text: txt += 1
        else:
            ov = collections.Counter()
            for gm in grams(ct):
                for t in g_gram[gm]: ov[t] += 1
            if ov and ov.most_common(1)[0][1] >= 3: near += 1

print(f"scanned {tot:,} rows of the 0.6b's training manifests")
print(f"  rows containing 'fleurs'  : {fleurs_str}")
print(f"  basename overlap          : {base}")
print(f"  canonical text overlap    : {txt}")
print(f"  near-dup (>=3 shared 5-gr): {near}")
print("VERDICT:", "DISJOINT" if not (base or txt or near or fleurs_str) else "CONTAMINATED")
json.dump({"scanned": tot, "fleurs_str": fleurs_str, "basename": base,
           "text": txt, "near": near}, open("disjoint_report_nemotron.json", "w"), indent=1)
