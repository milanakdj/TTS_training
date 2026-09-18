"""Disjointness test for the FLEURS gold set against BOTH models' training data.

v1 used ">=3 shared 5-grams" as the near-duplicate signal. On a 1.5M-row Nepali
news corpus that fires on 4.8% of rows -- it is detecting ordinary phrase reuse,
not duplication, and a test that cries wolf on 72k rows tells you nothing. This
version reports Jaccard similarity over 5-gram sets, which is scale-free, and
flags only >=0.6. Exact canonical-text identity remains the check that decides
the verdict; the rest is there to catch a re-scrape under a different filename.
"""
import json, os, re, unicodedata, collections


def canon(t):
    t = unicodedata.normalize("NFC", t)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip().lower()


def grams(s, n=5):
    w = s.split()
    if len(w) < n:
        return frozenset()
    return frozenset(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))


gold = [json.loads(l) for l in open("fleurs_ne_test.raw.jsonl")]
g_text = {canon(r["text"]) for r in gold} | {canon(r["raw_text"]) for r in gold}
g_base = {os.path.basename(r["audio_filepath"]) for r in gold}
g_gramsets = {ct: grams(ct) for ct in g_text if grams(ct)}
inv = collections.defaultdict(set)
for ct, gs in g_gramsets.items():
    for gm in gs:
        inv[gm].add(ct)

CORPORA = {
    "run_v7 (parakeet-110m)": [
        "/workspace/asr_pretrain_v2/manifests/train_ne_human.jsonl",
        "/workspace/asr_pretrain_v2/manifests/train_ne_pseudo.jsonl",
        "/workspace/asr_pretrain_v2/manifests/train_ne_script.jsonl",
        "/workspace/asr_pretrain_v2/manifests/train_en_pseudo.jsonl"],
    "nemotron-0.6b": [
        "/workspace/gold_eval_dl/manifests/ne_corpus_train.jsonl",
        "/workspace/gold_eval_dl/manifests/ne_train.jsonl",
        "/workspace/gold_eval_dl/manifests/train_mix_langid.jsonl"],
}

report = {}
for name, files in CORPORA.items():
    tot = exact = base = near60 = 0
    best, best_ex = 0.0, None
    for tf in files:
        if not os.path.exists(tf):
            print("  MISSING", tf, flush=True)
            continue
        for line in open(tf):
            tot += 1
            d = json.loads(line)
            if os.path.basename(d.get("audio_filepath", "")) in g_base:
                base += 1
            ct = canon(d.get("text", ""))
            if ct in g_text:
                exact += 1
                continue
            tg = grams(ct)
            if not tg:
                continue
            cand = collections.Counter()
            for gm in tg:
                for t in inv.get(gm, ()):
                    cand[t] += 1
            if not cand:
                continue
            t, sh = cand.most_common(1)[0]
            j = sh / len(tg | g_gramsets[t])
            if j >= 0.6:
                near60 += 1
            if j > best:
                best, best_ex = j, (ct[:90], t[:90])
    v = "DISJOINT" if not (exact or base or near60) else "CONTAMINATED"
    report[name] = {"rows": tot, "exact_text": exact, "basename": base,
                    "near_jaccard_ge_0.6": near60, "max_jaccard": round(best, 4),
                    "verdict": v}
    print(f"\n{name}", flush=True)
    print(f"  rows scanned            : {tot:,}")
    print(f"  exact canonical text    : {exact}")
    print(f"  basename collision      : {base}")
    print(f"  near-dup (Jaccard>=0.6) : {near60}")
    print(f"  max Jaccard seen        : {best:.4f}")
    if best_ex:
        print(f"    closest train : {best_ex[0]}\n    closest gold  : {best_ex[1]}")
    print(f"  VERDICT: {v}", flush=True)

json.dump(report, open("disjoint_report.json", "w"), indent=1, ensure_ascii=False)
print("\nwrote disjoint_report.json", flush=True)
