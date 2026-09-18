"""Abort-before-launch checks. Every one of these is a failure we already paid for.

1. Language tags vs the vocab/prompt config -- "a bad key surfaces thousands of
   steps in, not at startup" (bare `ne` is absent from the prompt dictionary and
   killed a run ~2k steps in).
2. Tokenizer sanity (Step 3): nothing may tokenise to zero tokens, and the UNK
   rate must be ~0. Devanagari digits and ZWJ/ZWNJ currently become trainable
   <unk> targets if the normalisation is wrong.
3. Audio paths actually resolve. The manifests on the mount carry
   /projects/data/ttsteam/... which does not exist on this box.
4. Duration histogram vs max_duration -- so "we trained on 900 h of OOV audio"
   cannot silently mean "lhotse dropped all of it for being 39 s long".
5. English share of hours, against the postmortem's >=30% rule.
"""
import argparse, json, os, random, collections, sys
import sentencepiece as spm

ap = argparse.ArgumentParser()
ap.add_argument("--manifests", default="/workspace/asr_pretrain_v2/manifests")
ap.add_argument("--tokenizer",
                default="/root/tts/TTS_training/pretrain/asr_bilingual/ne_en_bpe.model")
ap.add_argument("--max-duration", type=float, default=60.0)
ap.add_argument("--min-en-frac", type=float, default=0.30)
ap.add_argument("--sample-paths", type=int, default=500)
ap.add_argument("--max-unk-rate", type=float, default=0.001)
a = ap.parse_args()

fails, warns = [], []

def check(ok, msg):
    (print if ok else print)(("  ok   " if ok else "  FAIL ") + msg)
    if not ok: fails.append(msg)

def warn(ok, msg):
    print(("  ok   " if ok else "  WARN ") + msg)
    if not ok: warns.append(msg)

train = os.path.join(a.manifests, "train_mix.jsonl")
rows = [json.loads(l) for l in open(train, encoding="utf-8")]
print(f"\n[preflight] {train}: {len(rows)} rows")

# 1 ---------------------------------------------------------------- lang tags
langs = collections.Counter(r.get("lang") for r in rows)
print(f"\n[1] language tags: {dict(langs)}")
check(set(langs) <= {"ne-NP", "en-US"},
      f"only known prompt keys present (got {sorted(set(langs))})")
check(None not in langs, "every row carries a lang field")

# 2 ---------------------------------------------------------------- tokenizer
print("\n[2] tokenizer")
sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
unk = sp.unk_id()
smp = random.Random(0).sample(rows, min(4000, len(rows)))
zero, ntok, nunk, nword = [], 0, 0, 0
for r in smp:
    ids = sp.encode(r["text"])
    if not ids: zero.append(r["audio_filepath"])
    ntok += len(ids); nunk += sum(1 for i in ids if i == unk)
    nword += len(r["text"].split())
check(not zero, f"no text tokenises to zero tokens ({len(zero)} bad)")
rate = nunk / max(ntok, 1)
check(rate <= a.max_unk_rate, f"UNK rate {rate:.5f} <= {a.max_unk_rate}")
print(f"       vocab {sp.get_piece_size()}, {ntok/max(nword,1):.2f} tok/word "
      f"over {len(smp)} sampled rows")

# 3 ---------------------------------------------------------------- paths
print("\n[3] audio paths")
smp = random.Random(1).sample(rows, min(a.sample_paths, len(rows)))
missing = [r["audio_filepath"] for r in smp if not os.path.exists(r["audio_filepath"])]
check(not missing, f"{len(smp)} sampled paths resolve ({len(missing)} missing)")
for m in missing[:3]: print("        e.g. " + m)

# 4 ---------------------------------------------------------------- durations
print(f"\n[4] durations vs max_duration={a.max_duration}s")
by = collections.defaultdict(lambda: [0.0, 0.0])   # source -> [total_h, kept_h]
for r in rows:
    b = by[r["source"]]; b[0] += r["duration"]
    if r["duration"] <= a.max_duration: b[1] += r["duration"]
for k, (t, kp) in sorted(by.items(), key=lambda x: -x[1][0]):
    frac = kp / t if t else 0
    line = f"{k:22s} {t/3600:8.1f} h -> {kp/3600:8.1f} h kept ({100*frac:5.1f}%)"
    if frac < 0.5: warn(False, line)
    else: print("  ok   " + line)
kept = sum(v[1] for v in by.values()) / 3600
print(f"       total kept at this cap: {kept:.1f} h")

# 4b --------------------------------------------------------------- T/U
print("\n[4b] CTC alignability (encoder frames vs target tokens)")
FPS = 12.5     # 10 ms stride, 8x subsampling
bad = collections.Counter(); tot = collections.Counter()
smp2 = random.Random(2).sample(rows, min(30000, len(rows)))
for r in smp2:
    u = max(len(sp.encode(r["text"])), 1)
    tot[r["source"]] += 1
    if r["duration"] * FPS < u: bad[r["source"]] += 1
worst = 0.0
for k in sorted(tot):
    frac = bad[k] / tot[k]
    worst = max(worst, frac)
    if frac > 0.01:
        print(f"  FAIL {k:22s} {100*frac:5.1f}% of rows have T < U")
check(worst <= 0.01,
      f"every source keeps T >= U (worst {100*worst:.1f}%) -- CTC cannot align a "
      f"target longer than the frames, and zero_infinity hides it")

# 5 ---------------------------------------------------------------- balance
print("\n[5] language balance (postmortem rule 3)")
h = collections.Counter()
for r in rows:
    if r["duration"] <= a.max_duration: h[r["lang"]] += r["duration"]
tot = sum(h.values())
en_frac = h["en-US"] / tot if tot else 0
print(f"       ne {h['ne-NP']/3600:.1f} h | en {h['en-US']/3600:.1f} h")
check(en_frac >= a.min_en_frac,
      f"English is {100*en_frac:.1f}% of kept hours (>= {100*a.min_en_frac:.0f}% required)")

# 6 ---------------------------------------------------------------- val sets
print("\n[6] validation covers both languages")
for n in ("val_mix.jsonl", "val_ne.jsonl", "val_en.jsonl"):
    p = os.path.join(a.manifests, n)
    check(os.path.exists(p), f"{n} exists")
if os.path.exists(os.path.join(a.manifests, "val_mix.jsonl")):
    vl = collections.Counter(json.loads(l)["lang"] for l in
                             open(os.path.join(a.manifests, "val_mix.jsonl"),
                                  encoding="utf-8"))
    check(len(vl) == 2, f"val_mix is bilingual {dict(vl)} -- a single-language "
                        "monitor hid a total collapse for 21 h on the last run")

print("\n" + "=" * 62)
if fails:
    print(f"PREFLIGHT FAILED: {len(fails)} blocking problem(s)")
    for f in fails: print("  - " + f)
    sys.exit(1)
print(f"PREFLIGHT PASSED" + (f" ({len(warns)} warning(s))" if warns else ""))
for w in warns: print("  - " + w)
