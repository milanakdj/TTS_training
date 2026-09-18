"""Assemble the ne+en manifests for the from-scratch bilingual pretrain.

Three Nepali sources plus one English source, written as plain NeMo manifests
(audio_filepath / duration / text / lang / source).

Why the mixture is what it is
-----------------------------
* English is **40% of hours by construction**, not a token replay. The previous
  run carried 301 h English against 1,906 h Nepali (13.6%) and English collapsed
  to a degenerate repetition loop that a mid-run fix could not reverse. See
  release/POSTMORTEM_english_collapse.md rule 3: "do not expect ~10% replay to
  preserve a language under full finetuning."
* English is Indian-accented (`en-in_snr*`), already on the mount with the same
  3-ASR consensus metadata as the Nepali `ans_*` sets. It matches the deployment
  domain better than MLS and needs no 35 GB download.
* The OOV audio ships as **clean + Seed-VC**, not clean + CPU-augmented. The
  augmented third of the "900 h" is noise/speed/pitch over the *same* utterances;
  SpecAugment in the model config already covers that role, while Seed-VC is the
  only lever that touches the actual defect (one synthetic voice, F0 spread
  13 Hz). Materialising it would have cost ~34 GB to triplicate the same text.
  To add it anyway: extract /workspace/oov_distill/out/hf/shard_*/ to wav and
  append with lang="ne-NP", source="oov_aug".

The transliteration gate
------------------------
~5% of the Premal rows are English article text read aloud as Devanagari
transliteration ("चाइना एक्सटेन्ड्स भिसा फ्री पोलिसी"). That is precisely the
English-audio -> Devanagari mapping that the last run died of, and the postmortem
flags this corpus as "worth re-checking if it is ever added". devfrac does NOT
catch it -- the script is Devanagari, only the words are English. So rows are
scored against a 54k-entry Nepali lexicon and dropped when too few of their words
are real Nepali.
"""
import argparse, json, os, sys, random, collections, re, unicodedata

R = "/workspace/proc_data_new"
SRC, DST = "/projects/data/ttsteam/proc_data_new", R
OOV = "/workspace/oov_distill/out"
HERE = os.path.dirname(os.path.abspath(__file__))
NE, EN = "ne-NP", "en-US"
# Step 5 of PRETRAINING_FROM_SCRATCH.md: upsample human labels relative to
# pseudo-labels so the model is not optimised toward canary's output
# distribution (86% of our Nepali carries canary's labels; canary is 0.155 WER
# on gold, so training on them is distillation from canary). Carried as a field
# rather than by duplicating rows -- lhotse applies per-source weights.
#   human  : read by a person, transcribed by a person
#   script : synthetic audio read FROM a known script; text is exact, voice is not
#   pseudo : machine transcript accepted by 3-ASR consensus
LABEL_KIND = {
    "indicvoices-r": "human", "indicvoices-r-long": "human", "rasa": "human",
    "ans_snr50": "pseudo", "ans_snr40-50": "pseudo", "mahadhwani": "pseudo",
    "podcast-index": "pseudo", "orpheus_snr50": "pseudo",
    "en-in_snr40-50": "pseudo", "en-in_snr50": "pseudo",
    "oov_clean": "script", "oov_vc": "script",
}          # prompt_dictionary keys; bare "ne" is absent
                                    # from the dict and dies ~2k steps in.

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/workspace/asr_pretrain_v2/manifests")
ap.add_argument("--en-frac", type=float, default=0.40,
                help="English share of total hours (postmortem rule 3: >=0.30)")
ap.add_argument("--max-s", type=float, default=60.0)
ap.add_argument("--min-s", type=float, default=1.0)
ap.add_argument("--oov-max-s", type=float, default=60.0)
ap.add_argument("--max-asr-cer", type=float, default=0.10)
ap.add_argument("--min-dnsmos", type=float, default=2.7)
ap.add_argument("--min-tu-ratio", type=float, default=1.3,
                help="minimum encoder-frames / target-tokens. CTC cannot align a "
                     "target longer than the frame sequence; at 8x subsampling "
                     "there are 12.5 frames/s, so this also rejects transcripts "
                     "that imply an impossible speaking rate.")
ap.add_argument("--min-ne-lex-frac", type=float, default=0.55,
                help="min fraction of Devanagari words found in the Nepali lexicon")
ap.add_argument("--vc-only-passed", type=int, default=1)
ap.add_argument("--val-per-lang", type=int, default=1200)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = random.Random(a.seed)

# ---------------------------------------------------------------- gates
def cer_ok(r):
    m = r.get("cer_wer_metrics") or {}
    v = [x for k, x in m.items() if k.startswith("cer_") and isinstance(x, (int, float))]
    if v and max(v) > a.max_asr_cer: return False
    return (r.get("spk_overlap_percent") or 0) <= 5

def consensus_ok(r):
    lo, av = r.get("low_cer_count"), r.get("available_cer_count")
    if isinstance(lo, (int, float)) and isinstance(av, (int, float)) and av >= 2:
        return lo >= 2
    return True

def acoustic_ok(r):
    d = r.get("dnsmos_mean")
    return not isinstance(d, (int, float)) or d >= a.min_dnsmos

# ---------------------------------------------------------------- lexicon
def load_lex():
    p = os.path.join(HERE, "assets", "ne_lexicon.tsv")
    s = set()
    if not os.path.exists(p): return s
    for line in open(p, encoding="utf-8"):
        if line.startswith("#"): continue
        w = line.split("\t")[0].strip()
        if w: s.add(w)
    return s

LEX = load_lex()
DEV = re.compile(r"[ऀ-ॿ]+")

def ne_lex_frac(t):
    """Fraction of Devanagari words present in the lexicon. English-read-as-
    Devanagari scores low because its words are transliterated English."""
    w = DEV.findall(t)
    if not w: return 1.0
    return sum(1 for x in w if x in LEX) / len(w)

# ---------------------------------------------------------------- nepali base
NE_SOURCES = [
    ("ans_snr50",          "ans_snr50.part1of2/train/manifest.jsonl",        cer_ok),
    ("ans_snr50",          "ans_snr50.part2of2/train/manifest.jsonl",        cer_ok),
    ("ans_snr40-50",       "ans_snr40-50.part1of2/train/manifest.jsonl",     cer_ok),
    ("ans_snr40-50",       "ans_snr40-50.part2of2/train/manifest.jsonl",     cer_ok),
    ("indicvoices-r",      "indicvoices-r/train/manifest.jsonl",             acoustic_ok),
    ("indicvoices-r-long", "indicvoices-r-long/train/manifest_train.jsonl",  acoustic_ok),
    ("mahadhwani",         "mahadhwani/train/manifest.jsonl",                consensus_ok),
    ("podcast-index",      "podcast-index/train/manifest.jsonl",             consensus_ok),
    ("orpheus_snr50",      "orpheus_snr50/train/manifest.jsonl",             cer_ok),
    ("rasa",               "ai4bharat___rasa/train/manifest.jsonl",          lambda r: True),
    ("rasa",               "ai4bharat___rasa/val/manifest.jsonl",            lambda r: True),
]
# `train/manifest.jsonl` is the concatenation of the `manifest.part_*.jsonl`
# shards beside it -- reading both double-counts the corpus. Use one.
EN_SOURCES = [
    ("en-in_snr40-50", "en-in_snr40-50/train/manifest.jsonl", cer_ok),
    ("en-in_snr50",    "en-in_snr50/train/manifest.jsonl",    cer_ok),
]

def scan(sources, langs, lang_tag, stats):
    rows, seen = [], set()
    for tag, rel, gate in sources:
        f = os.path.join(R, rel)
        if not os.path.exists(f):
            print(f"  MISSING {rel}"); continue
        n0 = len(rows)
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            if str(r.get("language", "")).lower() not in langs: continue
            d = float(r.get("duration") or 0)
            t = (r.get("text") or "").strip()
            if not t or not (a.min_s <= d <= a.max_s):
                stats[tag][2] += 1; continue
            if not gate(r):
                stats[tag][2] += 1; continue
            p = (r.get("audio_filepath") or "").replace(SRC, DST)
            # ans_snr40-50 and ans_snr50 overlap ~46% of the smaller band, and
            # the en-in pair overlaps likewise. Dedupe on the resolved path.
            if not p or p in seen:
                stats[tag][2] += 1; continue
            seen.add(p)
            rows.append(dict(audio_filepath=p, duration=round(d, 3), text=t,
                             lang=lang_tag, source=tag,
                             label_kind=LABEL_KIND.get(tag, "pseudo")))
            stats[tag][0] += 1; stats[tag][1] += d
        print(f"  {tag:22s} +{len(rows)-n0}")
    return rows

# ---------------------------------------------------------------- oov (premal)
def oov_rows():
    sel = {}
    for line in open(f"{OOV}/selected.jsonl", encoding="utf-8"):
        r = json.loads(line); sel[r["id"]] = r
    print(f"  selected.jsonl: {len(sel)} rows")

    # Which VC outputs cleared the speaker gate.
    passed = {}
    for mf in (f"{OOV}/vc/vc_manifest.jsonl", f"{OOV}/vc_all/vc_manifest.jsonl"):
        if not os.path.exists(mf): continue
        for line in open(mf, encoding="utf-8"):
            try: m = json.loads(line)
            except Exception: continue
            passed[m["id"]] = m
    print(f"  vc manifests: {len(passed)} scored, "
          f"{sum(1 for m in passed.values() if m['passed'])} passed the 0.85 gate")

    rows, drop_lex, drop_dur, miss = [], 0, 0, 0
    for i, r in sel.items():
        if not (a.min_s <= r["dur"] <= a.oov_max_s): drop_dur += 1; continue
        if ne_lex_frac(r["text"]) < a.min_ne_lex_frac: drop_lex += 1; continue
        clean = f"{OOV}/vc_src/{i}.wav"
        if os.path.exists(clean):
            rows.append(dict(audio_filepath=clean, duration=round(r["dur"], 3),
                             text=r["text"], lang=NE, source="oov_clean",
                             label_kind="script"))
        else: miss += 1
        m = passed.get(i)
        if m and (m["passed"] or not a.vc_only_passed):
            for cand in (f"{OOV}/vc_all/{i}_vc.wav", f"{OOV}/vc/{i}_vc.wav"):
                if os.path.exists(cand):
                    rows.append(dict(audio_filepath=cand, duration=round(m["dur"], 3),
                                     text=r["text"], lang=NE, source="oov_vc",
                                     label_kind="script"))
                    break
    print(f"  dropped {drop_lex} english-read-as-devanagari (lex<{a.min_ne_lex_frac}), "
          f"{drop_dur} out-of-duration, {miss} missing clean wav")
    return rows

# ---------------------------------------------------------------- build
def hours(rs): return sum(r["duration"] for r in rs) / 3600

print("[1/4] Nepali base corpus")
st = collections.defaultdict(lambda: [0, 0.0, 0])
ne_base = scan(NE_SOURCES, ("ne", "nepali"), NE, st)
print(f"  -> {len(ne_base)} rows / {hours(ne_base):.1f} h")

print("[2/4] OOV-distilled (Premal + Seed-VC)")
ne_oov = oov_rows()
print(f"  -> {len(ne_oov)} rows / {hours(ne_oov):.1f} h")

ne_all = ne_base + ne_oov
ne_h = hours(ne_all)

print("[3/4] English (Indian-accented)")
st_en = collections.defaultdict(lambda: [0, 0.0, 0])
en_pool = scan(EN_SOURCES, ("en", "english"), EN, st_en)
print(f"  -> pool {len(en_pool)} rows / {hours(en_pool):.1f} h")
# Sample English down to the target share of total hours.
target_en_h = ne_h * a.en_frac / (1 - a.en_frac)
rng.shuffle(en_pool)
en, acc = [], 0.0
for r in en_pool:
    if acc / 3600 >= target_en_h: break
    en.append(r); acc += r["duration"]
print(f"  target {target_en_h:.1f} h at en_frac={a.en_frac} -> kept "
      f"{len(en)} rows / {hours(en):.1f} h"
      + ("  *** POOL EXHAUSTED, share is below target ***"
         if hours(en) < target_en_h * 0.98 else ""))

# Preflight measured a 3.8% UNK rate on the raw text: the BPE has no uppercase,
# no digits and little punctuation, so those become trainable <unk> targets.
# Normalise to the tokenizer's character set, verbalising numbers first so they
# are spoken rather than deleted (Step 3).
print("[3.5/4] normalising text to the tokenizer character set")
import sentencepiece as spm
sys.path.insert(0, HERE)
from normalize import Normalizer
TOKDIR = "/workspace/asr_pretrain_v2/tokenizer"
_nz = Normalizer(spm.SentencePieceProcessor(
    model_file=os.path.join(TOKDIR, "tokenizer.model")))
_nz.sp = spm.SentencePieceProcessor(model_file=os.path.join(TOKDIR, "tokenizer.model"))

def norm_all(rs, tag):
    out, dropped = [], 0
    for r in rs:
        t = _nz(r["text"], r["lang"])
        if not t:
            dropped += 1; continue
        r["text"] = t; out.append(r)
    print(f"  {tag}: {len(out)} kept, {dropped} empty after normalisation")
    return out

ne_all = norm_all(ne_all, "nepali")
en = norm_all(en, "english")

# T/U gate. The Premal transcripts are the whole article while the audio is only
# an excerpt: median frames/tokens 0.69, and 76% of those rows have fewer encoder
# frames than target tokens. CTC cannot align that -- the loss goes infinite and
# zero_infinity silently drops it, so 11% of the mix was contributing nothing but
# noise. Measured after normalisation, because verbalising numbers lengthens U.
FPS = 12.5   # 10 ms window stride, 8x subsampling -> 80 ms per encoder frame
def tu_gate(rs, tag):
    out, dropped = [], collections.Counter()
    for r in rs:
        u = max(len(_nz.sp.encode(r["text"])), 1)
        if r["duration"] * FPS / u < a.min_tu_ratio:
            dropped[r["source"]] += 1; continue
        out.append(r)
    n = sum(dropped.values())
    print(f"  {tag}: dropped {n} rows below T/U {a.min_tu_ratio}"
          + (f" -- {dict(dropped)}" if n else ""))
    return out

print("[3.6/4] T/U gate (CTC alignability)")
ne_all = tu_gate(ne_all, "nepali")
en = tu_gate(en, "english")

print("[4/4] splits")
rng.shuffle(ne_all); rng.shuffle(en)
# Validation is held out from train and is *bilingual*: the last run monitored
# val_wer on Nepali only and stayed blind to a total English collapse for 21 h.
ne_val, ne_tr = ne_all[:a.val_per_lang], ne_all[a.val_per_lang:]
en_val, en_tr = en[:a.val_per_lang], en[a.val_per_lang:]
train = ne_tr + en_tr
rng.shuffle(train)

def dump(name, rs):
    p = os.path.join(a.out, name)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rs: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {name:24s} {len(rs):>8} rows  {hours(rs):>8.1f} h")
    return p

dump("train_mix.jsonl", train)

# Per-source weighting (Step 5): 86% of our Nepali carries canary's labels, and
# canary is 0.155 WER on gold -- training on them unweighted is distillation from
# canary. The doc says to use lhotse per-source weights rather than physically
# duplicating rows, so the mix is also emitted as separate buckets plus an
# input_cfg that upsamples human-labelled Nepali 3x relative to its hours.
BUCKET_W = {"ne_human": 3.0, "ne_script": 1.0, "ne_pseudo": 1.0, "en_pseudo": 1.0}
buckets = collections.defaultdict(list)
for r in train:
    lg = "ne" if r["lang"] == NE else "en"
    buckets[f"{lg}_{r['label_kind']}"].append(r)
icfg = []
for name, rs in sorted(buckets.items()):
    if not rs: continue
    path = dump(f"train_{name}.jsonl", rs)
    icfg.append({"type": "nemo", "manifest_filepath": path,
                 "weight": round(hours(rs) * BUCKET_W.get(name, 1.0), 3),
                 "tags": {"bucket": name}})
tw = sum(c["weight"] for c in icfg)
for c in icfg: c["weight"] = round(c["weight"] / tw, 6)
with open(os.path.join(a.out, "train_input_cfg.yaml"), "w") as fh:
    import yaml; yaml.safe_dump(icfg, fh, sort_keys=False)
print("\n== sampling weights (human Nepali upsampled 3x) ==")
for c in icfg:
    print(f"  {c['tags']['bucket']:14s} weight {c['weight']:.4f}")
dump("val_mix.jsonl", ne_val + en_val)
dump("val_ne.jsonl", ne_val)
dump("val_en.jsonl", en_val)

tot = hours(train)
by = collections.Counter()
for r in train: by[r["source"]] += r["duration"]
print("\n== train mix ==")
for k, v in sorted(by.items(), key=lambda x: -x[1]):
    print(f"  {k:22s} {v/3600:8.1f} h  {100*v/sum(by.values()):5.1f}%")
kind = collections.Counter()
for r in train: kind[r["label_kind"]] += r["duration"]
print("\n== label provenance ==")
for k, v in sorted(kind.items(), key=lambda x: -x[1]):
    print(f"  {k:22s} {v/3600:8.1f} h  {100*v/sum(kind.values()):5.1f}%")
en_share = sum(r["duration"] for r in train if r["lang"] == EN) / sum(
    r["duration"] for r in train)
print(f"\n  total {tot:.1f} h | English {100*en_share:.1f}% of hours")
json.dump({"total_h": tot, "en_share": en_share,
           "by_source_h": {k: v/3600 for k, v in by.items()},
           "by_label_kind_h": {k: v/3600 for k, v in kind.items()},
           "ne_h": ne_h, "rows": len(train)},
          open(os.path.join(a.out, "mix_report.json"), "w"), indent=2)
