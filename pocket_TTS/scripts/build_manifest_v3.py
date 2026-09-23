"""Nepali + Indian-English manifest v3 -- the mix that keeps English alive.

v2 was 2,210 h of Nepali through a tokenizer with no byte_fallback, so English and
digits reached the model as <unk> and were deleted. v3 changes three things:

  1. tokenizer_v3/ne_en_9682.model -- Kyutai's English vocab EXTENDED with Nepali,
     ids 0..3999 preserved, so the pretrained text embedding still applies and
     English arrives as inherited weights rather than something to relearn.
  2. English replay from en-in_snr{50,40-50}: Indian English, 24 kHz, already
     through the same 3-ASR consensus gate as ans_*. US English would be the wrong
     register for a model reading Nepali brand names.
  3. Nepali transcripts pass through ne_frontend.normalize(keep_latin=True),
     which drops the bracketed Latin glosses ("ट्याक्सी (Taxi)") that would
     otherwise be read aloud twice, verbalizes digits -- but KEEPS Latin words.

     keep_latin added 2026-09-18. The first v3 build transliterated Latin away
     (Airforce -> ऐर्फोर्के), which left 0 of 12,993 Nepali rows containing a
     single Latin character. The model would then have seen pure-Devanagari
     sentences and pure-Latin sentences and never a code-switched one -- which
     is precisely the input that breaks v2. ans_snr50 carries Latin in 16.6% of
     raw rows; that is real Nepali-English code-switching with matching audio,
     and throwing it away wasted the only supervision for the target case.
     The aligner needs no change: align_data.py filters each word to the CTC
     model's alphabet, so a Latin word normalizes to empty, keeps its surface in
     `words`, and gets null timings while the Devanagari around it aligns.

Numbers are deliberately still verbalized in text rather than taught from audio:
Nepali 0-99 is an irregular table, a lookup gets it right every time, and training
signal spent on it would be strictly worse. The tokenizer change is what stops a
bypassed frontend from deleting the sentence.

    python build_manifest_v3.py --en-hours 800
"""
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, "/root/tts/TTS_training/pocket_TTS/frontend")
from ne_frontend import normalize as ne_normalize

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/root/tts/TTS_training/pocket_TTS/manifests")
ap.add_argument("--min-s", type=float, default=1.0)
ap.add_argument("--max-s", type=float, default=30.0)
ap.add_argument("--max-asr-cer", type=float, default=0.10)
ap.add_argument("--min-dnsmos", type=float, default=2.7)
ap.add_argument("--en-hours", type=float, default=800.0)
ap.add_argument("--en-valid", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

R = "/workspace/proc_data_new"
SRC, DST = "/projects/data/ttsteam/proc_data_new", R
NE = ("ne", "nepali")
rng = random.Random(args.seed)


# ------------------------------------------------------------------ gates ---
def cer_ok(r):
    m = r.get("cer_wer_metrics") or {}
    vals = [v for k, v in m.items() if k.startswith("cer_") and isinstance(v, (int, float))]
    if vals and max(vals) > args.max_asr_cer:
        return False
    return (r.get("spk_overlap_percent") or 0) <= 5


def consensus_ok(r):
    lo, av = r.get("low_cer_count"), r.get("available_cer_count")
    if isinstance(lo, (int, float)) and isinstance(av, (int, float)) and av >= 2:
        return lo >= 2
    return True


def acoustic_ok(r):
    d = r.get("dnsmos_mean")
    return not isinstance(d, (int, float)) or d >= args.min_dnsmos


NE_SOURCES = [
    ("ans_snr50",          "ans_snr50.part1of2/train/manifest.jsonl",       cer_ok),
    ("ans_snr50",          "ans_snr50.part2of2/train/manifest.jsonl",       cer_ok),
    ("ans_snr40-50",       "ans_snr40-50.part1of2/train/manifest.jsonl",    cer_ok),
    ("ans_snr40-50",       "ans_snr40-50.part2of2/train/manifest.jsonl",    cer_ok),
    ("indicvoices-r",      "indicvoices-r/train/manifest.jsonl",            acoustic_ok),
    ("indicvoices-r-long", "indicvoices-r-long/train/manifest_train.jsonl", acoustic_ok),
    ("mahadhwani",         "mahadhwani/train/manifest.jsonl",               consensus_ok),
    ("podcast-index",      "podcast-index/train/manifest.jsonl",            consensus_ok),
    ("orpheus_snr50",      "orpheus_snr50/train/manifest.jsonl",            cer_ok),
    ("rasa",               "ai4bharat___rasa/train/manifest.jsonl",         lambda r: True),
    ("rasa",               "ai4bharat___rasa/val/manifest.jsonl",           lambda r: True),
]
EN_SOURCES = [
    ("en-in_snr50",    "en-in_snr50/train/manifest.jsonl",    cer_ok),
    ("en-in_snr40-50", "en-in_snr40-50/train/manifest.jsonl", cer_ok),
]


def speaker_key(r, tag):
    """A bare SPEAKER_01 is per-video: two videos' SPEAKER_01 are different people.

    Key on the channel/video directory as well, or prompt/target pairing silently
    joins unrelated voices (see pocket_TTS/CLAUDE.md).
    """
    spk = r.get("speaker") or r.get("speaker_id") or "?"
    if not re.fullmatch(r"SPEAKER_\d+", str(spk)):
        return f"{tag}/{spk}"
    parts = (r.get("audio_filepath") or "").strip("/").split("/")
    return f"{tag}/{'/'.join(parts[-3:-1])}/{spk}"


def read(rel, gate, langs, tag):
    for line in open(os.path.join(R, rel)):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if str(r.get("language", "")).lower() not in langs:
            continue
        d = float(r.get("duration") or 0)
        t = (r.get("text") or "").strip()
        p = (r.get("audio_filepath") or "").replace(SRC, DST)
        if not t or not p or not (args.min_s <= d <= args.max_s) or not gate(r):
            continue
        yield r, p, d, t


# ---------------------------------------------------------------- nepali ----
ne_rows, seen = [], set()
stats = collections.defaultdict(lambda: [0, 0.0])
for tag, rel, gate in NE_SOURCES:
    if not os.path.exists(os.path.join(R, rel)):
        print("MISSING", rel); continue
    for r, p, d, t in read(rel, gate, NE, tag):
        if p in seen:
            continue
        seen.add(p)
        txt = ne_normalize(t, keep_latin=True)
        if not txt:
            continue
        ne_rows.append({"path": p, "duration": round(d, 3), "transcript": txt,
                        "speaker": speaker_key(r, tag), "source": tag,
                        "language": "ne", "style": r.get("style")})
        stats[tag][0] += 1; stats[tag][1] += d

# --------------------------------------------------------------- english ----
# Pool first, then subsample balanced across speakers so 800 h is many voices
# rather than whichever channel happens to sort first.
pool = collections.defaultdict(list)
en_seen = set()
for tag, rel, gate in EN_SOURCES:
    if not os.path.exists(os.path.join(R, rel)):
        print("MISSING", rel); continue
    for r, p, d, t in read(rel, gate, ("en",), tag):
        if p in en_seen:
            continue
        en_seen.add(p)
        pool[speaker_key(r, tag)].append(
            {"path": p, "duration": round(d, 3), "transcript": re.sub(r"\s+", " ", t),
             "speaker": speaker_key(r, tag), "source": tag, "language": "en", "style": None})

for v in pool.values():
    rng.shuffle(v)
budget, en_rows = args.en_hours * 3600, []
keys = sorted(pool)
rng.shuffle(keys)
took, i = 0.0, 0
while took < budget and any(pool[k] for k in keys):          # round-robin over voices
    k = keys[i % len(keys)]; i += 1
    if pool[k]:
        row = pool[k].pop()
        en_rows.append(row); took += row["duration"]
        stats[row["source"]][0] += 1; stats[row["source"]][1] += row["duration"]

# ----------------------------------------------------------------- split ----
# Reuse v2's Nepali validation paths verbatim so v2/v3 WER stays comparable.
v2_valid = set()
vp = f"{args.out}/valid_v2.jsonl"
if os.path.exists(vp):
    v2_valid = {json.loads(l)["path"] for l in open(vp)}

ne_valid = [r for r in ne_rows if r["path"] in v2_valid]
ne_train = [r for r in ne_rows if r["path"] not in v2_valid]
rng.shuffle(en_rows)
en_valid, en_train = en_rows[:args.en_valid], en_rows[args.en_valid:]

train = ne_train + en_train
valid = ne_valid + en_valid
rng.shuffle(train)

os.makedirs(args.out, exist_ok=True)
for name, rows in (("train_v3", train), ("valid_v3", valid)):
    with open(f"{args.out}/{name}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

hne = sum(r["duration"] for r in train if r["language"] == "ne") / 3600
hen = sum(r["duration"] for r in train if r["language"] == "en") / 3600
print(f"\n{'source':22s} {'rows':>9s} {'hours':>9s}")
for k in sorted(stats, key=lambda x: -stats[x][1]):
    print(f"{k:22s} {stats[k][0]:9d} {stats[k][1]/3600:9.1f}")
print(f"\ntrain {len(train)} rows | ne {hne:.1f} h ({100*hne/(hne+hen):.1f}%) "
      f"| en {hen:.1f} h ({100*hen/(hne+hen):.1f}%) | total {hne+hen:.1f} h")
print(f"valid {len(valid)} rows (ne {len(ne_valid)}, en {len(en_valid)})")
en_voices = len({r["speaker"] for r in en_rows})
print(f"english: {en_voices} distinct (video, speaker) keys sampled from {len(keys)} in the pool")
