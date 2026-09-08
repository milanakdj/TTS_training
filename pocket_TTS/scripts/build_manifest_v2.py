"""Nepali pocket-tts manifest v2 -- the full sweep of /workspace/proc_data_new.

v1 used 223 h from the 5 directories I happened to look at. A scan of all 80 found
~8.5k h of Nepali. Most of that is the same audio in different processing states
(indic_voices_long -> IndicVoices -> indicvoices-r are one corpus), so this picks
one representative per corpus and gates on the quality metadata the team already
computed rather than taking hours at face value:

  ans_snr*        3 independent ASRs (saaras/canary/conformer) transcribed each
                  clip; keep only where they agree. spk_overlap_percent gates out
                  clips with more than one talker, which TTS must not learn.
  mahadhwani /    same 3-ASR setup exposed as low_cer_count; podcast-index sits at
  podcast-index   low_cer_count p50=1, so it needs the >=2 gate to be usable.
  indicvoices-r   already restored; dnsmos_mean/snr/c50 gate the residual noise.
  rasa            studio, ungated, and the only source of the 6 emotion registers.

Audio is referenced in place. Nothing is copied or moved.
"""
import argparse, json, os, random, collections

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/root/tts/TTS_training/pocket_TTS/manifests")
ap.add_argument("--min-s", type=float, default=1.0)
ap.add_argument("--max-s", type=float, default=30.0)
ap.add_argument("--max-asr-cer", type=float, default=0.10)
ap.add_argument("--min-dnsmos", type=float, default=2.7)
args = ap.parse_args()

R = "/workspace/proc_data_new"
SRC, DST = "/projects/data/ttsteam/proc_data_new", R
NE = ("ne", "nepali")


def cer_ok(r):
    m = r.get("cer_wer_metrics") or {}
    vals = [v for k, v in m.items() if k.startswith("cer_") and isinstance(v, (int, float))]
    if vals and max(vals) > args.max_asr_cer:
        return False
    if (r.get("spk_overlap_percent") or 0) > 5:
        return False
    return True


def consensus_ok(r):
    lo, av = r.get("low_cer_count"), r.get("available_cer_count")
    if isinstance(lo, (int, float)) and isinstance(av, (int, float)) and av >= 2:
        return lo >= 2
    return True


def acoustic_ok(r):
    d = r.get("dnsmos_mean")
    return not isinstance(d, (int, float)) or d >= args.min_dnsmos


SOURCES = [
    ("ans_snr50",       "ans_snr50.part1of2/train/manifest.jsonl",    cer_ok),
    ("ans_snr50",       "ans_snr50.part2of2/train/manifest.jsonl",    cer_ok),
    ("ans_snr40-50",    "ans_snr40-50.part1of2/train/manifest.jsonl", cer_ok),
    ("ans_snr40-50",    "ans_snr40-50.part2of2/train/manifest.jsonl", cer_ok),
    ("indicvoices-r",   "indicvoices-r/train/manifest.jsonl",         acoustic_ok),
    ("indicvoices-r-long", "indicvoices-r-long/train/manifest_train.jsonl", acoustic_ok),
    ("mahadhwani",      "mahadhwani/train/manifest.jsonl",            consensus_ok),
    ("podcast-index",   "podcast-index/train/manifest.jsonl",         consensus_ok),
    ("orpheus_snr50",   "orpheus_snr50/train/manifest.jsonl",         cer_ok),
    ("rasa",            "ai4bharat___rasa/train/manifest.jsonl",      lambda r: True),
    ("rasa",            "ai4bharat___rasa/val/manifest.jsonl",        lambda r: True),
]

rows, seen = [], set()
stats = collections.defaultdict(lambda: [0, 0.0, 0])  # kept, hours, rejected
for tag, rel, gate in SOURCES:
    f = os.path.join(R, rel)
    if not os.path.exists(f):
        print("MISSING", rel); continue
    for line in open(f):
        try: r = json.loads(line)
        except Exception: continue
        if str(r.get("language", "")).lower() not in NE: continue
        d = float(r.get("duration") or 0)
        t = (r.get("text") or "").strip()
        if not t or not (args.min_s <= d <= args.max_s):
            stats[tag][2] += 1; continue
        if not gate(r):
            stats[tag][2] += 1; continue
        p = (r.get("audio_filepath") or "").replace(SRC, DST)
        if not p or p in seen:
            stats[tag][2] += 1; continue
        seen.add(p)
        rows.append({"path": p, "duration": round(d, 3), "transcript": t,
                     "speaker": r.get("speaker_id"), "source": tag, "style": r.get("style")})
        stats[tag][0] += 1; stats[tag][1] += d
    print(f"  scanned {rel}", flush=True)

print(f"\n{'source':<22}{'kept':>9}{'hours':>9}{'rejected':>10}")
for k in sorted(stats, key=lambda x: -stats[x][1]):
    v = stats[k]; print(f"{k:<22}{v[0]:>9}{v[1]/3600:>9.1f}{v[2]:>10}")

rng = random.Random(20260905); rng.shuffle(rows)
nval = 1000
os.makedirs(args.out, exist_ok=True)
for name, part in (("train", rows[nval:]), ("valid", rows[:nval])):
    fp = os.path.join(args.out, f"{name}_v2.jsonl")
    with open(fp, "w") as fh:
        for r in part: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(part)} rows, {sum(x['duration'] for x in part)/3600:.1f} h -> {fp}")
print(f"\nTOTAL {len(rows)} clips, {sum(r['duration'] for r in rows)/3600:.1f} h, "
      f"{len({r['speaker'] for r in rows})} speakers")
