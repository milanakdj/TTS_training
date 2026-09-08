"""Keep only Nepal-native audio for the pocket-tts training set.

The user auditioned 28 clips and confirmed every ans_* channel sounds properly
Nepali. The AI4Bharat-collected corpora are the opposite: indicvoices-r Nepali is
100% West Bengal (Kalimpong/Darjeeling/Jalpaiguri/Alipurduar, all 908 speakers),
and Rasa was collected by the same group -- its Srijana voice was rejected by ear
as "a Hindi person speaking Nepali". Training a Nepali TTS on that teaches the
accent, so those sources are dropped here.

Alignment ran over everything, so this is a free post-filter; `source` survives
align_data unchanged.
"""
import argparse, json, collections

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--keep", default="ans_snr50,ans_snr40-50,mahadhwani,orpheus_snr50",
                help="comma-separated sources judged Nepal-native")
ap.add_argument("--require-words", action="store_true",
                help="drop rows without alignment; pocket-tts's _choose_cut returns "
                     "None for those, so they contribute no voice-prompt/target split")
args = ap.parse_args()
KEEP = {s.strip() for s in args.keep.split(",") if s.strip()}

kept = collections.Counter(); khrs = collections.Counter()
drop = collections.Counter(); dhrs = collections.Counter()
n_nowords = 0
with open(args.out, "w") as w:
    for line in open(args.inp):
        try: r = json.loads(line)
        except Exception: continue
        s = r.get("source", "?"); d = float(r.get("duration") or 0)
        if s not in KEEP:
            drop[s] += 1; dhrs[s] += d; continue
        if args.require_words and not r.get("words"):
            n_nowords += 1; continue
        kept[s] += 1; khrs[s] += d
        w.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"{'KEPT (Nepal-native)':<26}{'clips':>9}{'hours':>9}")
for s in sorted(khrs, key=lambda x: -khrs[x]): print(f"  {s:<24}{kept[s]:>9}{khrs[s]/3600:>9.1f}")
print(f"  {'TOTAL':<24}{sum(kept.values()):>9}{sum(khrs.values())/3600:>9.1f}")
print(f"\n{'DROPPED (Indian-Nepali)':<26}{'clips':>9}{'hours':>9}")
for s in sorted(dhrs, key=lambda x: -dhrs[x]): print(f"  {s:<24}{drop[s]:>9}{dhrs[s]/3600:>9.1f}")
if args.require_words: print(f"\ndropped for missing alignment: {n_nowords}")
print(f"\n-> {args.out}")
