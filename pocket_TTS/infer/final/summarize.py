"""Merge WER + similarity + the CPU bench into one report."""
import json, os, collections, statistics

R = "/root/tts/TTS_training/pocket_TTS"
F = f"{R}/infer/final"
wer = json.load(open(f"{F}/wer.json"))
sim = json.load(open(f"{F}/sim.json"))
bench = json.load(open(f"{F}/bench.json")) if os.path.exists(f"{F}/bench.json") else {}
wer_ft = json.load(open(f"{F}/wer_ft.json")) if os.path.exists(f"{F}/wer_ft.json") else None
ORDER = ["real_human", "teacher_24l", "student_6l"]

def mean(xs):
    return statistics.mean(xs) if xs else float("nan")

print("\n" + "=" * 78)
print("FINAL EVAL -- Nepali pocket-TTS depth distill (24L teacher -> 6L student)")
print("=" * 78)
print("\n100 held-out utterances, 77 speaker identities, 6 sources.")
print("Voice prompt is a different clip of the same identity; the target utterance")
print("is held out. real_human = the genuine recording of that same utterance,")
print("scored through the identical pipeline. It is the ceiling, not a competitor.\n")

print(f"{'system':<14} {'n':>4} {'WER':>8} {'CER':>8} {'SIM':>8}")
print("-" * 78)
for k in ORDER:
    w = [r for r in wer.get(k, []) if r["dev_ratio"] >= 0.5]
    s = sim.get(k, [])
    print(f"{k:<14} {len(w):>4} {mean([r['wer'] for r in w]):>8.3f} "
          f"{mean([r['cer'] for r in w]):>8.3f} {mean([r['sim'] for r in s]):>8.3f}")

if wer_ft:
    print("\nSame set, Nepali-finetuned Whisper (the sharper instrument):")
    print(f"{'system':<14} {'n':>4} {'WER':>8} {'CER':>8}")
    print("-" * 78)
    for k in ORDER:
        w = [r for r in wer_ft.get(k, []) if r["dev_ratio"] >= 0.5]
        print(f"{k:<14} {len(w):>4} {mean([r['wer'] for r in w]):>8.3f} "
              f"{mean([r['cer'] for r in w]):>8.3f}")

print("\nBy source (CER, the more stable of the two on Nepali):")
srcs = sorted({r["source"] for r in wer.get("real_human", [])})
print(f"{'source':<20}" + "".join(f"{k:>14}" for k in ORDER))
print("-" * 78)
for src in srcs:
    cells = []
    for k in ORDER:
        w = [r for r in wer.get(k, []) if r["source"] == src and r["dev_ratio"] >= 0.5]
        cells.append(f"{mean([r['cer'] for r in w]):>14.3f}")
    print(f"{src:<20}" + "".join(cells))

if bench:
    print("\nCPU speed (threads pinned to 4; oversubscribing erases the difference):")
    for k, rows in bench.items():
        rtf = mean([r["rtf"] for r in rows])
        print(f"  {k:<14} {rtf:>5.2f}x real-time")
    if "teacher_24l" in bench and "student_6l" in bench:
        t = mean([r["rtf"] for r in bench["teacher_24l"]])
        s = mean([r["rtf"] for r in bench["student_6l"]])
        print(f"  speedup: {s/t:.2f}x")
print()
