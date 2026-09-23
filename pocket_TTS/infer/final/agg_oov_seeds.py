"""Aggregate OOV tail retention across sampling seeds.

One clip per probe is not a measurement at this checkpoint: control_ne alone was
observed at 1.28 s, 3.28 s and empty from identical weights and settings, purely
on RNG state. A single-seed 22-probe tally therefore mostly reports the seed.
This prints per-probe retention as k/N over seeds, and the per-seed totals so the
spread is visible rather than averaged away.
"""
import json, sys, glob, os, statistics

F = os.path.dirname(os.path.abspath(__file__))
paths = sorted(glob.glob(f"{F}/oov_probe_v3_140k_*.json"))
runs = {os.path.basename(p).replace("oov_probe_v3_140k_", "").replace(".json", ""):
        json.load(open(p))["rows"] for p in paths}
if not runs:
    sys.exit("no per-seed jsons yet")

ids = [r["id"] for r in next(iter(runs.values()))]
kind = {r["id"]: r["kind"] for r in next(iter(runs.values()))}
N = len(runs)

print(f"seeds scored: {', '.join(runs)}  (N={N})\n")
print(f"{'probe':14s} {'kind':18s} {'kept':>6s}  {'sec (per seed)':>22s}")
by_kind = {}
for i in ids:
    keeps, secs = [], []
    for rows in runs.values():
        r = next((x for x in rows if x["id"] == i), None)
        if r is None:
            continue
        keeps.append(bool(r["tail_kept"])); secs.append(r["audio_sec"])
    k = sum(keeps)
    by_kind.setdefault(kind[i], []).append(k / len(keeps))
    flag = "  <-- unstable" if 0 < k < len(keeps) else ""
    print(f"{i:14s} {kind[i]:18s} {k:>3d}/{len(keeps):<2d}  "
          f"{' '.join(f'{s:4.1f}' for s in secs):>22s}{flag}")

print(f"\n{'kind':18s} {'mean retention':>15s}")
for k in sorted(by_kind):
    print(f"{k:18s} {statistics.mean(by_kind[k]):>14.2f}")

print()
for name, rows in runs.items():
    kept = sum(r["tail_kept"] for r in rows)
    print(f"seed {name:9s} TAIL RETENTION {kept}/{len(rows)} ({100*kept/len(rows):.0f}%)")
tot = [sum(r["tail_kept"] for r in rows) for rows in runs.values()]
print(f"\nacross seeds: min {min(tot)}/22  max {max(tot)}/22  mean {statistics.mean(tot):.1f}/22")
print("v2 student baseline: raw 10/22, frontend-normalized 20/22 (different model -- see note)")
