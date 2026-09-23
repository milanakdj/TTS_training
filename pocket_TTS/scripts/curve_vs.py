#!/usr/bin/env python3
"""Print a run's valid curve beside previous generations' at MATCHED steps.

v3 ran 200,000 steps behind a loss curve that was flat from ~100k and read as
converged. It had converged to +0.0831; v2 passed that on its way down by step
5,000 and finished at -0.0746. Nobody put the two side by side, so a dead run
burned 72 h of H100 and a stage-2 distill on top of it.

    scripts/curve_vs.py <run.log> [more.log ...]

Baselines are read from the v2 and v3 teacher logs in this repo, so they stay
honest if those files are ever regenerated.
"""
import re, sys, os
from pathlib import Path

R = Path("/root/tts/TTS_training/pocket_TTS")
BASE = {"v2 teacher": R / "train_teacher.log", "v3 teacher": R / "train_teacher_v3.log"}
# Runs with split valid sets log `valid[ne] @ step N:`; older runs log `valid @`.
# Matching only the untagged form would make this tool silently report nothing
# for exactly the bilingual runs it exists to judge.
PAT = re.compile(r"valid(?:\s*\[(\w+)\])? @ step (\d+): \{(.*)\}")
KEY = re.compile(r"'(\w+)': '([-\d.e+]+)'")


def curve(path, metric="flow_loss", which=None):
    """`which` selects the named valid set; None accepts untagged lines only."""
    out = {}
    if not Path(path).exists():
        return out
    for line in open(path, errors="ignore"):
        m = PAT.search(line)
        if not m:
            continue
        tag, step, body = m.group(1), m.group(2), m.group(3)
        if which is None:
            if tag is not None:
                continue
        elif tag != which:
            continue
        d = dict(KEY.findall(body))
        if metric in d:
            out[int(step)] = float(d[metric])
    return out


def main():
    runs = sys.argv[1:]
    if not runs:
        sys.exit(__doc__)
    metric = os.environ.get("METRIC", "flow_loss")
    # v2/v3 predate split valid sets, so their lines are untagged.
    # v2 and v3 have NO English valid set -- their only curve is Nepali. Showing
    # them in an English table invites comparing English against Nepali, so
    # NOBASE=1 drops them.
    series = ({} if os.environ.get("NOBASE") else
              {name: curve(p, metric) for name, p in BASE.items()})
    which = os.environ.get("SET", "ne")
    for p in runs:
        c = curve(p, metric, which=which) or curve(p, metric)   # tagged, else untagged
        series[Path(p).stem] = c
    series = {k: v for k, v in series.items() if v}
    steps = sorted({s for v in series.values() for s in v})
    names = list(series)

    print(f"\nvalid {metric} [set={os.environ.get('SET','ne')}] at matched steps  (lower is better)")
    print(f"{'step':>8}" + "".join(f"{n:>16}" for n in names))
    print("-" * (8 + 16 * len(names)))
    for s in steps:
        if s > 50000 and s % 25000:        # dense early, sparse late
            continue
        row = f"{s:>8}"
        for n in names:
            v = series[n].get(s)
            row += f"{v:>16.4f}" if v is not None else f"{'':>16}"
        print(row)

    print(f"\n{'run':>16}  {'best':>9}  {'at step':>8}  {'last':>9}")
    for n in names:
        b = min(series[n].values())
        at = min(series[n], key=series[n].get)
        last = series[n][max(series[n])]
        print(f"{n:>16}  {b:>9.4f}  {at:>8}  {last:>9.4f}")

    # The verdict line: is the new run on v2's trajectory or v3's?
    ref = {k: series[k] for k in ("v2 teacher", "v3 teacher") if k in series}
    for n in names:
        if n in BASE:
            continue
        common = [s for s in series[n] if all(s in r for r in ref.values())]
        if not common or len(ref) < 2:
            continue
        s = max(common)
        d2 = abs(series[n][s] - ref["v2 teacher"][s])
        d3 = abs(series[n][s] - ref["v3 teacher"][s])
        verdict = "tracking v2 (GOOD)" if d2 < d3 else "tracking v3 (BAD)"
        print(f"\n@ step {s}: {n} = {series[n][s]:.4f}  "
              f"v2 {ref['v2 teacher'][s]:.4f}  v3 {ref['v3 teacher'][s]:.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()
