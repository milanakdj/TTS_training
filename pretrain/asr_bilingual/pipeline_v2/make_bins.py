"""Precompute lhotse bucket duration bins.

DynamicBucketingSampler estimates its bins by consuming cuts at startup. Over
four lazily-multiplexed manifests on a network filesystem that estimate did not
finish in 9 minutes, which would have stalled every launch before step 1.

The durations are already in the manifests, so the bins are a quantile sweep over
a column we have -- seconds of work, once, instead of a scan per launch.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="/workspace/asr_pretrain_v2/manifests/train_mix.jsonl")
ap.add_argument("--out", default="/workspace/asr_pretrain_v2/manifests/bucket_duration_bins.json")
ap.add_argument("--num-buckets", type=int, default=30)
ap.add_argument("--max-duration", type=float, default=60.0)
ap.add_argument("--min-duration", type=float, default=0.3)
a = ap.parse_args()

d = []
with open(a.manifest, encoding="utf-8") as fh:
    for line in fh:
        v = json.loads(line)["duration"]
        if a.min_duration <= v <= a.max_duration: d.append(v)
d = np.array(d)
# num_buckets buckets need num_buckets-1 interior boundaries.
q = np.linspace(0, 100, a.num_buckets + 1)[1:-1]
bins = [round(float(x), 3) for x in np.percentile(d, q)]
# Strictly increasing, or lhotse rejects them.
out = []
for b in bins:
    if not out or b > out[-1]: out.append(b)
json.dump(out, open(a.out, "w"))
print(f"{len(d)} cuts, duration p1 {np.percentile(d,1):.2f}s p50 {np.percentile(d,50):.2f}s "
      f"p99 {np.percentile(d,99):.2f}s")
print(f"wrote {len(out)} bins -> {a.out}")
print("  " + ", ".join(f"{b:.1f}" for b in out))
