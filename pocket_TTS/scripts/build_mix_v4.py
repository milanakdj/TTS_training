"""Build a Nepali+English latents manifest at a chosen English share.

Reuses the latents already computed for v3 -- train_v3_ne_only_latents.jsonl and
the English rows of train_v3_aligned_latents.jsonl share a mimi_hash, so they are
interchangeable and nothing needs re-encoding (the encode pass costs ~65 min on
this box).

v3 used 27% English and destroyed Nepali. The VLM-forgetting result is that
replay gains saturate early -- most of the benefit by 3-10% -- so the budget
spent above that came straight out of the Nepali half for nothing.

    scripts/build_mix_v4.py --en-share 0.05
"""
import argparse, json, os, random, shutil

R = "/root/tts/TTS_training/pocket_TTS"
NE = f"{R}/manifests/train_v3_ne_only_latents.jsonl"
ALL = f"{R}/manifests/train_v3_aligned_latents.jsonl"
OUT_DIR = "/workspace/manifests_v4"          # / is at 98%; this file is ~2.2 GB

ap = argparse.ArgumentParser()
ap.add_argument("--en-share", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=13)
a = ap.parse_args()

ne_meta = json.load(open(NE.replace(".jsonl", ".meta.json")))
all_meta = json.load(open(ALL.replace(".jsonl", ".meta.json")))
assert ne_meta["mimi_hash"] == all_meta["mimi_hash"], "mimi hash mismatch: latents not interchangeable"

n_ne = sum(1 for _ in open(NE))
n_en_want = round(n_ne * a.en_share / (1 - a.en_share))
print(f"Nepali rows {n_ne:,}; want {n_en_want:,} English for {a.en_share:.0%} share")

en_lines = []
rng = random.Random(a.seed)
seen = 0
for line in open(ALL):
    if '"source": "en-in' not in line and '"en-in' not in line[:400]:
        continue
    d = json.loads(line)
    if not d.get("source", "").startswith("en-in"):
        continue
    seen += 1
    if len(en_lines) < n_en_want:          # reservoir sample: unbiased, one pass
        en_lines.append(line)
    else:
        j = rng.randrange(seen)
        if j < n_en_want:
            en_lines[j] = line
print(f"sampled {len(en_lines):,} of {seen:,} English rows")

out = f"{OUT_DIR}/train_mix{int(a.en_share*100)}en_latents.jsonl"
tmp = out + ".tmp"
shutil.copyfile(NE, tmp)
with open(tmp, "a") as f:
    f.writelines(en_lines)
os.replace(tmp, out)
json.dump(ne_meta, open(out.replace(".jsonl", ".meta.json"), "w"), indent=1)
total = n_ne + len(en_lines)
print(f"wrote {out}  {total:,} rows  ({len(en_lines)/total:.2%} English)  "
      f"{os.path.getsize(out)/1e9:.2f} GB")
print("NOTE: the .idx rebuilds itself on first use (mtime check in manifest.py).")
