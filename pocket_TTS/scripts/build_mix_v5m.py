"""Latents manifest for v5m / v5m_lwf: v5l's English (LibriHeavy round 1 + synthetic
concat/insert) plus LibriHeavy round 2 (prep_libriheavy_v5b.py), at ~15% English
instead of v5l's 4.9%. Same Nepali rows, same one-shared-repeat-factor rule as
build_mix_v5.py, so v5m differs from v5l only in how much real English it holds.

    python3 scripts/build_mix_v5m.py
"""
import json, os, random

R = "/root/tts/TTS_training/pocket_TTS"
MIX_DIR = "/workspace/manifests_v4"
NE = f"{R}/manifests/train_v3_ne_only_latents.jsonl"
SYN = "/workspace/v5_synth/audio/train_v5_synth_latents.jsonl"
LH = "/workspace/v5_synth/libriheavy/train_v5_libriheavy_latents.jsonl"

LH_B = "/workspace/v5_synth/libriheavy_b/train_v5_libriheavy_b_latents.jsonl"
TARGET_EN_SHARE = 0.15

def rows(path):  # as build_mix_v5.py: latents_file re-rooted at MIX_DIR
    base = os.path.dirname(path)
    out = []
    for line in open(path):
        d = json.loads(line)
        d["latents_file"] = os.path.relpath(os.path.join(base, d["latents_file"]), MIX_DIR)
        out.append(json.dumps(d, ensure_ascii=False) + "\n")
    return out


def meta(path):
    return json.load(open(path.replace(".jsonl", ".meta.json")))


ne_meta = meta(NE)
assert meta(LH_B)["mimi_hash"] == ne_meta["mimi_hash"], "LH_B mimi hash differs"
n_ne = sum(1 for _ in open(NE))
syn = rows(SYN)
cross = [l for l in syn if json.loads(l)["source"] in ("v5_concat", "v5_insert")]
en = rows(LH) + rows(LH_B) + cross
dup = max(1, round(TARGET_EN_SHARE * n_ne / (1 - TARGET_EN_SHARE) / len(en)))
en = en * dup
random.Random(17).shuffle(en)
out = f"{MIX_DIR}/train_v5m_latents.jsonl"
tmp = out + ".tmp"
with open(NE) as src, open(tmp, "w") as f:
    for line in src:
        f.write(line)
    f.writelines(en)
os.replace(tmp, out)
json.dump(ne_meta, open(out.replace(".jsonl", ".meta.json"), "w"), indent=1)
print(f"v5m: Nepali {n_ne:,} + English {len(en):,} (x{dup}) = {len(en)/(n_ne+len(en)):.2%} English -> {out}")
