"""Latents manifests for the three v5 probe arms (same Nepali rows in each).

  S  Nepali + Kyutai plain + concat + insert      does Kyutai's own English hold English?
  L  Nepali + LibriHeavy   + concat + insert      is clean real English as good?
  X  Nepali +                concat + insert      are the cross-lingual clips enough alone?

English rows are repeated by ONE factor, chosen so arm S is ~5% English (v4b's
share), and applied to every arm: the arms then differ only in which English
they contain, not in how often each clip is seen.

Mixes go in /workspace/manifests_v4, whose latents/ is a symlink to the Nepali
latents, so Nepali rows are copied verbatim; English rows get their latents_file
rewritten relative to that dir (the loader joins manifest-dir / latents_file).

    python3 scripts/build_mix_v5.py
"""
import json, os, random

R = "/root/tts/TTS_training/pocket_TTS"
MIX_DIR = "/workspace/manifests_v4"
NE = f"{R}/manifests/train_v3_ne_only_latents.jsonl"
SYN = "/workspace/v5_synth/audio/train_v5_synth_latents.jsonl"
LH = "/workspace/v5_synth/libriheavy/train_v5_libriheavy_latents.jsonl"
TARGET_EN_SHARE = 0.05


def rows(path):
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
for p in (SYN, LH):
    m = meta(p)
    assert m["mimi_hash"] == ne_meta["mimi_hash"], f"{p}: mimi hash differs from Nepali latents"
    if m["stitch_frames"] != ne_meta["stitch_frames"]:
        print(f"WARNING {p}: stitch_frames {m['stitch_frames']} vs Nepali {ne_meta['stitch_frames']}")

n_ne = sum(1 for _ in open(NE))
syn = rows(SYN)
kind = lambda l: json.loads(l)["source"]
plain = [l for l in syn if kind(l) == "v5_plain"]
cross = [l for l in syn if kind(l) in ("v5_concat", "v5_insert")]
lh = rows(LH)
arms = {"S": plain + cross, "L": lh + cross, "X": cross}
dup = max(1, round(TARGET_EN_SHARE * n_ne / (1 - TARGET_EN_SHARE) / len(arms["S"])))
print(f"Nepali {n_ne:,} | plain {len(plain):,} cross {len(cross):,} libriheavy {len(lh):,} | dup x{dup}")

for arm, en in arms.items():
    out = f"{MIX_DIR}/train_v5{arm.lower()}_latents.jsonl"
    en = en * dup
    random.Random(17).shuffle(en)
    tmp = out + ".tmp"
    with open(NE) as src, open(tmp, "w") as f:
        for line in src:
            f.write(line)
        f.writelines(en)
    os.replace(tmp, out)
    json.dump(ne_meta, open(out.replace(".jsonl", ".meta.json"), "w"), indent=1)
    print(f"arm {arm}: {n_ne + len(en):,} rows, {len(en)/(n_ne + len(en)):.2%} English -> {out}")
