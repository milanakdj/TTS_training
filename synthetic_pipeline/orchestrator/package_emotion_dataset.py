"""
Package the emotional Nepali TTS run into a training-ready dataset.

Reads the production JSONL manifest, keeps only gated rows, writes:
  <out>/metadata.parquet      one row per clip, Parler-ready columns
  <out>/metadata.csv          same, for eyeballing
  <out>/train.parquet         split
  <out>/validation.parquet    split
  <out>/DATASET_CARD.md       provenance, method, known limitations
  <out>/stats.json            per-emotion hours / counts / CER distribution

Audio itself is NOT copied -- the manifest points at the FLAC files already on
disk. Copying 100h would double the footprint on a disk that started at 89%.

Split policy: split on sentence_id, not on rows. ~20% of sentences appear in
all five emotions as contrastive pairs, so a naive row split would put the same
text in both train and validation and inflate the apparent validation score.

Run: .venv/bin/python package_emotion_dataset.py [--suffix ""] [--val-fraction 0.01]
"""
import argparse
import json
import os
import random

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--suffix", default="")
ap.add_argument("--val-fraction", type=float, default=0.01)
ap.add_argument("--tier", default="AB", choices=["A", "AB"],
                help="A = high-confidence subset only; AB = everything that passed the gate")
ap.add_argument("--seed", type=int, default=20260903)
ap.add_argument("--emotions", default=None,
                help="comma-separated subset to package. Use it to ship only the emotions "
                     "whose audio actually carries the emotion -- the manifest also holds "
                     "the angry/happy rows from the run that failed separation testing.")
ap.add_argument("--out-name", default=None,
                help="output directory name under the pipeline root (default: "
                     "dataset_emotion_nepali<suffix>)")
args = ap.parse_args()

PIPE = "/root/tts/TTS_training/synthetic_pipeline"
SUF = f"_{args.suffix}" if args.suffix else ""
MANIFEST = os.path.join(PIPE, "manifests", f"emotion_production{SUF}.jsonl")
OUT = os.path.join(PIPE, args.out_name or ("dataset_emotion_nepali" + SUF))
os.makedirs(OUT, exist_ok=True)

EMOTIONS = ["angry", "happy", "excited", "sad", "neutral"]
if args.emotions:
    want = {e.strip() for e in args.emotions.split(",") if e.strip()}
    unknown = want - set(EMOTIONS)
    if unknown:
        raise SystemExit("unknown emotion(s): " + ", ".join(sorted(unknown)))
    EMOTIONS = [e for e in EMOTIONS if e in want]

rows = []
with open(MANIFEST) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("kept"):
            continue
        if args.tier == "A" and r.get("tier") != "A":
            continue
        if r.get("emotion") not in EMOTIONS:
            continue
        rows.append(r)

if not rows:
    raise SystemExit(f"no kept rows in {MANIFEST}")

df = pd.DataFrame(rows)

# Columns a Parler-TTS finetune actually consumes, plus provenance/QC.
cols = ["audio_path", "audio_relpath", "text", "description", "emotion", "speaker",
        "language", "duration_s", "sr", "format", "tier",
        "cer_source", "cer_final", "cer_delta", "devanagari_ratio_final",
        "rms_p95", "centroid_hz", "speech_fraction", "peak",
        "edge_voice", "edge_rate", "edge_pitch", "edge_volume",
        "vc_backend", "vc_ref_path", "length_adjust",
        "sentence_id", "description_variant", "contrastive_pair", "transcript_final"]
df = df[[c for c in cols if c in df.columns]]

# Split on sentence_id so contrastive pairs never straddle the split.
rng = random.Random(args.seed)
sids = sorted(df["sentence_id"].unique())
rng.shuffle(sids)
n_val = max(1, int(len(sids) * args.val_fraction))
val_sids = set(sids[:n_val])
val = df[df["sentence_id"].isin(val_sids)].reset_index(drop=True)
train = df[~df["sentence_id"].isin(val_sids)].reset_index(drop=True)

df.to_parquet(os.path.join(OUT, "metadata.parquet"), index=False)
df.to_csv(os.path.join(OUT, "metadata.csv"), index=False)
train.to_parquet(os.path.join(OUT, "train.parquet"), index=False)
val.to_parquet(os.path.join(OUT, "validation.parquet"), index=False)

stats = {
    "total_clips": int(len(df)),
    "total_hours": round(float(df["duration_s"].sum()) / 3600, 3),
    "train_clips": int(len(train)),
    "validation_clips": int(len(val)),
    "tier_filter": args.tier,
    "per_emotion": {},
    "tier_counts": {k: int(v) for k, v in df["tier"].value_counts().items()},
    "cer_final": {
        "mean": round(float(df["cer_final"].mean()), 4),
        "median": round(float(df["cer_final"].median()), 4),
        "p90": round(float(df["cer_final"].quantile(0.90)), 4),
    },
    "cer_delta": {
        "mean": round(float(df["cer_delta"].mean()), 4),
        "median": round(float(df["cer_delta"].median()), 4),
        "p90": round(float(df["cer_delta"].quantile(0.90)), 4),
    },
}
for e in EMOTIONS:
    sub = df[df["emotion"] == e]
    if sub.empty:
        stats["per_emotion"][e] = {"clips": 0, "hours": 0.0}
        continue
    stats["per_emotion"][e] = {
        "clips": int(len(sub)),
        "hours": round(float(sub["duration_s"].sum()) / 3600, 3),
        "mean_duration_s": round(float(sub["duration_s"].mean()), 2),
        "mean_cer_final": round(float(sub["cer_final"].mean()), 4),
        "mean_rms_p95": round(float(sub["rms_p95"].mean()), 4),
        "mean_centroid_hz": round(float(sub["centroid_hz"].mean()), 1),
        "references_used": int(sub["vc_ref_path"].nunique()),
    }
with open(os.path.join(OUT, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

per_emo_table = "\n".join(
    f"| {e} | {stats['per_emotion'][e]['clips']} | {stats['per_emotion'][e]['hours']} | "
    f"{stats['per_emotion'][e].get('mean_duration_s', 0)} | "
    f"{stats['per_emotion'][e].get('mean_cer_final', 0)} | "
    f"{stats['per_emotion'][e].get('references_used', 0)} |"
    for e in EMOTIONS
)

card = f"""# Emotional Nepali TTS dataset — speaker "Amrita"

Synthetic, single-speaker, emotion-labelled Nepali speech built to finetune
[`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts)
for emotional delivery. **{stats['total_hours']} hours / {stats['total_clips']} clips**
across {len(EMOTIONS)} emotions, one consistent female voice.

## Contents

| emotion | clips | hours | mean dur (s) | mean CER | refs used |
|---|---|---|---|---|---|
{per_emo_table}

Splits are on `sentence_id`, not rows: ~20% of sentences appear in all five
emotions as contrastive minimal pairs, so a row-level split would leak the same
text into validation.

## How it was made

1. **Text** — Nepali sentences from the project text pool, filtered for length
   (6–24 words) and for the pool's mangled rows (orphan single characters,
   words truncated at a halant).
2. **Source speech** — Microsoft Edge TTS, voice `ne-NP-HemkalaNeural` (female).
   Per-emotion prosody is injected here via Edge's `rate`/`pitch`/`volume`.
3. **Voice conversion** — Seed-VC v1 re-renders each clip against a
   register-screened Gemini TTS reference clip of the matching emotion, giving
   one consistent timbre ("Amrita") across the whole set.
4. **QC** — faster-whisper `large-v3-turbo` transcribes both the pre- and
   post-conversion audio; rows are gated on the CER *delta*.

### Why prosody is injected at the Edge stage

Seed-VC takes F0 contour and timing from the **source** and timbre, energy and
register from the **reference** (measured: contour r≈0.70 to source vs r≈0.15
to reference; output pause fraction completely invariant to the reference).
With `length_adjust=1.0` every emotion came out byte-identical in duration to
its source, which made every "fast"/"slow" word in the captions false. Edge's
`rate` resynthesises at the new tempo rather than stretching a mel, so it is
the clean lever. Measured pass-through: duration ~1:1.

`pitch` is a *weak* lever — Seed-VC renormalises absolute F0 to the reference,
collapsing a ~9-semitone source swing to ~1.1 semitones at the output. It is
set per emotion anyway because it widens contour variance, but **pace and
reference voice-quality do the real work.**

### Why the references are register-constrained

An earlier reference set had Gemini shouting an octave above the speaker's
natural register when prompted for "angry" and "happy" — median F0 407/420 Hz
vs 206 Hz for neutral, with under 2.3% of spectral energy below 250 Hz. That
broke speaker identity (CAM++ cross-emotion cosine 0.529 against a
same-speaker threshold of ~0.562 — effectively four different speakers) and
made angry and happy acoustically identical. Every reference prompt now pins
the register explicitly, and a screening gate rejects any clip outside
162–257 Hz, below 0.61 cosine to the neutral centroid, under 60% speech, or
truncated mid-decay.

### Why the text is mostly emotion-neutral

Parler must learn emotion from the **caption**, not from text semantics. A set
built only from emotion-congruent sentences would teach it to read sentiment
off the words and ignore the description at inference. ~20% of sentences are
rendered in all five emotions as contrastive pairs where the caption is the
only thing that varies.

## Quality gate

- keep when `cer_final - cer_source <= 0.15`, `cer_final <= 0.55`, and
  `devanagari_ratio >= 0.50`
- `tier == "A"` additionally requires `cer_delta <= 0.08` and `cer_final <= 0.32`
- tier counts: {stats['tier_counts']}
- CER final: mean {stats['cer_final']['mean']}, median {stats['cer_final']['median']}, p90 {stats['cer_final']['p90']}
- CER delta: mean {stats['cer_delta']['mean']}, median {stats['cer_delta']['median']}, p90 {stats['cer_delta']['p90']}

**Absolute CER is not a quality score here.** Clean Edge-TTS Nepali already
scores ~0.25 against this ASR service, so absolute CER mostly measures
Whisper's weakness on Nepali. One clip scored 0.453 purely because Whisper
wrote `1,47,000` where the text said `एक लाख सैंतालीस हजार`. The delta
between pre- and post-conversion CER cancels that constant bias; the
`devanagari_ratio` field catches Whisper's occasional romanised decodes, which
produced bogus CER≈1.0 on perfectly good audio.

## Using it with Parler

`text` is the transcript, `description` is the caption. Every caption begins
with **Amrita** — the Nepali speaker token in the pretrained checkpoint. Drop
that name and the voice changes between runs. Three paraphrase variants exist
per emotion (`description_variant` 0/1/2) so the model does not memorise one
exact caption string.

## Known limitations

- **Synthetic throughout.** Source is a cloud TTS voice, timbre comes from
  another TTS voice via conversion. No human recordings anywhere in the chain.
- **Emotion is carried by pace, energy and timbre, not pitch level.** All five
  emotions share ~one register by design, because that is what keeps the
  speaker identity stable.
- **Few references per emotion.** See `references_used` above. Gemini free-tier
  quota capped reference generation. References supply only timbre and
  register (contour comes from the source, and sources are tens of thousands of
  distinct sentences), but per-reference artefacts could still be memorised.
- **ASR-gated, not human-listened.** The gate catches content damage, not
  whether a clip sounds convincingly emotional. Spot-check before trusting.
- Sample rate is {int(df['sr'].iloc[0])} Hz; Parler's DAC operates at 44.1 kHz, so the
  trainer will resample.

Generated {pd.Timestamp.now().strftime('%Y-%m-%d')}. Manifest: `{MANIFEST}`.
"""
with open(os.path.join(OUT, "DATASET_CARD.md"), "w") as f:
    f.write(card)

print(f"wrote {OUT}")
print(f"  {stats['total_clips']} clips, {stats['total_hours']}h "
      f"(train {len(train)} / val {len(val)}), tier filter {args.tier}")
for e in EMOTIONS:
    s = stats["per_emotion"][e]
    print(f"  {e:8s} {s['clips']:6d} clips  {s['hours']:6.2f}h")
