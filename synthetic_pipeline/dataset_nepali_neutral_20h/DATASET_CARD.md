# Emotional Nepali TTS dataset — speaker "Amrita"

Synthetic, single-speaker, emotion-labelled Nepali speech built to finetune
[`ai4bharat/indic-parler-tts`](https://huggingface.co/ai4bharat/indic-parler-tts)
for emotional delivery. **20.062 hours / 9949 clips**
across 1 emotions, one consistent female voice.

## Contents

| emotion | clips | hours | mean dur (s) | mean CER | refs used |
|---|---|---|---|---|---|
| neutral | 9949 | 20.062 | 7.26 | 0.2717 | 6 |

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
- tier counts: {'A': 6362, 'B': 3587}
- CER final: mean 0.2717, median 0.2597, p90 0.3929
- CER delta: mean 0.0292, median 0.0328, p90 0.1034

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
- Sample rate is 24000 Hz; Parler's DAC operates at 44.1 kHz, so the
  trainer will resample.

Generated 2026-09-04. Manifest: `/root/tts/TTS_training/synthetic_pipeline/manifests/emotion_production.jsonl`.
