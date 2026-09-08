# Reference-clip audit — `emotions/dataset_gemini_{angry,sad,happy,neutral}`

25 Gemini TTS clips (`gemini-3.1-flash-tts-preview`, voice `Zephyr`, prompted "Nepali female
speaker named Amrita"), 24 kHz mono PCM_16, 4.36–11.76 s. Audited as candidate **Seed-VC
reference** clips. Nothing under `emotions/` was modified.

---

## Verdict up front

**It is not one consistent voice.** The four emotion folders split into two voices that a
speaker-verification system reads as different people:

| group | median F0 | offset vs neutral | CAM++ cohesion |
|---|---|---|---|
| `neutral` (6 clips) | 204 Hz | — | 0.764 |
| `sad` (7 clips) | 283 Hz | **+5.7 st** | 0.678 |
| `angry` (5 clips) | 413 Hz | **+12.2 st** | 0.655 |
| `happy` (7 clips) | 429 Hz | **+12.9 st** | 0.697 |

`angry` and `happy` sit **almost exactly one octave above** `neutral`. Median 413–421 Hz is
outside the adult-female speaking range (typically 165–255 Hz) — it is child / falsetto /
shriek register. `angry`×`neutral` = 0.475 and `angry`×`sad` = 0.441 in CAM++ cosine, both
**below the 0.562 same-speaker threshold I measured on this exact data** (calibration below).

I confirmed this is real and not an F0 estimation artifact, and I confirmed it propagates
into converted output. Details in §1–§4.

Recommended action: **use `neutral` as the anchor identity and regenerate `angry`/`happy`
(and preferably `sad`) with an explicit pitch/register constraint in the prompt.** Do not run
the large conversion job on this reference set as it stands.

---

## 1. Voice consistency (speaker embeddings)

### Encoders used

1. **CAM++** — `campplus_cn_common.bin` (192-d), loaded from
   `vc_service/seed-vc/campplus_cn_common.bin` with `modules.campplus.DTDNN.CAMPPlus`,
   feature pipeline copied verbatim from `vc_service/server.py:238-243`
   (`torchaudio.compliance.kaldi.fbank`, 80 mel bins, `dither=0`, 16 kHz, per-utterance mean
   subtraction). Run on CPU. **This is the encoder Seed-VC actually conditions on**, so its
   verdict is operationally binding, not just diagnostic.
2. **Resemblyzer** 0.1.4 (GE2E, 256-d), already present in `vc_service/.venv` — independent
   cross-check, different architecture, different training data (English VoxCeleb/LibriSpeech
   vs CAM++'s Chinese CN-Celeb).

### Where the same-speaker threshold sits

The published ModelScope/3D-Speaker card for this checkpoint does not ship a threshold in
`infer_sv.py` (I checked), so I calibrated empirically **on this project's own domain** —
Nepali, 5–11 s clips, same encoder, same feature pipeline:

- 160 clips from `manifests/reference_pool_v2/` → only **22 survive dedup at cos > 0.995**
  (the pool is heavily duplicated — a separate finding, see §5).
- Agglomerative clustering (average linkage, distance cut 0.45) → 10 pseudo-speakers.

| CAM++ pair type | n | mean | spread |
|---|---|---|---|
| same pseudo-speaker (real Nepali pool) | 19 | 0.748 | p5 = 0.607, min = 0.569 |
| different pseudo-speaker (real Nepali pool) | 212 | 0.296 | p95 = 0.544, max = 0.604 |
| Edge-TTS `ne-NP-HemkalaNeural`, known identical voice | 45 | **0.937** | min 0.879 |
| Gemini clips × real pool (known different) | 1000 | 0.046 | max 0.431 |
| Gemini clips × Edge-TTS (known different) | 250 | 0.261 | max 0.440 |

**EER threshold ≈ 0.562** (FAR 2.4 %, FRR 0.0 %). Read 0.56 as the decision line, 0.61+ as
confidently same speaker, below 0.54 as different-speaker territory.

*Caveat, stated plainly:* this calibration rests on 19 same-speaker pairs with cluster-derived
pseudo-labels, which is partly circular. Treat 0.562 as ±0.05. It does not change any
conclusion below — the failing cross-emotion pairs are at 0.44–0.50, clear of that band.

### CAM++ pairwise results (25 clips, 300 pairs)

| | mean | min |
|---|---|---|
| **all within-emotion pairs** (n=67) | **0.700** | 0.521 |
| **all cross-emotion pairs** (n=233) | **0.529** | 0.282 |

| pair | n | mean | min | verdict @ 0.562 |
|---|---|---|---|---|
| angry × happy | 35 | 0.657 | 0.482 | same speaker |
| happy × neutral | 42 | 0.558 | 0.407 | **borderline / fail** |
| happy × sad | 49 | 0.531 | 0.385 | **different speaker** |
| neutral × sad | 42 | 0.501 | 0.355 | **different speaker** |
| angry × neutral | 30 | 0.475 | 0.296 | **different speaker** |
| angry × sad | 35 | 0.441 | 0.282 | **different speaker** |

**154 of 300 pairs (51.3 %) fall below the same-speaker threshold** — 150 of them
cross-emotion, only 4 within-emotion. The failure is structured exactly along the emotion-folder
boundary. Within-emotion cohesion is fine everywhere (0.655–0.764, all above threshold).

Worst single pair: `angry_002` × `sad_005` at **0.282** — that is *lower* than Gemini-vs-Edge-TTS
(0.44 max), i.e. those two clips look less like the same person than one of these clips does
compared to a completely unrelated TTS voice.

### Resemblyzer cross-check — same shape, milder magnitude

| | mean | min |
|---|---|---|
| within-emotion (n=67) | 0.863 | 0.758 |
| cross-emotion (n=233) | 0.770 | 0.653 |
| Gemini × Edge-TTS (different speaker) | 0.610 | — |
| Gemini × real pool (different speaker) | 0.532 | — |
| Edge-TTS × Edge-TTS (same voice) | 0.944 | 0.874 |

Resemblyzer agrees on **direction and ranking** (`angry`/`happy` cluster together; `neutral`
is the outgroup; angry×neutral 0.744 and happy×neutral 0.743 are the two lowest cross-emotion
means) but is **less alarmed**: cross-emotion 0.770 still sits well above its different-speaker
region (~0.53–0.61). GE2E is known to be more prosody-invariant than CAM++.

**How to reconcile the two:** the honest reading is *same vocal tract, radically different
phonation register*, not two anatomically different speakers. Supporting evidence for the
"same vocal tract" half:

| vocal-tract cue (F0-independent-ish) | angry | happy | neutral | sad | spread |
|---|---|---|---|---|---|
| F4 (LPC, 11 kHz resample, order 12) | 3210 Hz | 3202 Hz | 3338 Hz | 3321 Hz | **4 %** |
| F3 | 2279 | 2138 | 2268 | 2415 | 12 % |

F4 barely moves — vocal-tract length is consistent. F1 does move a lot (509→421→381 Hz
angry→neutral→sad) but that is the expected consequence of loud/high phonation with a raised
larynx, not of a different speaker. *Caveat:* LPC formant tracking is unreliable when
F0 > 350 Hz (harmonics spaced 400–500 Hz undersample the envelope), so the angry/happy formant
numbers carry real error bars. F4 is the most robust of them.

**But that reconciliation does not rescue the reference set**, because Seed-VC conditions on the
CAM++ vector, and §4 shows the divergence survives conversion intact.

### Two hypotheses I tested and ruled out

**(a) Silence/breath in the reference is diluting the embedding.** `server.py` runs
`kaldi.fbank` over the *entire* reference with no VAD or trimming, and the `sad` clips are only
44–66 % speech — so this was a plausible cheap fix. Recomputing every embedding with silence
stripped (`librosa.effects.split`, `top_db=30`):

| | raw (as `server.py` does it) | silence-stripped |
|---|---|---|
| within-emotion mean | 0.700 | 0.672 |
| cross-emotion mean | 0.529 | 0.522 |
| angry × neutral | 0.475 | 0.448 |

No improvement — slightly worse. **The drift is in the voiced speech itself, not in the
pauses.** (Still worth knowing that `server.py` does no reference VAD; see §5.)

**(b) It's a pyin octave error.** Ruled out with five estimators (§2).

---

## 2. Are the emotions acoustically distinct?

### F0 verification (because the whole finding hangs on it)

My first pass used `fmax=500`, and **every angry and happy clip pinned the ceiling** — the
measurement itself was truncated. Re-run at `fmax=900` and cross-checked with five independent
estimators on strongly-voiced 40 ms frames (medians, Hz):

| clip | pyin | yin | autocorr | cepstrum |
|---|---|---|---|---|
| `angry_003` | 516 | 489 | 511 | 511 |
| `happy_008` | 518 | 498 | 533 | 522 |
| `neutral_004` | 192 | 197 | 218 | 220 |
| `sad_006` | 229 | 234 | 240 | 239 |

Four algorithms agree to within ~5 % on every clip. (A fifth, harmonic-peak spacing, returned a
constant ~160–176 Hz for all 25 — my peak-distance parameter was wrong; that estimator is
discarded, not evidence.) Additional disconfirmation of the octave-error hypothesis: no angry or
happy clip ever dips below 169 Hz anywhere in the utterance, whereas a genuine ~200 Hz voice
would touch 120–150 Hz at phrase ends, as all six `neutral` clips do. **The F0 is real.**

### Per-emotion aggregates (mean ± sd)

| metric | angry (5) | happy (7) | neutral (6) | sad (7) |
|---|---|---|---|---|
| duration (s) | 6.52 ± 0.89 | 7.39 ± 1.11 | 5.78 ± 0.73 | 9.27 ± 1.81 |
| **F0 median (Hz)** | **412 ± 67** | **421 ± 68** | **210 ± 22** | **280 ± 36** |
| F0 mean (Hz) | 421 ± 60 | 435 ± 50 | 214 ± 11 | 290 ± 40 |
| F0 sd (Hz) | 97 ± 11 | 104 ± 20 | 54 ± 9 | 80 ± 43 |
| F0 sd (semitones) | 3.87 ± 0.43 | 4.19 ± 0.88 | 4.36 ± 0.61 | 4.44 ± 1.99 |
| F0 p5–p95 range (st) | 12.96 ± 2.11 | 13.46 ± 2.17 | 13.67 ± 1.55 | 15.05 ± 7.59 |
| F0 max (Hz) | 716 ± 101 | 751 ± 99 | 366 ± 29 | 583 ± 208 |
| frac. frames > 500 Hz | 0.22 ± 0.17 | 0.28 ± 0.17 | **0.00 ± 0.00** | 0.03 ± 0.04 |
| RMS (dBFS, whole file) | −16.4 ± 1.5 | −18.5 ± 1.6 | −19.7 ± 1.4 | −23.6 ± 1.5 |
| speech-median level (dB) | −22.3 ± 2.1 | −25.9 ± 1.7 | −24.2 ± 1.4 | −29.0 ± 2.1 |
| pause fraction | 0.263 ± 0.041 | 0.235 ± 0.054 | 0.255 ± 0.046 | **0.448 ± 0.067** |
| syllable rate (peaks/s, whole file) | 4.36 ± 0.34 | 4.85 ± 0.28 | 4.91 ± 0.48 | **3.14 ± 0.47** |
| syllable rate (per second of speech) | 5.92 ± 0.32 | 6.36 ± 0.29 | 6.58 ± 0.37 | 5.68 ± 0.41 |
| spectral centroid (Hz) | 2461 ± 391 | 2768 ± 217 | 2275 ± 203 | 2140 ± 305 |
| spectral tilt (dB/kHz, 0.1–8 k) | −3.61 ± 0.82 | −3.30 ± 0.64 | −2.54 ± 0.32 | −2.91 ± 0.53 |
| 2–8 k / 0–1 k ratio (dB) | −13.1 ± 2.7 | −13.1 ± 3.5 | −17.9 ± 1.7 | −18.1 ± 3.2 |
| F4 (Hz) | 3210 ± 93 | 3202 ± 92 | 3338 ± 80 | 3321 ± 96 |

Median-F0 offsets vs `neutral`: **angry +12.24 st, happy +12.89 st, sad +5.70 st.**

### Do they separate?

**Three of the four separate; angry and happy do not.**

| comparison | separates? | evidence |
|---|---|---|
| neutral vs everything | **yes, cleanly** | F0 210 Hz vs 280–421 Hz; zero frames > 500 Hz; CAM++ neutral cohesion 0.764 is the tightest of the four |
| sad vs everything | **yes, cleanly** | pause fraction 0.448 vs 0.235–0.263 (non-overlapping ranges); syllable rate 3.14/s vs 4.36–4.91/s (non-overlapping); level −29.0 dB vs −22 to −26 dB; duration 9.3 s vs 5.8–7.4 s |
| **angry vs happy** | **NO** | F0 median 412 vs 421 Hz; F0 sd 97 vs 104 Hz; pause 0.263 vs 0.235; rate 5.92 vs 6.36 syl/s-speech; 2–8 k/0–1 k ratio −13.1 vs −13.1 dB (identical); F4 3210 vs 3202 Hz (identical). **CAM++ angry×happy = 0.657, higher than several within-emotion pairs.** MFCC(c1–c13) centroid distance angry↔happy = **14.0** vs angry↔neutral = 57.7; the within-emotion MFCC spread is 22.6–27.6, i.e. **angry and happy centroids are closer to each other than a typical clip is to its own folder's centroid.** LTAS log-spectrum correlation angry↔happy = 0.961. |

The only measure that puts any daylight between angry and happy is spectral centroid
(2461 vs 2768 Hz, overlapping ± 1 sd) — and that is confounded by text, since the happy set
contains English fricative-heavy loanwords (§3).

**Plainly: the angry and happy prompts produced the same thing — loud high-pitched excited
speech.** As references they cannot teach a model to distinguish anger from joy. `neutral`
and `sad` are genuinely and separably rendered.

Secondary point worth flagging: `neutral` and `sad` have *larger* F0 range in semitones
(13.7, 15.1 st) than `angry` and `happy` (13.0, 13.5 st). The high-arousal clips are
high-pitched but relatively *compressed* in relative pitch movement — they are shouted at a
plateau, not expressively contoured. Emotion here is encoded as register, not as intonation.

---

## 3. Per-clip technical inspection

Columns: `dur` s; `F0med`/`F0sd`/`F0p5`/`F0p95` Hz (pyin, fmin 70 / fmax 900, voiced+speech
frames only); `rng` = p95/p5 in semitones; `>500` = fraction of voiced frames above 500 Hz;
`RMS` dBFS; `pause` = fraction of frames > 35 dB below peak frame; `cent` = spectral centroid Hz;
`tilt` dB/kHz; `syl/s` energy-envelope peaks per second; `F4` Hz; `art` = count of non-speech
above-noise-floor segments ≥ 80 ms (breath/sniff candidates).

| clip | dur | F0med | F0sd | F0p5 | F0p95 | rng | >500 | RMS | pause | cent | tilt | syl/s | F4 | art |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `angry_001` | 6.32 | 452 | 98 | 335 | 725 | 13.4 | 0.23 | -17.8 | 0.25 | 2678 | -2.25 | 4.11 | 3334 | 2 |
| `angry_002` | 5.08 | 414 | 96 | 300 | 554 | 10.6 | 0.19 | -15.4 | 0.30 | 2475 | -3.64 | 3.94 | 3095 | 3 |
| `angry_003` | 7.08 | 511 | 102 | 351 | 666 | 11.1 | 0.53 | -14.6 | 0.21 | 3015 | -3.29 | 4.80 | 3278 | 1 |
| `angry_004` | 6.36 | 318 | 77 | 222 | 473 | 13.2 | 0.03 | -18.5 | 0.24 | 1841 | -4.30 | 4.72 | 3230 | 1 |
| `angry_005` | 7.76 | 367 | 111 | 247 | 643 | 16.6 | 0.14 | -15.6 | 0.32 | 2294 | -4.58 | 4.25 | 3114 | 3 |
| `happy_001` | 7.72 | 355 | 149 | 248 | 654 | 16.8 | 0.23 | -17.7 | 0.21 | 2902 | -3.48 | 4.92 | 3189 | 2 |
| `happy_002` | 7.76 | 499 | 84 | 347 | 621 | 10.1 | 0.48 | -17.4 | 0.23 | 2921 | -3.38 | 4.90 | 3310 | 1 |
| `happy_003` | 7.96 | 429 | 93 | 266 | 590 | 13.8 | 0.23 | -17.0 | 0.19 | 2409 | -2.77 | 4.90 | 3245 | 1 |
| `happy_004` | 7.72 | 360 | 105 | 227 | 559 | 15.6 | 0.12 | -22.2 | 0.36 | 2706 | -2.26 | 4.27 | 3189 | 2 |
| `happy_007` | 8.88 | 442 | 91 | 351 | 666 | 11.1 | 0.25 | -17.6 | 0.20 | 2682 | -3.37 | 4.73 | 3322 | 1 |
| `happy_008` | 5.16 | 526 | 106 | 308 | 652 | 13.0 | 0.58 | -18.5 | 0.23 | 3127 | -3.33 | 5.23 | 3110 | 3 |
| `happy_009` | 6.56 | 337 | 100 | 244 | 543 | 13.8 | 0.09 | -18.9 | 0.22 | 2632 | -4.52 | 5.03 | 3051 | 2 |
| `neutral_001` | 5.36 | 254 | 69 | 132 | 337 | 16.2 | 0.00 | -21.8 | 0.31 | 2368 | -2.53 | 4.66 | 3389 | 1 |
| `neutral_002` | 6.56 | 199 | 57 | 148 | 322 | 13.4 | 0.00 | -20.9 | 0.25 | 2476 | -2.52 | 4.57 | 3445 | 1 |
| `neutral_003` | 6.16 | 220 | 48 | 145 | 298 | 12.5 | 0.00 | -18.4 | 0.22 | 2376 | -2.16 | 5.52 | 3407 | 0 |
| `neutral_004` | 6.20 | 191 | 61 | 141 | 342 | 15.4 | 0.00 | -17.6 | 0.32 | 1858 | -3.01 | 4.19 | 3266 | 1 |
| `neutral_005` | 6.04 | 209 | 46 | 140 | 283 | 12.3 | 0.00 | -19.9 | 0.21 | 2203 | -2.84 | 5.46 | 3224 | 2 |
| `neutral_006` | 4.36 | 188 | 46 | 145 | 295 | 12.3 | 0.00 | -19.4 | 0.22 | 2370 | -2.16 | 5.05 | 3300 | 1 |
| `sad_001` | 10.68 | 250 | 24 | 225 | 301 | 5.0 | 0.00 | -25.0 | 0.52 | 2219 | -2.21 | 2.53 | 3370 | 7 |
| `sad_002` | 7.72 | 353 | 97 | 156 | 490 | 19.9 | 0.04 | -22.3 | 0.46 | 2167 | -3.64 | 2.85 | 3253 | 4 |
| `sad_003` | 11.76 | 283 | 163 | 198 | 822 | 24.6 | 0.12 | -24.0 | 0.54 | 2433 | -2.28 | 2.64 | 3333 | 6 |
| `sad_004` | 7.04 | 300 | 70 | 234 | 387 | 8.7 | 0.01 | -24.1 | 0.38 | 2590 | -2.69 | 3.69 | 3271 | 1 |
| `sad_005` | 9.04 | 263 | 44 | 214 | 361 | 9.1 | 0.00 | -24.0 | 0.42 | 1966 | -3.43 | 3.76 | 3292 | 2 |
| `sad_006` | 11.24 | 231 | 53 | 164 | 341 | 12.7 | 0.00 | -25.3 | 0.48 | 1580 | -3.37 | 3.02 | 3522 | 6 |
| `sad_007` | 7.40 | 283 | 109 | 114 | 495 | 25.4 | 0.03 | -20.6 | 0.34 | 2021 | -2.77 | 3.51 | 3203 | 2 |
### Global health checks — all 25 clips pass

| check | result |
|---|---|
| **Clipping** | **None.** Zero int16 samples at ±32767; zero above ±32000; zero runs of ≥3 samples above ±32700. Peak amplitude 0.517–0.876 (angry loudest, sad quietest). Comfortable headroom throughout. |
| **DC offset** | **Negligible.** Max \|mean\| = 1.76e-4 (`angry_005`) = −75 dBFS. Typical 1e-6 to 1e-4. No removal needed. |
| **Format** | Uniform: 24000 Hz, 1 channel, PCM_16. No resampling artifacts; energy present up to 12 kHz in all clips (−17 to −41 dB re: total in the 10–12 kHz band), so nothing is a low-bandwidth upsample. |
| **Silence padding** | Leading 0.12–0.44 s, trailing 0.00–1.08 s. Only `sad_003` (1.08 s trailing) is excessive. |
| **Language** | ASR returns `language: "ne"` for all 25. No wrong-language clip. |
| **Gender** | All female-consistent. `neutral` at 204 Hz median is textbook adult female. `angry`/`happy` at 413–421 Hz is above the female range but in the direction of *higher*, i.e. child/falsetto, never male. No wrong-gender clip. |
| **Voicing/creak** | `f0_min` hits the analysis floor in `neutral` (5/6 clips reach 117–131 Hz) and in `sad_002`/`sad_005`/`sad_007` (106–146 Hz), consistent with mild phrase-final creak. Frame-to-frame F0 perturbation ("jitter" proxy) is 0.021–0.053 across all clips with no emotion pattern — `sad_003` (0.053) is the most perturbed. Nothing pathological. |

### Truncation

| clip | last sample | last 40 ms re: speech median | trailing silence | verdict |
|---|---|---|---|---|
| **`angry_004`** | **−1141** | **−12.3 dB** | **0.000 s** | **Truncated — cut off mid-decay while still phonating.** Only clip that fails. |
| `angry_001` | 19 | −21.3 dB | 0.000 s | Abrupt but the tail is 21 dB down; ends on a decaying consonant. Marginal — usable. |
| all other 23 | 0 to ±25 | −40.6 to −223 dB | 0.17–1.08 s | Clean. |

No clip is truncated at the *start* — leading 40 ms is 30–57 dB below speech level everywhere.

### Non-speech artifacts — the sad prompt's "sniffling and trembling"

The sad prompt explicitly requested sniffling and trembling. **It did take effect, and it does
show up as sound events.** Counting non-speech segments ≥ 80 ms that sit above the noise floor
but below the speech threshold:

| emotion | mean artifact segments/clip | worst clips |
|---|---|---|
| angry | 2.0 | `angry_002`, `angry_005` (3) |
| happy | 1.7 | `happy_008`, `happy_009` (3) |
| neutral | 1.0 | `neutral_005` (2) |
| **sad** | **4.0** | **`sad_001` (7), `sad_003` (6), `sad_006` (6)** |

Their character (spectral distribution, voicing):

- `sad_001` @ 1.80–2.12 s — 0.32 s, −29.8 dB re: speech, **0 % voiced**, energy 43 % in 1–4 kHz.
- `sad_001` @ 0.29–0.40 s and 0.44–0.56 s — two 0.11–0.12 s unvoiced bursts, 26–36 % of energy
  above 4 kHz. This is the sniff/inhale signature.
- `sad_003` @ 1.02–1.27 s — 0.24 s, 41 % of energy above 4 kHz, unvoiced.
- `sad_006` @ 0.84–1.04 s and 4.96–5.18 s — 0.20–0.22 s unvoiced, mid-band dominant.

**Assessment: these are breaths and sniffs, not overlaid sound effects.** They sit 26–50 dB
below the speech level, so they will not dominate a reference. But two things follow:

1. They inflate the pause fraction to 0.45 and mean the `sad` clips are **only 44–66 % speech**
   (`sad_001` 44.2 %, `sad_003` 48.3 %, `sad_002` 50.8 %, `sad_006` 52.1 %). Since
   `server.py` computes the CAM++ fbank over the *whole* reference with no VAD, roughly half
   of what defines the `sad` style vector is not voice.
2. Stripping them does **not** fix the cross-emotion drift (§1, hypothesis (a)) — so they are a
   quality issue, not the cause of the identity problem.

### Language and accent

All 25 decode as Nepali. Two happy clips are **code-switched into English**, which is a real
concern for a timbre reference because English phonation and vowel space differ:

- `happy_007`: "**ओ माइ गौड**" (oh my god) — English interjection, sentence-initial.
- `happy_008`: "**येस, फाइनली** … **सेलिबरेशन**" (yes, finally … celebration) — three English
  loanwords in a 5.16 s clip, and it is the highest-F0 clip in the set (525 Hz median, 58 % of
  frames above 500 Hz).

I cannot assess *audible English accent on the Nepali* by measurement — that needs a native
listener, and I am flagging it as unresolved rather than guessing. What I can say: the ASR
transcripts of the `neutral` set are clean, well-formed, idiomatic Nepali (formal announcement
register), whereas the `angry`, `happy` and `sad` transcripts are noticeably more garbled. That
is at least partly the ASR struggling with 400–500 Hz shouted speech rather than proof of bad
pronunciation — but `happy_002` ("अरे बाफ रहे … मेरो ता हात खुट्टाने कापी राय चाखो सिले") and
`sad_003` ("सब्वैकुरा समाप्तब अवो फिरीपवईले जोस्तो कीपणी उने चाहिना") are garbled enough that
mispronunciation cannot be ruled out. **Recommend a native-speaker listening pass on the
angry/happy/sad sets before use.**

---

## 4. Downstream confirmation — the drift survives conversion

The preceding sections are about the references. This section tests the thing that actually
matters: **does the inconsistency reach the output dataset?** I sent 14 read-only `/convert`
requests to the already-running Seed-VC replica on port 8032 (GPU was idle at 0 % util; no
service was restarted, no file under `emotions/` touched): 2 flat Edge-TTS source clips ×
7 references spanning all four emotions.

### The same source text, converted with different-emotion references

| reference used | output median F0 (`row0`) | output median F0 (`row3`) |
|---|---|---|
| `neutral_002` | 213 Hz | 230 Hz |
| `neutral_004` | 217 Hz | 222 Hz |
| `sad_001` | 261 Hz | 260 Hz |
| `sad_004` | 295 Hz | 313 Hz |
| `angry_005` | 432 Hz | 432 Hz |
| `happy_007` | 511 Hz | 499 Hz |
| `angry_003` | **544 Hz** | **560 Hz** |

Source clips are 216 Hz and 215 Hz. **A 16-semitone spread in the output from identical input
text, driven only by which reference was picked.** CAM++ cosine *between the outputs*:

| | mean | min |
|---|---|---|
| outputs from same-emotion references | 0.790 / 0.759 | — |
| outputs from different-emotion references | **0.622 / 0.579** | **0.468** |

Several output pairs land below the 0.562 same-speaker line. **The finished dataset would
contain audio that a speaker-verification system splits into multiple speakers, all labelled
"Amrita" — exactly the failure mode this audit was commissioned to catch.**

Seed-VC is not smoothing the problem out; it transfers reference identity faithfully:

| | value |
|---|---|
| output ↔ reference CAM++ cosine | **0.787–0.909** (target identity is hit) |
| output ↔ source CAM++ cosine | 0.119–0.332 (little source leakage) |

### What the reference actually controls — and what it does not

This is the most consequential secondary finding.

| property | comes from |
|---|---|
| timbre / speaker identity | **reference** (0.79–0.91 cosine to it) |
| pitch register | **reference** (213 → 560 Hz, tracks the reference) |
| overall level | **reference** (sad refs → quietest outputs, −20 to −21 dB) |
| **pause structure / timing** | **SOURCE** — output pause fraction is 0.164–0.208 for `row0` and 0.268–0.293 for `row3` **regardless of reference**. The `sad` references' defining 0.448 pause fraction does **not** transfer. |
| **intonation contour shape** | **SOURCE** — F0-contour Pearson r(output, source) = 0.33–0.87 (mean ≈ 0.70) vs r(output, reference) = **−0.37 to +0.62 (mean ≈ +0.15)**. |

So the two features that most cleanly distinguish `sad` from the rest — slow rate and long
pauses — are the two that Seed-VC discards. What survives into the "sad" partition of the
dataset is just *quieter and slightly higher-pitched*. Meanwhile "angry" and "happy" are
already indistinguishable from each other in the references (§2) and remain so in the output
(CAM++ 0.807–0.895 between angry- and happy-referenced outputs).

**Net effect if you ship this: an emotion-labelled dataset whose labels encode pitch height and
loudness, not emotion — and whose speaker identity varies with the label.**

---

## 5. Per-clip keep / drop recommendation

Two decisions are separable, and I recommend them separately.

**Decision A — which folders can coexist as one speaker.** `neutral` is the anchor: it has the
tightest internal cohesion (0.764), a normal adult-female F0 (204 Hz, matching the 215 Hz
Edge-TTS source), zero frames above 500 Hz, the cleanest transcripts, and no truncation.
`sad` at +5.7 st is recoverable. `angry` and `happy` at +12–13 st are not the same voice and
cannot be reconciled by any post-processing I would trust.

**Decision B — individual clip quality.** Scored against the `neutral` centroid (the anchor
identity) and against each clip's own folder:

| clip | CAM++ → neutral anchor | CAM++ → own folder | Resemblyzer → neutral | recommendation | reason |
|---|---|---|---|---|---|
| `angry_001` | 0.483 | 0.626 | 0.745 | **DROP** | F0 452 Hz (+13.8 st); anchor 0.483; abrupt end (tail −21 dB, 0 s trailing silence) |
| `angry_002` | 0.363 | 0.633 | 0.708 | **DROP** | F0 414 Hz; **worst clip in the set** — anchor 0.363, reads as a different speaker vs 17/24 others, min pair 0.282; shortest angry clip (5.08 s) |
| `angry_003` | 0.520 | 0.609 | 0.728 | **DROP** | F0 511 Hz, 53% of frames >500 Hz — extreme register; anchor 0.520 |
| `angry_004` | 0.549 | 0.686 | 0.765 | **DROP** | **truncated mid-decay** (last 40 ms only −12.3 dB below speech, last sample −1141); highest-anchor angry clip (0.549) so worth regenerating this text |
| `angry_005` | 0.460 | 0.720 | 0.774 | **DROP** | F0 367 Hz; anchor 0.460 |
| `happy_001` | 0.595 | 0.752 | 0.805 | **DROP** (register) | best-behaved happy clip technically — anchor 0.595, own-folder 0.752, clean — but F0 355 Hz (+9.6 st) still off-anchor |
| `happy_002` | 0.527 | 0.681 | 0.720 | **DROP** | F0 499 Hz, 48% >500 Hz; anchor 0.527; transcript badly garbled |
| `happy_003` | 0.561 | 0.704 | 0.761 | **DROP** (register) | F0 429 Hz; anchor 0.561; otherwise clean |
| `happy_004` | 0.622 | 0.697 | 0.742 | **DROP** (register) | F0 360 Hz; **highest happy anchor score (0.622)**; quietest happy clip (−22.2 dBFS); worth regenerating this text |
| `happy_007` | 0.595 | 0.771 | 0.733 | **DROP** | F0 442 Hz; English code-switch ("ओ माइ गौड") |
| `happy_008` | 0.440 | 0.650 | 0.709 | **DROP** | F0 525 Hz — highest in the set, 58% >500 Hz; anchor 0.440; three English loanwords in 5.16 s; only 5.16 s long |
| `happy_009` | 0.565 | 0.619 | 0.732 | **DROP** | F0 337 Hz; anchor 0.565; two unvoiced artifact bursts in the first 0.66 s |
| `neutral_001` | 0.759 | 0.759 | 0.862 | **KEEP** | anchor 0.759; clean; formal-announcement register; 5.36 s |
| `neutral_002` | 0.798 | 0.798 | 0.883 | **KEEP** — best in set | anchor 0.798 (highest); clean; 6.56 s; F0 199 Hz |
| `neutral_003` | 0.740 | 0.740 | 0.892 | **KEEP** | anchor 0.740; **zero** artifact segments — the cleanest clip audited |
| `neutral_004` | 0.796 | 0.796 | 0.894 | **KEEP** | anchor 0.796; clean; F0 191 Hz (lowest, most neutral) |
| `neutral_005` | 0.748 | 0.748 | 0.875 | **KEEP** | anchor 0.748; clean; 0.12 s low-level segment at t=0 (harmless) |
| `neutral_006` | 0.741 | 0.741 | 0.886 | **KEEP** (lowest priority) | anchor 0.741; shortest clip in the set (4.36 s) — near the low end for a reliable reference |
| `sad_001` | 0.586 | 0.679 | 0.772 | **KEEP with caveat** | anchor 0.586; **7 breath/sniff segments**, only 44.2% speech; flattest F0 in the set (sd 24 Hz, range 5.0 st) — genuinely subdued |
| `sad_002` | 0.516 | 0.709 | 0.687 | **KEEP with caveat** | anchor 0.516 (below threshold); F0 353 Hz is high for sad; only 50.8% speech; 0.69 s trailing silence |
| `sad_003` | 0.414 | 0.640 | 0.731 | **DROP** | anchor 0.414 — lowest of any sad clip; F0 range 24.6 st and 12% of frames >500 Hz (unstable); 48.3% speech; 6 artifact segments; 1.08 s trailing silence; most perturbed F0 in the set |
| `sad_004` | 0.571 | 0.708 | 0.737 | **KEEP** — best sad | anchor 0.571; 65.7% speech (highest of the sad set); only 1 artifact segment; stable F0 (sd 70 Hz, range 8.7 st) |
| `sad_005` | 0.462 | 0.664 | 0.771 | **DROP** | anchor 0.462; reads as a different speaker vs 18/24 others (joint worst); lowest centroid similarity (0.670) |
| `sad_006` | 0.529 | 0.681 | 0.749 | **KEEP with caveat** | anchor 0.529; quietest clip in the set (−25.3 dBFS, speech median −31.6 dB); 6 artifact segments; 52.1% speech |
| `sad_007` | 0.427 | 0.667 | 0.778 | **DROP** | anchor 0.427; different-speaker vs 18/24 others; lowest centroid similarity in the set (0.657); F0 range 25.4 st (erratic) |

### Summary of recommendation

| | clips | action |
|---|---|---|
| **KEEP** | `neutral_001…006` (6) | Use as the anchor identity. All 6 usable. |
| **KEEP with caveat** | `sad_001`, `sad_002`, `sad_004`, `sad_006` (4) | Usable *only* if you accept a +3 to +6 st register shift on the "sad" partition and pre-trim the breath segments. `sad_004` is the best of them. |
| **DROP** | `sad_003`, `sad_005`, `sad_007` (3) | Off-anchor and/or unstable even relative to their own folder. |
| **DROP — whole folder** | all 5 `angry` + all 7 `happy` (12) | Wrong voice, not merely wrong emotion. `angry_004` additionally truncated. |

That leaves **6 confidently usable references, plus 4 conditional** — from 25.

The two clips I would specifically re-prompt with the same text, because they scored highest
within their folders and only the register let them down, are `happy_004` (anchor 0.622) and
`angry_004` (anchor 0.549 — also the truncated one).

---

## 6. Transcripts (all 25)

Produced by the running ASR QC service, `POST http://localhost:8013/transcribe`, multipart
field `audio` (Whisper `large-v3-turbo`, decoded as Nepali). Every clip returned
`language: "ne"`.

**Read these as ASR output, not ground truth.** Whisper large-v3-turbo is weak on Nepali, and
weaker still on 400–500 Hz shouted speech, so the angry/happy/sad rows contain transcription
errors on top of whatever the audio actually says. The `neutral` rows are clean and idiomatic
and can be trusted. Where you need real ground truth for the angry and happy sets, these should
be corrected by a native speaker — I would not seed a training manifest from the non-neutral
rows as-is.

| clip | transcript |
|---|---|
| `angry_001` | च्छा, तिम्रो यो हरका देखे रह मला घिन लाग छा, मा बकवा सुनन या ब्वसे की होई ना? |
| `angry_002` | चौप लाग! मले कुने इस पश्टिकरन चाहिदे ना अफनो बटोला गिहाँ बाटा? |
| `angry_003` | तो संग इस्तो आसा अलिक दिवनी थिये ना, तो इले मले धोका दीस, मतले कोईले माफ कर दीना |
| `angry_004` | तूरान्तै यहा बाटन निसकी, मलाय तिम्रो अनुआर पनी हेरनु छैना, फेरी कहिले यहा न आउनु। |
| `angry_005` | अरे सुना मला योकुरा अलिकति पनी मन परेना म धेरे रिसा आयको छु तुरुन्त यहाँ बट जाओ |
| `happy_001` | सुन्नाना सुन्नाना तोले पत्याउने गारोंच बर्करे की बयो मतो खुसिले बागले बैसकिए |
| `happy_002` | अरे बाफ रहे मैली सोची की पनी थे नही इस तो उला वने रहा मेरो ता हात खुट्टाने कापी राय चाखो सिले |
| `happy_003` | हामिले जीतियों अन्तत आमरो सपना पूरा भयो मत उफ्री उफ्री रूनमात्र बागी छा |
| `happy_004` | चीटो मलाई काल गौर था? मतलाई सबपई कु रन्ना बनी बस नही सक दीना चीटो गौर? |
| `happy_007` | ओमाइ गौड एर तो यो खबर मत पत्याउना ही सोगी रहीगी छही ना कत्ती देरे खुसी को कुरा |
| `happy_008` | येस, फाइनली काम बढ़ियो, अब तामीली भप्या सेलिबरेशेन गर नहीं पड़छा |
| `happy_009` | मतो खुशीले आस्ते आस्ते सासे फेरना सकी रहिकी छाईना, खत्ती देरे मजा आयो. |
| `neutral_001` | नमस्ते, आजको कारेक्रम्मा यहां हरु सबैलाई हार्दिक स्वागत्छ। |
| `neutral_002` | क्रिपया ध्यान दिनु होला, यो बस केही समय पछी आफनो गन्तव्य तर्फ प्रस्थान गर दैचा। |
| `neutral_003` | नेपाल एउटास सुन्दर र बिवीत संस्कृतिले भर्येको बहु भाशिक देश हो। |
| `neutral_004` | तपाईले पठाउनु भईको इमेल प्राप्त भईयो, म अध्यन गरेर छिट्टै जवावति ने छु। |
| `neutral_005` | काठमान्डो उपत्यकाको मौसम आज दिन भरिनै सामान्य पया सफा रहने छुआ. |
| `neutral_006` | थब जानकारीको लागी हामरो आधिकारिक वेबसाइट हेर नसकनु हुने छा. |
| `sad_001` | आजमलाई एत्ति एधेरै एकलो पन महसुच बहिरा छकी रून पनी गारो बहिरा छ |
| `sad_002` | मलै आफकरी द्यो मलै चाहे रपनी कि एराम्रो गर्नो सकी ना |
| `sad_003` | सब्वैकुरा समाप्तब अवो फिरीपवईले जोस्तो कीपणी उने चाहिना |
| `sad_004` | मेरो मुठो एत्ती धेरे दुखिर आये चाकी म सब तमा प्येप्त गर्ना नही सक्ति ना |
| `sad_005` | उकोसेले मेरो भावन बुजी दिदैना सोपे जाना मलाई छोडेरा टाडा जानचन |
| `sad_006` | मावा देरे थागी सके, मावित्रावा अलिकतिपनी हिम्मत बागी छैनो |
| `sad_007` | किना मलाईने इस तो उन्छा? मैले तको साईकु नोराम्र चाहे कि थीना नी? |
Content notes:
- `neutral` reads as a coherent, deliberately chosen set of formal registers: a welcome
  address, a bus announcement, a fact about Nepal, an email acknowledgement, a weather report,
  a website pointer. Good reference material.
- `angry` reads as confrontation/betrayal ("get out of here right now", "you deceived me").
- `happy` reads as elation ("we won, our dream came true"), with English code-switching in
  `happy_007` and `happy_008`.
- `sad` reads as grief/loneliness ("I feel so alone today", "everything is over").
- No clip is off-topic for its label; the *text* prompts landed correctly. It is the *delivery*
  that went wrong.

---

## 7. Is 5–7 references per emotion enough prosodic variety?

**The question is aimed at the wrong lever.** §4 measured where output prosody comes from:

- F0-contour shape correlation, output vs **source**: r = 0.33–0.87, mean ≈ **0.70**
- F0-contour shape correlation, output vs **reference**: r = −0.37 to +0.62, mean ≈ **+0.15**
- Output pause fraction is set by the source and is **invariant to the reference** (0.164–0.208
  for one source, 0.268–0.293 for the other, across all 7 references)

**Seed-VC takes intonation contour and timing from the source clip, not the reference.** So
"will every output inherit the same intonation contour?" — yes, and adding references will not
change that. The contour monotony you are worried about will be inherited from the flat
Edge-TTS Nepali source clips, and the fix lives on the source side, not the reference side.

What the reference *does* control is register, timbre and level — and for those, the references
already have adequate spread. Within-emotion F0-contour-shape correlation among references:

| emotion | n | mean pairwise r | max pairwise r | contour sd (st), range |
|---|---|---|---|---|
| angry | 5 | +0.084 | +0.414 | 3.68 – 7.40 |
| happy | 7 | +0.172 | +0.567 | 3.46 – 7.78 |
| neutral | 6 | +0.299 | +0.552 | 4.41 – 7.98 |
| sad | 7 | −0.007 | +0.471 | 5.83 – 11.15 |

Mean pairwise r of 0.0–0.30 means the references are already near-decorrelated in contour
shape — they are not the bottleneck. And note the output's own contour sd (2.4–7.2 st) is
consistently *larger* than the source's (1.96–2.22 st): the reference does scale up the pitch
dynamic range roughly 2×, it just does not impose its own shape.

### How many more I would generate, and why

Not for variety — **for identity coverage and statistical power**. My concrete numbers:

| purpose | count | reasoning |
|---|---|---|
| Re-generate `angry` and `happy` at the anchor register | **15 per emotion** | You currently have 5 and 7, and after this audit 0 usable. Ask for at most +2 to +4 st over the neutral median (i.e. target 230–280 Hz, not 420 Hz) and explicitly forbid shouting/falsetto. Generate 15, then screen with the CAM++ ≥ 0.61 gate against the neutral centroid and expect to keep 8–12. |
| Re-generate `sad` | **12** | 4 of 7 are conditionally usable now. Ask for reduced volume and slower rate but a *normal* pitch register, and no sniffing (the sniffs cost you 35–56 % of each reference's usable duration). Keep 8+. |
| Extend `neutral` | **12** total (add ~6) | 6 is thin for the *anchor* identity, and the anchor is what every other emotion gets validated against — a noisy centroid propagates error into every gate decision. Also, at 6 clips, one bad clip is 17 % of your identity definition. |
| Duration target | **8–12 s each** | Your current spread is 4.36–11.76 s. `neutral_006` (4.36 s) and `happy_008` (5.16 s) are short; CAM++ embeddings get noticeably noisier below ~5 s of *speech*, and the sad clips only retain 44–66 % speech after silence. Ask for longer text. |

**Total: ~54 clips replacing the current 25**, of which I would expect ~35–40 to survive
screening. That is the number that matters, not 5–7.

### The screening gate I would put in front of the big job

Non-negotiable, because this failure was invisible to the checks that were in place:

1. Compute the CAM++ centroid of the accepted `neutral` set — that is the canonical Amrita.
2. Reject any candidate reference with cosine < **0.61** to that centroid (the p5 of
   same-speaker pairs in my calibration). Flag 0.54–0.61 for listening.
3. Reject any candidate whose median F0 is more than **±4 semitones** from the neutral median
   (204 Hz → accept 162–257 Hz). This one check alone would have caught all 12 angry/happy
   clips instantly.
4. Reject any candidate with < 60 % speech after VAD, or with a truncated tail (last 40 ms
   within 20 dB of the speech median).
5. After the batch runs, re-screen a random 200 *outputs* against the same centroid. §4 shows
   the reference's problem reaches the output at 0.79–0.91 fidelity, so output screening is a
   genuine independent check, not a formality.

### Note on the previous misleading result

Consistent with the earlier finding on this project that pitch *std* misleads: F0 **sd in
semitones** is 3.87 / 4.19 / 4.36 / 4.44 across angry / happy / neutral / sad — essentially
flat, and it would have told you these four folders were prosodically equivalent and
identity-consistent. The signal is entirely in F0 **median** (an octave apart) and in the
speaker embedding. Pitch std was uninformative here for the third time; median F0 plus a
speaker encoder are the two measures that worked.

---

## 8. Incidental findings outside the audit scope

Two things I ran into while building the calibration set. Neither is part of the 25-clip
question, but both affect the same downstream dataset.

**1. `manifests/reference_pool_v2/` is heavily duplicated.** I sampled 160 of the 600 clips and
embedded them with CAM++. **Only 22 survive deduplication at cosine > 0.995** — 138 of 160 are
near-exact duplicates of another clip in the sample, several at cosine = 1.000 (bit-identical
or near). Clustering the 22 unique embeddings gives 10 pseudo-speakers at a 0.45 distance cut,
6 at 0.55. This corroborates and sharpens the earlier finding that the v2 pool collapsed: the
effective size is closer to **~20–80 unique clips over ~6–10 voices**, not 600. Worth a
dedicated dedup pass.

**2. `vc_service/server.py` applies no VAD or trimming to the reference.** At line 238 the
CAM++ fbank is computed over the entire reference waveform:

```python
feat2 = torchaudio.compliance.kaldi.fbank(
    ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
)
```

The only preprocessing is a 25-second truncation (`ref_audio[: sr * 25]`, line 201). For the
`sad` references — 44–66 % speech — roughly half of the style vector is derived from silence
and breath noise. I tested whether fixing this would help the identity problem and it does not
(§1), so this is not urgent. But it is free accuracy: trimming the reference before the fbank
call would tighten every style vector, and it matters more as reference clips get longer or
noisier.

---

## Appendix — method and reproducibility

| what | how |
|---|---|
| Environment | `vc_service/.venv` (torch 2.4.0+cu121, torchaudio 2.4.0, librosa 0.10.2, numpy 1.26.4, scipy 1.13.1, Resemblyzer 0.1.4). No new venv was needed; nothing was installed into the pipeline tree. |
| CAM++ | `vc_service/seed-vc/campplus_cn_common.bin` via `modules.campplus.DTDNN.CAMPPlus(feat_dim=80, embedding_size=192)`, CPU, feature pipeline copied from `server.py:238-243`. |
| Resemblyzer | `VoiceEncoder("cpu")` + `preprocess_wav`, defaults. |
| F0 | `librosa.pyin(fmin=70, fmax=900, frame_length=2048, hop_length=256)`, restricted to frames that are both voiced and above the speech threshold. Cross-checked with `librosa.yin`, raw autocorrelation and real-cepstrum peak-picking on 40 ms frames. |
| Speaking rate | Peaks in a Savitzky-Golay-smoothed RMS envelope (min spacing 100 ms, height ≥ 10 % of max) — a syllable *estimate*, not a forced alignment. Reported both per second of file and per second of speech. |
| Formants | Resample to 11025 Hz, pre-emphasis, Hamming window, `librosa.lpc(order=12)`, roots with bandwidth < 500 Hz and 150–5200 Hz retained, median over voiced-and-loud frames. Unreliable above F0 ≈ 350 Hz — noted where it matters. |
| Pause fraction / VAD | Frames > 35 dB below the peak frame count as speech (`n_fft=1024`, `hop=256`); silence stripping used `librosa.effects.split(top_db=30)`. |
| ASR | `POST http://localhost:8013/transcribe`, multipart field `audio`. Read-only; the service was not restarted. |
| VC test | 14 `POST /convert` calls to the already-running replica on port 8032 while the GPU was idle (0 % util). Read-only; no service restarted or killed; outputs written to scratch only. |
| Not measured | Audible accent quality, naturalness, and whether a human hears one speaker — all require a native listener. Flagged as open in §3. |

### Confidence

| claim | confidence |
|---|---|
| angry/happy median F0 is ~+12 st above neutral | **high** — 4 independent estimators agree |
| angry and happy are not acoustically distinguishable from each other | **high** — agreement across F0, energy, rate, spectral tilt, F4, MFCC, LTAS and CAM++ |
| the reference set's inconsistency reaches converted output | **high** — measured directly on 14 conversions |
| intonation contour comes from the source, not the reference | **high** — r ≈ 0.70 vs r ≈ 0.15, consistent across both sources and all 7 references |
| the exact same-speaker threshold is 0.562 | **medium** — 19 same-speaker pairs, cluster-derived pseudo-labels; treat as ±0.05. Conclusions are robust to this because the failing pairs sit at 0.44–0.50. |
| angry/happy are the *same vocal tract* in a different register rather than a different voice | **medium** — F4 is stable within 4 % and Resemblyzer is much less alarmed than CAM++, but LPC formants are unreliable at 400–500 Hz F0. Either way the references are unusable, so this ambiguity does not change the recommendation. |
| whether a native listener hears one speaker | **not assessed** |
