---
license: cc-by-4.0
language:
  - ne
  - en
tags:
  - automatic-speech-recognition
  - speech
  - nepali
  - ctc
  - parakeet
  - nemo
  - fastconformer
library_name: nemo
datasets:
  - ai4bharat/indicvoices_r
  - ai4bharat/Rasa
metrics:
  - wer
base_model: nvidia/parakeet-tdt_ctc-110m
---

# Parakeet-CTC Nepali 110M

A 110.8M-parameter CTC speech recogniser for **Nepali**, with English carried
along, intended for on-device inference. It is a FastConformer encoder
initialised from [`nvidia/parakeet-tdt_ctc-110m`](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m)
with a **randomly initialised 4,000-class CTC head** over a custom Nepali+English
BPE, then trained on 4,201.6 h of mixed Nepali and Indian-accented English.

> **Correction (2026-09-18).** An earlier version of this card led with
> "Held-out human Nepali WER: 0.176 (200 clips … disjoint from training)".
> **That set is not disjoint from training** — all 200 rows appear verbatim in
> `train_ne_human.jsonl`. The number has been relabelled as a seen-data score
> throughout. See [Results](#results).

**Gold-standard Nepali WER: 0.302 / CER 0.103** — FLEURS `ne_np` test, 726 clips
/ 2.28 h, 100% expert human transcripts, **verified disjoint from training**
(`gold_eval/`). This is the number to quote.

For context, and to show why the older figures were misleading:

| set | n | WER | what it actually measures |
|---|---|---|---|
| **FLEURS `ne_np` test (gold)** | 726 | **0.302** | **accuracy — unseen, human-labelled** |
| `val_ne` (unseen, ~82% machine-labelled) | 600 | 0.152 | mostly agreement with the labelling system |
| `hum_heldout` (**in training**, ~61 passes) | 200 | 0.176 | memorisation; not a generalisation estimate |

The model generalises **1.7x worse** than its own "held-out" set suggested. Read
[Results](#results) and [Training data](#training-data--label-provenance) before
quoting anything here.

## What this model is, and what it is not

This is a **research pretraining artifact**, published for reproducibility. It is
not a product, and it is not a from-scratch model.

- **It is not trained from scratch.** The encoder is NVIDIA's parakeet weights.
  Whatever this model does well on English, it inherited. Five genuine
  from-scratch attempts on this corpus all collapsed to the all-blank CTC
  attractor (`docs/DIAGNOSIS_blank_collapse.md` is the full postmortem).
- **It is not license-independent.** The CC-BY-4.0 tag below covers *these*
  weights, but the encoder lineage means the model is a derivative of NVIDIA's
  release, not a clean-room one. If you need independence from NVIDIA, this is
  not that artifact.
- **Its accuracy is capped by its labels, not its size.** 89.9% of the training
  hours carry machine-generated transcripts. See
  [Training data](#training-data--label-provenance).
- **Its original headline eval was contaminated.** `manifests/hum_heldout.jsonl`
  was described and used as a held-out set, but every one of its 200 rows is
  present verbatim — same path, text and duration — in `train_ne_human.jsonl`,
  which trained at weight 0.2115 (~61 passes over each clip). The eval set was
  never excluded from the training manifest; `code/pilot_train.py` contains no
  exclusion logic. Found post-publication and corrected here rather than quietly
  re-scored.
- **It now has a gold number, added after the fact.** The original release shipped
  with no training-disjoint human-labelled evaluation at all. `gold_eval/` adds
  one (FLEURS `ne_np` test) along with the script that proves the set is absent
  from the training corpus. Human spot-checks have still not been done.
- **It stopped improving well before it stopped training.** Nepali WER was flat
  from step ~35k to 50k — six consecutive evals at `val-every 2500`, which met
  the run's own pre-registered stop condition. Training continued to 50k anyway.
  Documented rather than hidden because it is the most useful thing in the repo.

## Model details

| | |
|---|---|
| Total parameters | **110,814,625** (110.8M) |
| Architecture | FastConformer encoder + CTC head (`EncDecCTCModelBPE`) |
| Encoder | parakeet-tdt_ctc-110m, all 692 tensors loaded, 0 missing |
| `d_model` / layers / heads | 512 / 17 / 8 |
| Subsampling / conv kernel | 8x depthwise-separable / 9 |
| Decoder | CTC, **4,001 output classes** (4,000 BPE pieces + blank at index 4000) |
| Tokenizer | SentencePiece BPE, 4,000 pieces, **~1.70 tok/word** on Nepali, 0% UNK |
| Input | 16 kHz mono, 80-bin log-mel, n_fft 512, 25 ms window / 10 ms stride, per-feature normalisation |
| Output | Lowercased Devanagari / Latin text, punctuation stripped, numbers verbalised |
| Framework | NVIDIA NeMo |
| Checkpoint size | 444 MB (`.nemo`) |

The tokenizer is a deliberate departure from the shipped parakeet BPE, and the
reason is stronger than the earlier version of this card claimed.

**Corrected 2026-09-18.** This card previously said the shipped BPE "costs 5.12
tok/word on Nepali" against "2.39" for ours. Both numbers were wrong, measured
on the published artifacts:

| tokenizer | vocab | tok/word on Nepali | UNK rate |
|---|---|---|---|
| shipped `parakeet-tdt_ctc-110m` BPE | 1,024 | 2.06 (meaningless — see below) | **48.9%** |
| this model's BPE | 4,000 | **1.69–1.74** | **0.0%** |

The shipped BPE does not encode Nepali *at all*: it is a 1,024-piece English
vocabulary with no byte fallback, so roughly every Devanagari word becomes a
single `<unk>`. Its apparent "2.06 tok/word" is the cost of discarding the text,
not representing it. The old "5.12" figure is not that tokenizer's output — it is
simply the **character count per Nepali word** (measured: 5.09), i.e. what a
character-level fallback would cost.

So the real argument for a custom tokenizer is not "2.1x fewer frames", it is
that the donor vocabulary is **unusable for Nepali** and a from-scratch BPE gets
1.70 tok/word at zero UNK — about 3x better than character fallback. Measured on
20k rows each of `train_ne_pseudo` and `train_ne_human`; on out-of-domain FLEURS
text ours rises to 2.01 tok/word.

Text normalisation (number verbalisation in particular) is baked into the
tokenizer's training data, so the model expects already-verbalised text.

## Training data & label provenance

4,201.6 h across 12 sources, mixed by weighted lhotse `input_cfg`. Weights are
sampling *weights*, not hours; `ne_human` in particular is upsampled 3x.

| bucket | weight | hours | rows | what it is |
|---|---|---|---|---|
| `ne_pseudo` | 0.3924 | 1,919.0 | 626,267 | Nepali, machine-labelled |
| `en_pseudo` | 0.3799 | 1,858.1 | 498,254 | Indian-accented English, machine-labelled |
| `ne_human` | 0.2115 | 344.8 | 125,076 | Nepali, human transcripts |
| `ne_script` | 0.0163 | 79.7 | 7,160 | Nepali, read from script |
| **total** | | **4,201.6** | **1,256,757** | |

**Label provenance — read this before quoting any number.**

| label source | hours | share |
|---|---|---|
| machine-generated (pseudo) | 3,777.2 | **89.9%** |
| human-transcribed | 344.7 | 8.2% |
| read-from-script | 79.7 | 1.9% |

The human hours are the AI4Bharat three — `indicvoices-r` (162.3 h),
`indicvoices-r-long` (127.9 h) and `rasa` (54.6 h) — and nothing else. Every
other Nepali hour, and all English hours, carry transcripts produced by an
automated system. **A model trained on those labels is, in the large, distilling
that system, and its held-out scores partly measure agreement with it rather
than accuracy against ground truth.**

**Licensing of the training audio.** The corpus is **not redistributable**:
1,911.6 h — 81.6% of the 2,343.5 h Nepali portion — is YouTube-derived
(`ans_snr40-50`, `ans_snr50`, `mahadhwani`; the last, despite the name, resolves
to YouTube video IDs in its manifest). Only the AI4Bharat portion (344.7 h,
CC-BY-4.0) is clean, and it is already public at source. English audio is likewise YouTube-sourced
(`en-in_snr40-50`, `en-in_snr50`). **This repo therefore ships weights only, not
data, and no audio from training is included or downloadable here.**

**English ended up at 44.2% of hours**, above the 40% target. English was sized
against the pre-gate Nepali figure; a later alignability + Devanagari-lexicon
gate then cut ~452 h (16%) of Nepali, after the target was fixed. This is a
known mis-sizing, documented in `manifests/mix_report.json` and not corrected.

Preprocessing gates applied before a row was admitted: duration 1–60 s, ASR
CER ≤ 0.10, DNSMOS ≥ 2.7, speaker overlap ≤ 5%, 3-ASR consensus ≥ 2 of 3,
frames-per-token ≥ 1.3, Devanagari-lexicon fraction ≥ 0.55, path dedupe.

## Training procedure

| | |
|---|---|
| Steps | 50,000 (completed; `rc=0`) |
| Audio seen | ≤ 50,000 x 7,200 s = **100,000 h** (~23.8x the corpus). 1,200 s is lhotse's `max_batch_duration` *cap*, not realised duration, so this is an upper bound |
| Update size | `batch_duration` 1,200 s x `accum` 6 = **7,200 s/update** |
| Optimiser / LR | AdamW-style NeMo default, peak **5e-4**, warmup 2,000, cosine to 1e-6 |
| Precision | bf16 |
| Max utterance | 20 s, bucketed (30 buckets, shuffle_n 2048) |
| Regularisation | grad-clip 1.0, 2 freq masks, 10 time masks |
| Hardware | **1x H100 80GB** |
| Wallclock | 28.5 h (2026-09-16 08:41 to 2026-09-17 13:15, host-local UTC+05:30 — not UTC), 2.93 it/s |
| Validation | every 2,500 steps |

The initialisation is the whole story of this run: `run_v7_encinit` differs from
the failed from-scratch arms only in loading parakeet's encoder and lowering LR
to 5e-4. At step 2,500 it was already at loss/token 1.398 / WER 0.382, where the
best from-scratch run sat at 8.291 / WER 1.000 / 98.7% blank after 97,500 h seen.

## Results

**Instrument.** `eval_probe.py` in `eval/`, greedy CTC decode, fixed utterance
set, loss reported per *target token*. The logged training loss is deliberately
not used: lhotse bucketing swings it 40–1800 on a model that has learned nothing,
so it cannot be compared across steps.

### Seen-data human Nepali (`manifests/hum_heldout.jsonl`) — **not held out**

200 clips / 0.3644 h, **100% human transcripts**, all `ne-NP`, drawn from
`indicvoices-r` (119), `rasa` (48), `indicvoices-r-long` (33).

**All 200 rows are in the training set.** Each matches a row in
`train_ne_human.jsonl` on path, text and duration. That bucket trained at weight
0.2115, so over ~100,000 h seen it supplied ~21,150 h drawn from a 344.7 h pool —
roughly **61 passes over every clip below**. The filename is a misnomer kept only
so the manifest still matches the numbers it produced.

| step | loss/token | WER | blank% | empty hyp |
|---|---|---|---|---|
| 2,500 | 1.398 | 0.382 | 75.9 | 0 / 200 |
| ~35,000 | 0.534 | 0.177 | 71.1 | 0 / 200 |
| 40,799 | 0.522 | 0.178 | 70.9 | 0 / 200 |
| **50,000 (final)** | **0.515** | **0.176** | **70.9** | **0 / 200** |

Treat 0.176 as an **upper bound measured on memorised data**, not a
generalisation estimate. It is the exact artifact this project's own protocol
warns about: scoring a model on a split of its own training corpus.

**The plateau still stands as a finding.** 15,000 further steps, with the
learning rate annealed to its 1e-6 floor, moved WER by 0.002 — inside noise at
n=200, and on data the model had already seen ~61 times. Steps are not the
binding constraint; labels are. The final checkpoint is the one shipped, so there
is no better intermediate to prefer.

### Genuinely unseen — validation sets

These three **are** disjoint from training (verified: 0 overlapping paths against
all four training manifests). They are the only generalisation evidence here.

| set | n | composition | loss/token | WER | blank% |
|---|---|---|---|---|---|
| **`val_ne.jsonl` (600-clip prefix)** | 600 | Nepali; prefix is 491 pseudo / 102 human / 7 script (full set 973 / 211 / 16) | 1.558 | **0.152** | 67.8 |
| `val_en.jsonl` (600-clip prefix) | 600 | Indian-accented English, **100% machine-labelled** | 0.084 | 0.049 | 48.5 |
| `val_mix.jsonl` (logged, best ckpt) | 2,400 | 1,200 en-US + 1,200 ne-NP | — | 0.0742 | — |

**0.152 is the closest thing to a real Nepali number here, and it is still not a
clean one.** `val_ne` is ~82% machine-labelled, so a model trained on the same
machine labels is partly being scored on its own teacher's output — much of that
WER is agreement, not accuracy. The higher loss/token (1.558 vs the seen set's
0.515) is the tell: these rows are genuinely harder. `val_mix`'s 0.0742 is
roughly half English, which is native to the inherited encoder, so it is not a
Nepali number at all.

Only **102 of the 600** `val_ne` clips carry human transcripts. For a clean
number, see the gold set below.

### Primary — FLEURS `ne_np` test (gold, training-disjoint)

726 clips / 2.284 h, **100% expert human transcripts**, read speech, from a
source that is not in this corpus at all. Artifacts in `gold_eval/`.

| model | params | WER | CER | WER (no digits in ref) |
|---|---|---|---|---|
| **`parakeet-ctc-nepali-110m`** (this model) | 110.8M | **0.3021** | 0.1025 | 0.2804 |
| `nemotron-asr-nepali-0.6b` (sibling finetune) | 600M | 0.3117 | 0.0988 | 0.2914 |

Identical references, identical normalizer, identical metric, 0 empty hypotheses
from either model — this **is** a head-to-head, unlike the comparison the earlier
card attempted. At 5.4x fewer parameters this model is marginally ahead on WER
and marginally behind on CER; the honest reading is that the two are
**indistinguishable in quality**, which is the interesting result given the size
difference.

**Disjointness was verified, not assumed** (`gold_eval/verify_disjoint2.py`,
output in `gold_eval/verify2.log`). Across **1,256,757** training rows for this
model and **1,521,426** for the sibling: 0 exact canonical-text matches, 0
basename collisions, 0 near-duplicates at Jaccard ≥ 0.6 over 5-gram sets, max
Jaccard 0.148 (ordinary phrase reuse). The canonical form used for matching
strips Devanagari vowel signs, so it is *lossier* than the text itself — it
collides more readily than the real strings would, which makes a zero-overlap
result conservative in the right direction.

**On the digit split.** FLEURS references contain ASCII digits ("सन् 1800 को");
this model emits verbalised numbers. References are therefore verbalised with the
training normalizer before scoring. That is still imperfect — the verbalizer
renders 1800 as *एक हजार आठ सय* while the speaker reads *अठार सय*, both correct
Nepali — which is why the 138 digit-bearing clips score ~0.10 WER worse than the
588 without. The no-digit column is the fairer read of acoustic accuracy.

### Reproducibility

The **gold** numbers are fully reproducible: `gold_eval/` ships the manifest, the
fetch script, the scorer, the comparison, and the captured stdout of every run.

The **seen-data and validation** tables above are not. They came from interactive
`eval_probe.py` runs whose output was never written to disk, and no `eval_probe`
log for `run_v7_encinit` survives — those rows can only be re-derived by
re-running the instrument. The single externally corroborated figure among them
is `val_mix` **0.0742**, which appears in the checkpoint filename written by
`exp_manager` (`run_v7_encinit--val_wer=0.0742-epoch=0.ckpt`).

**English at 0.049 is not a claim.** All 600 clips are machine-labelled, and
English is native to the inherited encoder, so this reflects both the shared
label source and the donor weights rather than anything this run contributed.
The blank rate is the honest tell: 48.5% on English versus 67.8% on Nepali —
the model is far more confident in the language it did not have to learn.

### Reference point

The sibling 0.6B finetune [`milanakdj/nemotron-asr-nepali-0.6b`](https://huggingface.co/milanakdj/nemotron-asr-nepali-0.6b)
reports **0.204 WER / 0.076 CER** (raw; 0.186 / 0.068 normalised) on its own
600-clip internal Nepali set. An earlier version of this card cited 0.195 for it,
which does not match that model's own card. **Do not compare either figure to
anything here** — different set, different instrument. The
meaningful comparison is the head-to-head in
[Primary](#primary--fleurs-ne_np-test-gold-training-disjoint), where both models
were run on the same gold set under the same normalizer and land at 0.302 vs
0.312.

The gap between the sibling's self-reported 0.195 and the 0.312 it scores here is
itself worth noting: it is a set-difficulty and normalisation difference, not a
regression. FLEURS is read Wikipedia-style prose from a different domain than
either model's training data.

## Usage

Requires NVIDIA NeMo. The model is a plain CTC model: pass paths, get text. No
prompt or language tag is needed, unlike the sibling nemotron checkpoint.

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecCTCModelBPE.restore_from("parakeet-ctc-nepali-110m.nemo")
model.eval()

transcripts = model.transcribe(["clip.wav"], batch_size=8)
print(transcripts[0].text)
```

Or from the CLI. **Corrected 2026-09-18** — this card previously gave
`python -m nemo.collections.asr.transcribe`, which does not exist in any NeMo
release and fails with `ModuleNotFoundError`. NeMo ships transcription as a
script in its source tree, not an installed module, so you need a checkout:

```bash
# from a NeMo source checkout (the script is not part of the installed package)
python examples/asr/transcribe_speech.py \
  model_path=parakeet-ctc-nepali-110m.nemo \
  audio_dir=/path/to/wavs \
  output_filename=hyps.json
```

If you do not have a NeMo checkout, use the three-line Python API above — it is
the supported path and needs nothing extra.

Input must be 16 kHz mono. Output is normalised text (lowercase, no punctuation,
numbers already verbalised) — if you need punctuation or casing, restore it
downstream.

## Colab Usage

This model is a NeMo `.nemo` checkpoint and requires **NVIDIA NeMo Toolkit** for inference.

### Google Colab

A GPU runtime is recommended. The following example was tested on Google Colab with a T4 GPU.

#### 1. Install NeMo

```python
!pip install -q "nemo_toolkit[asr]==3.0.0"
```

After installation, **restart the Colab runtime** before importing NeMo.

#### 2. Authenticate with Hugging Face

If the model repository is private, create a Hugging Face token with read access and add it to Colab Secrets as `HF_TOKEN`.

```python
from google.colab import userdata
from huggingface_hub import login

hf_token = userdata.get("HF_TOKEN")
login(token=hf_token)
```

#### 3. Download the model

```python
!hf download milanakdj/parakeet-ctc-nepali-110m \
    parakeet-ctc-nepali-110m.nemo \
    --local-dir /content
```

#### 4. Load the model and transcribe audio

```python
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
    "/content/parakeet-ctc-nepali-110m.nemo"
)

model.eval()

transcripts = model.transcribe(
    ["/content/my_clip.flac"],
    batch_size=8
)

print(transcripts[0].text)
```

Replace `/content/my_clip.flac` with the path to your own audio file.

### Supported audio

The model is configured for **16 kHz** audio. NeMo's audio processing can handle common input formats such as `.flac` and perform resampling when necessary.

### Example output

```text
कोरोनाको गीतफ्ट बनाउँदा यी गायक गायिकालाई गालीको वर्षा
```

### Notes

* This is a **CTC-based Nepali ASR model**.
* The model uses a SentencePiece BPE tokenizer.
* Transcription returns NeMo `Hypothesis` objects; use `.text` to obtain the transcription string.
* The model output is generally normalized text and may not contain punctuation.
* `batch_size` can be reduced if GPU memory is limited.



## Limitations

- **Encoder licence lineage.** The CC-BY-4.0 tag on this repo covers these
  weights, but the encoder derives from NVIDIA's parakeet release. Redistribution
  and commercial use should be reviewed against that lineage, not just this tag.
- **89.9% machine-labelled training data.** Errors in the labelling system are
  inherited and, on the pseudo-labelled share, partly *rewarded*. Expect the
  model to reproduce that system's biases and failure modes.
- **Expect ~0.30 WER on real unseen Nepali, not ~0.18.** The gold figure is
  0.3021 on FLEURS; the 0.176 was seen data and the 0.152 is ~82%
  machine-labelled. Quote 0.302.
- **The gold set is read speech from one domain.** FLEURS is read
  Wikipedia-style prose. It is training-disjoint and human-labelled, which is
  what was missing, but it is not conversational, spontaneous, noisy or
  dialectal Nepali. One clean number is not broad coverage.
- **No human spot-checks.** All figures remain automatic-scorer output.
- **Eval results are not reproducible from this repo.** The `eval_probe` runs
  behind the tables were not logged to disk; only the `val_mix` 0.0742 is
  externally corroborated.
- **Training audio is not redistributable** (81.6% of the Nepali portion is
  YouTube-derived) and is not included here in any form.
- **Plateau.** Nepali WER has not improved since ~step 35k. Scaling this recipe
  further on this corpus is not expected to help.
- **Blank-heavy output.** 70.9% blank frames on the seen-data set and **71.16%**
  on the unseen gold set — high, but stable across the two, and 0 of 726 gold
  clips produced an empty hypothesis. (An earlier version of this card predicted
  the blank rate would be worse out of domain; measured, it is not.) Still worth
  watching for truncated output on audio unlike either set.
- **English is incidental.** It is 44.2% of training hours because of a sizing
  error and is native to the encoder; it is not a supported target language and
  has not been evaluated on a human-labelled English set.
- **Nepali only, one variety.** Trained on `ne-NP` sources; no dialect or
  code-switching evaluation has been done.
- **No punctuation, casing, or timestamps.** CTC greedy decode over a
  normalised BPE.
- **Small eval sets.** 200 clips / 0.36 h (seen) and 600 clips (unseen).
  Differences below ~0.01 WER on this instrument are not resolvable.
- **Checkpoint averaging does not help here — tested, not assumed.** The protocol
  expects ~3–5% relative from averaging the top checkpoints. Averaging all four
  saved checkpoints (3 top-k + train-end) and scoring on the gold set gives
  **WER 0.3039 / CER 0.1083 vs the shipped 0.3021 / 0.1075** — very slightly
  *worse*. All four come from the last ~5k steps of an already-annealed,
  plateaued run, so they sit at effectively the same point in weight space and
  there is nothing to average away. The shipped weights are the final step, and
  that is the right choice. Script: `gold_eval/avg_ckpt.py`.

## Files in this repo

| path | what |
|---|---|
| `parakeet-ctc-nepali-110m.nemo` | the model (444 MB) |
| `code/run_v7_encinit.sh` | exact training invocation, incl. the pre-registered kill criterion |
| `code/pilot_train.py` | training entrypoint |
| `eval/eval_probe.py` | the scoring instrument used for every number above |
| `manifests/hum_heldout.jsonl` | the 200-clip human Nepali scoring set (paths only, no audio). **Misnomer — it is not held out; all 200 rows are in `train_ne_human.jsonl`** |
| `manifests/train_input_cfg.yaml` | bucket weights as trained |
| `manifests/mix_report.json` | hours by source and by label provenance |
| `gold_eval/` | the training-disjoint gold evaluation: FLEURS manifest, fetch/score/compare scripts, the disjointness proof, and **captured logs** for every number |
| `docs/DIAGNOSIS_blank_collapse.md` | why from-scratch failed and encoder-init did not |
| `docs/WHY_THIS_REPO.md` | the goal this run was and was not pursuing |

## Provenance

Trained 2026-09-16/17 on 1x H100 as `run_v7_encinit`. Donor weights:
`nvidia/parakeet-tdt_ctc-110m` (CC-BY-4.0), revision
`431a349f3051ab85c22b9b7a2741b5fe77065665`.
