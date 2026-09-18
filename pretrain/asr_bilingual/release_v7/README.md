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

**Held-out human Nepali WER: 0.176** (200 clips, all human-labelled, disjoint
from training). See [Results](#results) — and read the caveats, because this
number is only meaningful next to the data-provenance section.

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
- **It stopped improving well before it stopped training.** Nepali held-out WER
  was flat from step ~35k to 50k. More steps would not have helped. This is
  documented rather than hidden because it is the most useful thing in the repo.

## Model details

| | |
|---|---|
| Total parameters | **110,814,625** (110.8M) |
| Architecture | FastConformer encoder + CTC head (`EncDecCTCModelBPE`) |
| Encoder | parakeet-tdt_ctc-110m, all 692 tensors loaded, 0 missing |
| `d_model` / layers / heads | 512 / 17 / 8 |
| Subsampling / conv kernel | 8x depthwise-separable / 9 |
| Decoder | CTC, **4,001 output classes** (4,000 BPE pieces + blank at index 4000) |
| Tokenizer | SentencePiece BPE, 4,000 pieces, **2.39 tok/word** on Nepali |
| Input | 16 kHz mono, 80-bin log-mel, n_fft 512, 25 ms window / 10 ms stride, per-feature normalisation |
| Output | Lowercased Devanagari / Latin text, punctuation stripped, numbers verbalised |
| Framework | NVIDIA NeMo |
| Checkpoint size | 444 MB (`.nemo`) |

The tokenizer is a deliberate departure from the shipped parakeet BPE, which
costs **5.12 tok/word** on Nepali. At 2.39 the same audio produces fewer than
half the output frames, which is what makes a 110M CTC model viable at this
latency. Text normalisation (number verbalisation in particular) is baked into
the tokenizer's training data, so the model expects already-verbalised text.

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
1,919.1 h (86.6% of the Nepali portion) is YouTube-derived, including
`mahadhwani`, which despite the name resolves to YouTube video IDs in its
manifest. Only the AI4Bharat portion (296.6 h, CC-BY-4.0) is clean, and it is
already public at source. English audio is likewise YouTube-sourced
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
| Audio seen | 50,000 x 7,200 s = **100,000 h** (~23.8x the corpus) |
| Update size | `batch_duration` 1,200 s x `accum` 6 = **7,200 s/update** |
| Optimiser / LR | AdamW-style NeMo default, peak **5e-4**, warmup 2,000, cosine to 1e-6 |
| Precision | bf16 |
| Max utterance | 20 s, bucketed (30 buckets, shuffle_n 2048) |
| Regularisation | grad-clip 1.0, 2 freq masks, 10 time masks |
| Hardware | **1x H100 80GB** |
| Wallclock | 28.5 h (2026-09-16 08:41 to 2026-09-17 13:15 UTC), 2.92 it/s |
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

### Primary — held-out human Nepali (`manifests/hum_heldout.jsonl`)

200 clips / 0.36 h, **100% human transcripts**, all `ne-NP`, drawn from
`indicvoices-r` (119), `rasa` (48), `indicvoices-r-long` (33), disjoint from
training.

| step | loss/token | WER | blank% | empty hyp |
|---|---|---|---|---|
| 2,500 | 1.398 | 0.382 | 75.9 | 0 / 200 |
| ~35,000 | 0.534 | 0.177 | 71.1 | 0 / 200 |
| 40,799 | 0.522 | 0.178 | 70.9 | 0 / 200 |
| **50,000 (final)** | **0.515** | **0.176** | **70.9** | **0 / 200** |

**The plateau is the finding.** 15,000 further steps, with the learning rate
annealed to its 1e-6 floor, moved WER by 0.002 — inside noise at n=200. Steps are
not the binding constraint; labels are. The best number came from the run's last
checkpoint, so there is no better intermediate to prefer.

### Secondary — mixed and per-language validation

| set | n | composition | loss/token | WER | blank% |
|---|---|---|---|---|---|
| `val_ne.jsonl` (600-clip prefix) | 600 | Nepali; full set is 973 pseudo / 211 human / 16 script | 1.558 | 0.152 | 67.8 |
| `val_en.jsonl` (600-clip prefix) | 600 | Indian-accented English, **100% machine-labelled** | 0.084 | 0.049 | 48.5 |
| `val_mix.jsonl` (logged, best) | 2,400 | 1,200 en-US + 1,200 ne-NP | — | 0.0742 | — |

**Do not read the 0.152 as better than the 0.176.** `val_ne` is ~81%
machine-labelled, so a model trained on the same machine labels is partly being
scored on its own teacher's output — the lower WER is agreement, not accuracy.
The higher loss/token (1.558 vs 0.515) at the same time is the tell: those rows
are genuinely harder, but their references bend toward whatever the model
already says. `val_mix`'s 0.0742 is roughly half English, which is native to the
inherited encoder, so it is not a Nepali number at all.

**English at 0.049 is not a claim.** All 600 clips are machine-labelled, and
English is native to the inherited encoder, so this reflects both the shared
label source and the donor weights rather than anything this run contributed.
The blank rate is the honest tell: 48.5% on English versus 67.8% on Nepali —
the model is far more confident in the language it did not have to learn.

### Reference point

The sibling 0.6B finetune [`milanakdj/nemotron-asr-nepali-0.6b`](https://huggingface.co/milanakdj/nemotron-asr-nepali-0.6b)
scores **0.195 WER / 0.075 CER on a 600-clip Nepali gold set**. That is a
*different set and a different instrument* — **not a head-to-head** — but it puts
this 110M model in the same band at roughly one-fifth the parameters.

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

Or from the CLI:

```bash
python -m nemo.collections.asr.transcribe \
  model=parakeet-ctc-nepali-110m.nemo \
  audio=clip.wav
```

Input must be 16 kHz mono. Output is normalised text (lowercase, no punctuation,
numbers already verbalised) — if you need punctuation or casing, restore it
downstream.

## Limitations

- **Encoder licence lineage.** The CC-BY-4.0 tag on this repo covers these
  weights, but the encoder derives from NVIDIA's parakeet release. Redistribution
  and commercial use should be reviewed against that lineage, not just this tag.
- **89.9% machine-labelled training data.** Errors in the labelling system are
  inherited and, on the pseudo-labelled share, partly *rewarded*. Expect the
  model to reproduce that system's biases and failure modes.
- **Training audio is not redistributable** (86.6% YouTube-derived) and is not
  included here in any form.
- **Plateau.** Nepali WER has not improved since ~step 35k. Scaling this recipe
  further on this corpus is not expected to help.
- **Blank-heavy output.** 70.9% blank frames on human Nepali — high, though all
  200 held-out clips produced non-empty hypotheses. Watch for empty or truncated
  output on out-of-domain audio.
- **English is incidental.** It is 44.2% of training hours because of a sizing
  error and is native to the encoder; it is not a supported target language and
  has not been evaluated on a human-labelled English set.
- **Nepali only, one variety.** Trained on `ne-NP` sources; no dialect or
  code-switching evaluation has been done.
- **No punctuation, casing, or timestamps.** CTC greedy decode over a
  normalised BPE.
- **Small held-out set.** 200 clips / 0.36 h. Differences below ~0.01 WER on this
  instrument are not resolvable.

## Files in this repo

| path | what |
|---|---|
| `parakeet-ctc-nepali-110m.nemo` | the model (444 MB) |
| `code/run_v7_encinit.sh` | exact training invocation, incl. the pre-registered kill criterion |
| `code/pilot_train.py` | training entrypoint |
| `eval/eval_probe.py` | the scoring instrument used for every number above |
| `manifests/hum_heldout.jsonl` | the 200-clip human held-out set (paths only, no audio) |
| `manifests/train_input_cfg.yaml` | bucket weights as trained |
| `manifests/mix_report.json` | hours by source and by label provenance |
| `docs/DIAGNOSIS_blank_collapse.md` | why from-scratch failed and encoder-init did not |
| `docs/WHY_THIS_REPO.md` | the goal this run was and was not pursuing |

## Provenance

Trained 2026-09-16/17 on 1x H100 as `run_v7_encinit`. Donor weights:
`nvidia/parakeet-tdt_ctc-110m` (CC-BY-4.0), revision
`431a349f3051ab85c22b9b7a2741b5fe77065665`.
