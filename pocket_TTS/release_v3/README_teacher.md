---
license: other
language:
- ne
- en
pipeline_tag: text-to-speech
tags:
- pocket-tts
- nepali
- failed-run
- archive
---

# pocket-tts-nepali-24l-teacher-v3 — FAILED RUN, ARCHIVED FOR THE RECORD

> **Do not use this model.** It does not produce intelligible Nepali. It is
> published private, as a record of a training run that failed, so the failure is
> reproducible and a future attempt has a baseline to beat.
>
> **The working Nepali model is [`milanakdj/pocket-tts-nepali-6l`](https://huggingface.co/milanakdj/pocket-tts-nepali-6l)** (v2, the 6-layer student).

## What this was meant to be

Stage 1 of v3: a 24-layer teacher finetuned from Kyutai's English release, with a
**grafted tokenizer** — Kyutai's original 4,000 pieces kept byte-identical, plus
5,682 appended Nepali pieces — and the pretrained English embedding rows
inherited rather than reset (`graft_text_embedding: 4000`). v2's Nepali-only
tokenizer had no byte fallback and *deleted* Latin words and ASCII digits instead
of speaking them. The graft was meant to fix that and add usable English.

A 6-layer student was then depth-distilled from this checkpoint. Both stages ran
to completion, 200,000 steps each, no crash.

## What actually happened

The canonical eval: **100 held-out utterances, 77 speaker identities, 6 sources**
(all verified still held out under v3), scored with
`himalaya-ai/whisper-large-v3-nepali-final`. `real human` is the genuine
recording of the same utterance through the identical pipeline — the ceiling,
and the check that the metric can see what it is measuring. The v2 rows
reproduce the published v2 table exactly, which is how we know the harness is
unchanged and the v3 rows can be trusted.

| system | WER | CER | speaker sim | CPU speed |
|---|---|---|---|---|
| real human (ceiling) | 0.343 | 0.122 | 0.807 | — |
| **v2 student 6L (shipped)** | **0.322** | **0.139** | **0.841** | **5.88x RT** |
| v2 teacher 24L | 0.402 | 0.177 | 0.826 | 2.10x RT |
| **v3 teacher 24L** | **0.812** | **0.632** | **0.701** | **2.10x RT** |
| **v3 student 6L** | **0.724** | **0.578** | **0.635** | **5.93x RT** |

v3 is roughly four times worse on CER than the model it was meant to replace,
and it is worse on every source individually. Speaker similarity collapsed too
(0.635 against v2's 0.841, below even the real-human cross-utterance ceiling of
0.807) — it does not clone the voice properly either. The one thing unchanged is
speed: 5.93x against 5.88x real-time, measured in the same run on the same box.
v3 costs everything and buys nothing.

A second set of **40 utterances, disjoint from the eval set**, was used to pick
`eos_threshold` and temperature so that nothing was tuned on the test data. It
carries the same verdict: real human CER 0.149, v2 student **0.107**, v3 student
0.506 at its best settings.

The training objective said so from the start, which is the part worth
remembering:

| teacher, valid @ 200k | flow_loss | eos_loss | flow_diag |
|---|---|---|---|
| v2 | **-0.0746** | 0.0078 | 14.43 |
| v3 (this model) | **+0.0831** | 0.0276 | 16.93 |

This run at step 200,000 is worse than v2 was at step **5,000** (0.0453). The
curve is flat from ~100k, which reads as convergence only if you do not put the
previous generation's curve beside it. It converged to a much worse point.

## The trap, if you load it anyway

At pocket-tts's default `eos_threshold=-4.0` this checkpoint emits about **0.49x**
the duration it should, silently. The threshold is not expressible in
`config.yaml` (pocket-tts's `Config` schema is strict and has no such field), so
it must be passed to `load_model`:

```python
from pocket_tts.models.tts_model import TTSModel
model = TTSModel.load_model(
    config="hf://milanakdj/pocket-tts-nepali-24l-teacher-v3-failed/config.yaml",
    eos_threshold=1.0,   # and it is still unguided, so still bad
)
```

Fixing the duration does not fix the content. The elevated `eos_loss` above is
the honest reading: the EOS head is badly fit, and so is the rest of the model.

## Probable cause

The v3 models speak **English better than Nepali** — the distilled student
renders "The quick brown fox jumps over the lazy dog." verbatim while its Nepali
is unintelligible.

v2 reset the entire text embedding and learned 4,000 Nepali pieces from scratch
with 100% of the signal on Nepali. v3 inherited the pretrained English rows and
had to learn **5,682 fresh Nepali rows** — 1.4x more pieces, each seen
proportionally less often — while roughly 27% of the corpus (800 h of ~3,010 h)
was English replay. The graft delivered what it promised, and it was paid for out
of the Nepali budget.

Hypothesis, not proven cause. The cheapest next test is to re-run **stage 1 only**
with the English replay cut to ~5%, and compare the valid `flow_loss` curve
against v2's at the same step rather than against itself.

## Contents

| file | what |
|---|---|
| `model.safetensors` | step-200000 EMA export |
| `training/checkpoint_00200000.pt` | the **training** checkpoint — needed to distill a new student, because `builders.py` does a plain `torch.load` and reads `payload["ema"]`, which the safetensors export has already merged in and cannot be recovered from |
| `config.yaml` | inference config (paths point into this repo) |
| `tokenizer/ne_en_9682.*` | the grafted tokenizer — **must** travel with the weights |
| `ne_frontend.py` | Nepali text frontend (number verbalization) |
| `training_args.yaml` | the exact training arguments |
| `eval_results.json` | every number above, machine-readable |
| `inference.py` | runnable example, with the threshold caveat |

## License and data

Trained on a Nepali corpus that is **not redistributable** — 86.6% of it is
YouTube audio; only the 296.6 h of AI4Bharat material is CC-BY. The weights are
shared privately for research and record-keeping; no voice prompt audio ships
with this repo, because the training corpus contains real people who did not
consent to having their voices redistributed as cloning prompts.
