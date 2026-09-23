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

# pocket-tts-nepali-6l-v3 — FAILED RUN, ARCHIVED FOR THE RECORD

> **Do not use this model.** It does not produce intelligible Nepali. It is
> published private, as a record of a training run that failed, so the failure is
> reproducible and a future attempt has a baseline to beat.
>
> **The working Nepali model is [`milanakdj/pocket-tts-nepali-6l`](https://huggingface.co/milanakdj/pocket-tts-nepali-6l)** (v2).

## What this was meant to be

v3 of the Nepali pocket-TTS student: a 6-layer model depth-distilled from a
24-layer teacher, with a **grafted tokenizer** — Kyutai's original 4,000 pieces
kept byte-identical, plus 5,682 appended Nepali pieces — so that English words,
digits and punctuation would be representable instead of collapsing to `<unk>`.
v2's tokenizer had no byte fallback and *deleted* any Latin word or ASCII digit
rather than mispronouncing it. v3 was supposed to fix exactly that.

It trained to completion: 200,000 steps, no crash, a loss curve that was flat
from ~100k.

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

| teacher, valid @ 200k | flow_loss | eos_loss |
|---|---|---|
| v2 | **-0.0746** | 0.0078 |
| v3 | **+0.0831** | 0.0276 |

v3 at step 200,000 is worse than v2 was at step 5,000. The flat curve read as
convergence only because nobody put the previous generation's curve next to it.

## The trap, if you load it anyway

At pocket-tts's default `eos_threshold=-4.0` this checkpoint ends the utterance
after a few frames — about **0.26x** the duration it should produce, silently,
with no error. Passing `eos_threshold=0.0` restores the *length*. It does not
restore the *content*: CER stays flat at 0.51–0.58 across every threshold. Early
stopping is a symptom of a badly fit model, not a configuration mistake.

`eos_threshold` cannot be set in `config.yaml` — pocket-tts's `Config` schema is
strict and has no such field — so it has to be passed to `load_model`.

```python
from pocket_tts.models.tts_model import TTSModel
model = TTSModel.load_model(
    config="hf://milanakdj/pocket-tts-nepali-6l-v3-failed/config.yaml",
    eos_threshold=0.0,   # without this you get ~a quarter of the audio
)
```

## Probable cause

The model speaks **English better than Nepali** — it renders "The quick brown fox
jumps over the lazy dog." verbatim while its Nepali is unintelligible.

v2 reset the entire text embedding and learned 4,000 Nepali pieces from scratch
with 100% of the signal on Nepali. v3 inherited the pretrained English rows and
had to learn **5,682 fresh Nepali rows** — 1.4x more pieces, each seen
proportionally less often — while roughly 27% of the corpus was English replay.
The graft delivered what it promised, and it was paid for out of the Nepali
budget.

That is a hypothesis consistent with every measurement, not a proven cause.

## Contents

| file | what |
|---|---|
| `model.safetensors` | step-200000 EMA export |
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
