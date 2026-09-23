---
license: other
language:
- ne
- en
pipeline_tag: text-to-speech
tags:
- pocket-tts
- nepali
- english
- bilingual
- teacher-checkpoint
---

# pocket-tts-nepali-en-24l-teacher-v4

The 24-layer bilingual (Nepali+English) teacher checkpoint that
[`milanakdj/pocket-tts-nepali-en-6l-v4`](https://huggingface.co/milanakdj/pocket-tts-nepali-en-6l-v4)
(the shippable 6-layer student) was depth-distilled from. Published mainly so a
future re-distillation doesn't have to retrain this stage — `builders.py` does a
plain `torch.load` and reads `payload["ema"]`, which only the training
checkpoint (`training/checkpoint_00250000.pt`, included here) has; the
`model.safetensors` export has already merged the EMA in and cannot be used for
that.

**This repo has no standalone eval.** The pocket-tts inference path has no
`cfg_coef`, and this teacher was only ever sampled with guidance
(`sample_cfg_coef 2.0`) during training and distillation — loading it through
`TTSModel` the normal way measures the *unguided* teacher, which is not the
model the student learned from and not representative of deployed quality. Use
the student for anything except re-distillation. See the student's model card
for the full bilingual eval and the training recipe.

## Training

Finetuned from Kyutai's English pocket-tts release, 250,000 steps, 5% English
replay (down from a failed v3 attempt at 27%), grafted tokenizer
(`tokenizer_v3/ne_en_9682.model`: Kyutai's 4,000 pieces kept byte-identical +
5,682 appended Nepali pieces), inherited English embedding rows left unfrozen.
Final valid `flow_loss`: Nepali -0.0698 (still inching down at 250k), English
0.1887 (flat since ~10k steps of 250k — the extra budget over v2's 200k bought
nothing on English and should not be repeated without a reason tied to
Nepali).

## Contents

| file | what |
|---|---|
| `model.safetensors` | step-250000 EMA export (unguided inference) |
| `training/checkpoint_00250000.pt` | the training checkpoint — needed to distil a new student |
| `config.yaml` | inference config (paths point into this repo) |
| `tokenizer/ne_en_9682.*` | the grafted tokenizer |
| `ne_frontend.py` | Nepali text frontend |
| `training_args.yaml` | exact training arguments |

## License and data

Same corpus, same redistribution restrictions as the student — see its model
card.
