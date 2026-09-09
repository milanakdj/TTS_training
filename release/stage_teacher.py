"""Stage the 24-layer teacher for its own repo.

The teacher is slower AND worse than the student it produced -- 1.53x vs 5.04x
real-time, WER 0.402 vs 0.322. It is published only as a distillation source, so the
card has to say that loudly enough that nobody downloads 1.34 GB expecting the
bigger model to be the better one.

    python3 stage_teacher.py <repo_id>
"""
import os, shutil, sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "milanakdj/pocket-tts-nepali-24l-teacher"
SRC = "/root/tts/TTS_training/pocket_TTS"
OUT = "/workspace/hf_release/staged_teacher"
WEIGHTS = f"{SRC}/runs/nepali_teacher_24l/model.safetensors"

os.makedirs(f"{OUT}/tokenizer", exist_ok=True)
for src, dst in [(WEIGHTS, f"{OUT}/model.safetensors"),
                 (f"{SRC}/tokenizer/nepali_bpe4000.model", f"{OUT}/tokenizer/nepali_bpe4000.model"),
                 (f"{SRC}/tokenizer/nepali_bpe4000.vocab", f"{OUT}/tokenizer/nepali_bpe4000.vocab")]:
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
        shutil.copy(src, dst)
        print(f"copied {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.1f} MB)")

cfg = open(f"{SRC}/infer/nepali_teacher_24l.yaml").read()
cfg = cfg.replace(f"weights_path: {SRC}/runs/nepali_teacher_24l/model.safetensors",
                  f"weights_path: hf://{REPO}/model.safetensors")
cfg = cfg.replace(f"tokenizer_path: {SRC}/tokenizer/nepali_bpe4000.model",
                  f"tokenizer_path: hf://{REPO}/tokenizer/nepali_bpe4000.model")
assert "hf://" in cfg and SRC not in cfg, "path rewrite failed"
open(f"{OUT}/config.yaml", "w").write(cfg)
print("wrote config.yaml")

open(f"{OUT}/README.md", "w").write(f"""---
license: cc-by-4.0
language:
- ne
library_name: pocket-tts
pipeline_tag: text-to-speech
base_model: kyutai/pocket-tts
base_model_relation: finetune
tags:
- text-to-speech
- nepali
- teacher-model
---

# Nepali Pocket-TTS — 24-layer teacher (distillation source)

> ## Do not use this model for inference. Use the student.
>
> This is the 24-layer teacher that produced
> **[himalaya-ai/pocket-tts-nepali-6l](https://huggingface.co/himalaya-ai/pocket-tts-nepali-6l)**.
> The student is **three times faster, three times smaller, and measurably better**.
> The teacher is published only so the distillation can be reproduced.

| | teacher (this repo) | student |
|---|---|---|
| backbone layers | 24 | 6 |
| parameters | 336.1M | **109.5M** |
| size | 1.34 GB | **438 MB** |
| CPU speed | 1.53x real-time | **5.04x real-time** |
| WER | 0.402 | **0.322** |
| CER | 0.177 | **0.139** |
| speaker similarity | 0.826 | **0.841** |

## Why the smaller model wins

Both models generated the same 100 held-out utterances. The student is ahead on CER
in all six kinds of speech in the corpus individually, so this is not one category
carrying an average.

The cause is guidance. Distillation targets were computed with classifier-free
guidance at coefficient 2.0, so the student learned the teacher's *guidance-corrected*
output distribution. **Guidance does not exist anywhere in the Pocket-TTS inference
path** — `grep -rn cfg pocket_tts/` returns nothing outside the word "config" — so a
deployed teacher can only ever be sampled unguided. The student absorbed a capability
the teacher cannot use at inference time.

This is a statement about *deployable* configurations. It is **not** a claim that 6
layers hold more knowledge than 24. If guidance were implemented at inference, this
teacher would very likely be the better model — but it would need two forward passes
per frame, landing near 0.77x real-time, i.e. slower than playback.

## When this model is the right choice

- Reproducing or extending the depth distillation.
- Distilling a different student size or shape.
- Research into what the guidance step actually contributes.

For synthesizing Nepali speech, use the student.

## Usage

Same API as the student; only the config differs. Requires
`huggingface-cli login` with an approved account.

```python
from pocket_tts.models.tts_model import TTSModel
import torch
torch.set_num_threads(4)   # pin threads; unpinned measurements are meaningless
model = TTSModel.load_model(config="hf://{REPO}/config.yaml")
model.to("cpu")
state = model.get_state_for_audio_prompt("your_voice_3_to_5s.wav")
audio = model.generate_audio(state, "नमस्ते, तपाईंलाई कस्तो छ?")
```

## Training

Kyutai's 24-layer Pocket-TTS English release, finetuned on roughly 2,000 hours of
Nepali speech for 200,000 steps at batch size 64, with a fresh Nepali BPE-4000
tokenizer and a reinitialised text embedding. The corpus is mostly YouTube-derived
audio plus open-source Nepali speech datasets; **it is not redistributable and is not
released.** Among the open-source portion,
[`ai4bharat/indicvoices_r`](https://huggingface.co/datasets/ai4bharat/indicvoices_r)
and [`ai4bharat/Rasa`](https://huggingface.co/datasets/ai4bharat/Rasa) are CC-BY-4.0
and are named because that licence requires attribution.

Nepali only — the text embedding was reset, so there is no English capability.

## Attribution

CC-BY-4.0, matching the upstream weights. Architecture, training code, the English
checkpoint this was finetuned from, and the unmodified Mimi codec weights are
Kyutai's: <https://github.com/kyutai-labs/pocket-tts>.
""")
print("wrote README.md")
print(f"\nstaged for {REPO} -> {OUT}")
print("\n".join(f"  {f}" for f in sorted(os.listdir(OUT))))
