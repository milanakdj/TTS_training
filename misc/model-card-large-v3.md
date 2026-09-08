---
language:
- ne
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- nepali
- asr
- speech-recognition
- finetuned
base_model: openai/whisper-large-v3
datasets:
- lilgoose7777/slr-combined-nepali-tts2
metrics:
- wer
- cer
model-index:
- name: whisper-large-v3-nepali-final-largev3_1
  results:
  - task:
      type: automatic-speech-recognition
      name: Automatic Speech Recognition
    dataset:
      name: slr-combined-nepali-tts2
      type: lilgoose7777/slr-combined-nepali-tts2
      split: test
    metrics:
    - type: wer
      value: 10.98
      name: WER
    - type: cer
      value: 3.43
      name: CER
---

# Whisper large-v3 — Nepali ASR

Nepali speech-to-text, fine-tuned from [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3).

**Test WER 10.98% · Test CER 3.43%** on 5,000 held-out clips.

> ## ⚠️ Read this before you write any code
>
> **This model does not work with plain `model.generate()`.** It also does not work
> with `pipeline("automatic-speech-recognition")`, faster-whisper, whisper.cpp, or
> any tool that assumes the standard Whisper decoder prefix.
>
> It was trained with a **doubled `<|startoftranscript|>`** token in the decoder
> prefix, so it must be decoded with that same prefix. Use the `transcribe()`
> function in [Quickstart](#quickstart) — it is self-contained and handles this.
>
> | Decoder prefix | WER |
> |---|---|
> | `[sot, sot, ne, transcribe, notimestamps]` — correct for this model | **10.98%** |
> | `[sot, ne, transcribe, notimestamps]` — what `generate()` sends | 465% (nonsense) |
>
> The weights are sound. Only the prefix convention is non-standard. See
> [Known issue](#known-issue-doubled-decoder-prefix) for the cause and the fix.

---

## Quickstart

```python
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = "milanakdj/whisper-large-v3-nepali-final-largev3_1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(MODEL_ID, language="nepali", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE).eval()


def transcribe(audio_batch, max_new_tokens=225):
    """Transcribe Nepali audio with the doubled-sot prefix this model requires.

    audio_batch: list of 1-D float32 numpy arrays, 16 kHz mono.
    returns:     list of strings, same order.
    """
    tok = processor.tokenizer
    tid = tok.convert_tokens_to_ids
    # The doubled <|startoftranscript|> is the whole point. One sot decodes garbage.
    prefix = [tid("<|startoftranscript|>"), tid("<|startoftranscript|>"),
              tid("<|ne|>"), tid("<|transcribe|>"), tid("<|notimestamps|>")]
    eos = tid("<|endoftext|>")

    feats = processor.feature_extractor(
        audio_batch, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE, dtype=model.dtype)

    cur = torch.tensor([prefix] * len(audio_batch), device=DEVICE)
    out, past = cur, None
    done = torch.zeros(len(audio_batch), dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        enc = model.get_encoder()(feats)
        for _ in range(max_new_tokens):
            res = model(encoder_outputs=enc, decoder_input_ids=cur,
                        past_key_values=past, use_cache=True)
            past = res.past_key_values
            nxt = res.logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
            out = torch.cat([out, nxt[:, None]], dim=1)
            done |= nxt == eos
            if bool(done.all()):
                break
            cur = nxt[:, None]
    return processor.batch_decode(out, skip_special_tokens=True)
```

### One file

```python
import librosa

audio, _ = librosa.load("clip.wav", sr=16000)
print(transcribe([audio])[0])
```

### A batch

```python
clips = [librosa.load(p, sr=16000)[0] for p in ["a.wav", "b.wav", "c.wav"]]
for path, text in zip(["a.wav", "b.wav", "c.wav"], transcribe(clips)):
    print(f"{path}: {text}")
```

Audio longer than 30 seconds must be split into chunks first — Whisper's encoder
takes a fixed 30-second window.

---

## Results

Scored on the held-out test split: the same dataset slice, the same shuffle seed
(42), and the same 80/10/10 carve used during training, so no test clip was ever
trained on.

| Metric | Value |
|---|---|
| Test WER | **10.98%** |
| Test CER | **3.43%** |
| Clips scored | 5,000 of 17,700 |
| Decoding | greedy, doubled-sot prefix, max 225 new tokens |
| Checkpoint | `checkpoint-26550` (end of epoch 3) |

### Sample predictions

| Prediction | Reference |
|---|---|
| आधुनिक र उत्तराधुनिक | आधुनिक र उत्तरआधुनिक |
| त्यसोमा ४४ | तेस्रोमा ४४ |
| नेपाली फौजले डोटीमाथि | नेपाली फौजले डोटीमाथि |

### Per-epoch numbers from the training log

These are **not** comparable to the test WER above. The training loop decoded with
the standard single-sot prefix, which this model cannot use, so its WER is
meaningless. The loss is the informative column.

| Epoch | eval_loss | eval_wer (broken decode) |
|---|---|---|
| 1 | 0.0568 | 377.1% |
| 2 | 0.0438 | 893.6% |
| 3 | 0.0402 | 588.5% |

A loss of 0.04 with a WER in the hundreds is the signature of the prefix bug: the
model predicts near-perfectly under teacher forcing and cannot generate on its own.

---

## Intended use

- Transcribing Nepali speech in clean, close-microphone recordings.
- A starting point for further fine-tuning on Nepali audio.
- Research and benchmarking of Nepali ASR.

### Out of scope

- **Noisy, far-field, or multi-speaker audio.** The training data is clean
  single-speaker studio recordings. Expect a large accuracy drop in the wild.
- **Any drop-in Whisper pipeline.** The prefix convention breaks standard tooling.
- Medical, legal, safety, or any setting where a transcription error causes harm.
  This model has not been validated for such use.

---

## Limitations and bias

- **One voice.** The corpus is single-speaker text-to-speech style audio. The model
  has effectively not learned speaker variation, accent variation, or noise
  robustness. A 10.98% WER here does not predict 10.98% on real recordings.
- **Reading style.** The source text is read prose. Spontaneous speech, hesitation,
  and code-switching between Nepali and English are under-represented.
- **Domain.** Vocabulary follows the source corpus. Names, technical terms, and
  numerals outside it will be weaker.
- **No timestamps.** Decoding forces `<|notimestamps|>`. Timestamped output was
  never trained and is not supported.
- **Non-standard decoding.** See below.

---

## Training

### Data

| Parameter | Value |
|---|---|
| Dataset | [`lilgoose7777/slr-combined-nepali-tts2`](https://huggingface.co/datasets/lilgoose7777/slr-combined-nepali-tts2) |
| Rows requested | 177,000 |
| Cleaning | rows with empty or missing `text` dropped |
| Shuffle seed | 42 |
| Split | 80% / 10% / 10% |
| Train / val / test | 141,600 / 17,700 / 17,700 |
| Audio | 16 kHz mono, log-mel, 128 bands |
| Label cap | 448 tokens |

### Procedure

| Parameter | Value |
|---|---|
| Base model | `openai/whisper-large-v3` (1.54B params) |
| Language / task | `nepali` / `transcribe` |
| Epochs | 3 (26,550 optimizer steps) |
| Learning rate | 5e-6, linear decay, 5% warmup |
| Per-device batch | 1 |
| Gradient accumulation | 16 (effective batch 16) |
| Optimizer | Adafactor |
| Precision | bf16 |
| Gradient checkpointing | on |
| Final train loss | 0.0137 |
| Wall clock | ~79.7 hours on one shared NVIDIA GB10 (DGX Spark) |

Adafactor and gradient checkpointing were not stylistic choices: the GPU was
shared with another workload, leaving roughly a 14-20 GB slice. Adafactor's
factored second moment keeps the optimizer state in megabytes instead of
gigabytes, which is what made a 1.54B-parameter fine-tune fit at all.

Per-epoch checkpoints: [`milanakdj/whisper-large-v3-nepali-checkpoints`](https://huggingface.co/milanakdj/whisper-large-v3-nepali-checkpoints)

---

## Known issue: doubled decoder prefix

### What happened

The fine-tuning collator stripped the leading `<|startoftranscript|>` from the
labels only when this held:

```python
labels[:, 0] == tokenizer.bos_token_id
```

For Whisper, `bos_token_id` is `<|endoftext|>` (50257). Labels start with
`<|startoftranscript|>` (50258). The condition is never true, so the token was
never stripped. `Seq2SeqTrainer` then prepended `decoder_start_token_id` again
while shifting labels into `decoder_input_ids`:

```
trained on :  [sot, sot, lang, task, notimestamps, w1, w2, ...]
generate() :  [sot,      lang, task, notimestamps, ...]
```

Every decoder position is off by one. The correct line compares against
`decoder_start_token_id`.

### Why it went unnoticed for 80 hours

Teacher forcing hides it completely. `eval_loss` fell to 0.04 and kept falling,
because during training the model is always fed the doubled prefix it learned.
Only autoregressive generation exposes the mismatch, and generation was measured
by a metric nobody read until the run finished.

**Lesson worth copying:** if `eval_wer` exceeds 100% while `eval_loss` is low,
stop the run. That is not a weak model. That is broken generation.

### How to fix it properly

Feeding the doubled prefix (as `transcribe()` above does) recovers full accuracy,
but leaves the model incompatible with standard tooling. The permanent fix is a
short corrective fine-tune from these weights with the collator repaired: a few
hundred steps at a low learning rate, encoder frozen, is enough to relearn the
standard prefix without relearning Nepali. That is roughly 1-3 hours, not another
80.

---

## Reproducing the evaluation

```bash
python whisper-eval.py \
  --ckpt milanakdj/whisper-large-v3-nepali-final-largev3_1 \
  --n 5000 --batch 16 --n-sot 2
```

The script rebuilds the identical test split from the dataset (same row slice,
same seed, same carve), so the number is reproducible from scratch.

---

## Citation

```bibtex
@misc{whisper-large-v3-nepali-2026,
  title  = {Whisper large-v3 fine-tuned for Nepali ASR},
  author = {milanakdj},
  year   = {2026},
  url    = {https://huggingface.co/milanakdj/whisper-large-v3-nepali-final-largev3_1}
}
```

Base model: Radford et al., *Robust Speech Recognition via Large-Scale Weak
Supervision*, 2022 — [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3).

## License

Apache 2.0, following the base model.
