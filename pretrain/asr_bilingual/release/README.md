---
language:
- ne
license: other
library_name: nemo
tags:
- automatic-speech-recognition
- speech
- nepali
- rnnt
- nemo
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
pipeline_tag: automatic-speech-recognition
---

# nemotron-asr-nepali-0.6b

Nepali ASR, finetuned from
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
(0.6B prompt-conditioned streaming RNN-T).

**Mid-training snapshot**, exported at global step ~147,700 of a planned 418,187.
Published to preserve progress, not as a finished release. See
[`WHY_THIS_REPO.md`](WHY_THIS_REPO.md) for what is stored here and why.

## Results

600-clip held-out Nepali gold test, `jiwer` on raw manifest text, measured **on the
exact `.nemo` published here**:

| | WER | CER |
|---|---|---|
| base model, before finetuning | 1.440 | 0.878 |
| **this checkpoint** | **0.204** | **0.076** |

Normalised (language tags, punctuation and case stripped — diagnostic only):
WER 0.186 / CER 0.068.

The base model exposes a `ne-NP` prompt slot but was never trained on Nepali, hence
the 1.440 starting point: this is teaching a language, not adapting one.

## Inference

> **Read this section before using the model.** It is prompt-conditioned, and the
> two obvious ways to call it both fail. Everything below was verified against this
> exact `.nemo`.

You must pass a **NeMo jsonl manifest** in which every row carries three fields:
`lang`, `target_lang` and `prompt_mode`. Nothing else works.

```python
import json
import soundfile as sf
import nemo.collections.asr as nemo_asr

m = nemo_asr.models.ASRModel.restore_from("nemotron_ne_nepali_best.nemo").eval()

clips = ["/path/a.wav", "/path/b.wav"]

rows = []
for p in clips:
    info = sf.info(p)                       # don't hardcode the duration
    rows.append({
        "audio_filepath": p,
        "duration": info.frames / info.samplerate,
        "text": "",                 # ignored at inference, but the key must exist
        "lang": "ne-NP",
        "target_lang": "ne-NP",     # selects prompt index 46
        "prompt_mode": "langID",    # forces that index -- see the warning below
    })

with open("infer.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

hyps = m.transcribe("infer.jsonl", batch_size=8, num_workers=0,
                    verbose=False, target_lang="ne-NP")
hyps = hyps[0] if hyps and isinstance(hyps[0], list) else hyps
print([h.text if hasattr(h, "text") else str(h) for h in hyps])
```

### Required manifest fields (all verified by testing)

Yes — **one row per clip, giving the audio path and its length.** Both are
mandatory, and so is `text`:

| field | required? | what happens if you omit / get it wrong |
|---|---|---|
| `audio_filepath` | **yes** | — |
| `duration` | **yes** | omitting it raises `KeyError: 'duration'` |
| `text` | **yes** (may be `""`) | omitting it raises `ValueError: Expected either str or list input, but got NoneType` |
| `lang` / `target_lang` | **yes** | selects the prompt (`ne-NP` -> index 46) |
| `prompt_mode` | **yes** (`"langID"`) | omitting it triples WER, silently — see above |

**A wrong `duration` does not truncate your audio.** Measured on a 7.102 s file:
declaring `0.5` or `14.204` both still loaded the full 113,632 samples (7.102 s).
The value is used for duration *filtering* and bucketing, not for slicing. So an
overstated value can get a row **dropped** by a `max_duration` filter (you then get
back fewer hypotheses than you had clips) — but it will never silently cut audio
off. Compute it properly regardless; it costs nothing:

```python
import soundfile as sf
info = sf.info(path)
duration = info.frames / info.samplerate
# librosa.get_duration(path=path) or torchaudio.info(path) work equally well
```

**Sample rate is handled for you.** The dataloader resamples to 16 kHz
automatically — a 24 kHz file loaded as 16 kHz without complaint. Mono is expected.

Clips longer than ~20 s were not seen in training (`max_duration=20`), so segment
long audio rather than relying on the model to cope.

`num_workers=0` keeps output order identical to the manifest; do not assume
ordering if you raise it.

### Two failure modes, both verified

**1. A plain list of file paths does not work at all.**

```python
m.transcribe(["a.wav", "b.wav"], target_lang="ne-NP")   # raises
# ValueError: Unknown prompt key: 'None'
```

`transcribe()` accepts `target_lang`, but it is only a fallback used when the
dataloader emits no prompt indices (`_transcribe_forward` in
`rnnt_bpe_models_prompt.py` reads `batch[4]` first). With list input the lhotse
dataloader *does* build a prompt tensor, and it logs
`configuration keys ignored by Lhotse dataloader: default_lang, ...` — so the
language never reaches it and the lookup gets `None`.

**2. Omitting `prompt_mode` silently triples your error rate.**

`LhotseSpeechToTextBpeDatasetWithPromptIndex` reads the mode from
`cut.custom["prompt_mode"]` and otherwise falls back to
`default_prompt_mode='unified'`, which routes ~50% of utterances through the
language-agnostic `auto` prompt (index 101) instead of `ne-NP` (index 46).
Measured on the same 600 clips with this model:

| manifest | WER | CER |
|---|---|---|
| `prompt_mode: langID` | **0.204** | **0.076** |
| `prompt_mode` omitted (`unified` default) | 0.610 | 1.228 |

The dangerous part is that this is **not flaky**. The dataloader RNG is seeded, so
a manifest missing `prompt_mode` returns the same degraded output every run and
looks perfectly stable. Always set the field.

## Limitations

- **English does not work.** The base model is multilingual; this finetune was
  trained on a 90.6%-Nepali mixture and English catastrophically regressed —
  `en-US` audio decodes to repeated Devanagari characters (test WER 1.000,
  CER 2.898). Both the `en-US` and `auto` prompt slots are unusable. Use the base
  model for non-Nepali audio. Full analysis in
  [`POSTMORTEM_english_collapse.md`](POSTMORTEM_english_collapse.md); per-checkpoint
  numbers for both languages in `eval_history.csv`.
- Mid-training checkpoint; training was stopped early because Nepali had flattened.
- Evaluated on one 600-clip internal gold test of read/prompted speech. **No public
  benchmark numbers** (FLEURS, Common Voice), so these are not directly comparable
  to published Nepali ASR results.
- Not evaluated for code-switching, spontaneous/conversational speech, telephony
  audio, or children's speech.
- Reference transcripts contain Latin-script glosses (e.g. `पिनकोड (Pincode)`) that
  the model correctly omits, so raw WER slightly overstates true error (~0.008).

## Training

- Base `nvidia/nemotron-3.5-asr-streaming-0.6b`, RNN-T, tokenizer unchanged. The
  tokenizer splits Nepali at ~5.12 tokens/word (near character level) vs 2.39 for a
  proper Indic BPE, inflating RNN-T target length ~2.1x — paid deliberately, since
  replacing the vocabulary would reinitialise the joint and prediction nets.
- Data: 2,207 h — 1,906 h Nepali + 301 h English replay (far too small to preserve
  English; see Limitations). Segmented to 0.3–20 s. Predominantly web/YouTube-derived
  Nepali audio plus AI4Bharat corpora (indicvoices-r, rasa) and mahadhwani.
- AdamW, cosine schedule, peak LR 1e-4, 2,000 warmup steps, bf16-mixed, lhotse
  duration-bucketed batching at 200 s/batch, single H100.

## Licence and data provenance

Released as `other`. The training corpus is predominantly YouTube-derived Nepali
audio and is **not itself redistributable**; only model weights and manifests
(path + transcript references, no audio) are stored here. Downstream users are
responsible for confirming compatibility with their use case. NVIDIA's licence
terms for the base model apply.
