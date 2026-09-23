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
- zero-shot-voice-cloning
---

# pocket-tts-nepali-en-6l-v4

A **bilingual Nepali+English** text-to-speech model with zero-shot voice cloning,
6-layer depth-distilled student, running **5.87x faster than real time on 4 CPU
threads**. This is v4 of this project's Nepali pocket-TTS line: the first
checkpoint that speaks both languages instead of one.

> **This is a candidate replacement, not yet promoted.** The production model is
> still [`milanakdj/pocket-tts-nepali-6l`](https://huggingface.co/milanakdj/pocket-tts-nepali-6l)
> (v2, Nepali-only). v4 matches v2 on Nepali and adds working English; promoting
> it is a separate decision pending a listening pass, not an automatic swap.

## Why this exists

v2 shipped Nepali-only: its tokenizer had no byte fallback, so every English
word, digit or hyphen silently collapsed to `<unk>` and got *deleted* rather than
mispronounced (CER 0.879–1.030 on English text — worse than saying nothing).

**v3 tried to fix this and failed outright** (CER 0.578 vs v2's 0.139 on
Nepali) by changing the tokenizer, the embedding init, the data mix and the
alignment pipeline all in one 72h run, with a training objective that looked
converged and was not — see the archived
[`milanakdj/pocket-tts-nepali-6l-v3-failed`](https://huggingface.co/milanakdj/pocket-tts-nepali-6l-v3-failed)
for the post-mortem.

**v4 changed one thing at a time**, validated each change against v2's own
logged loss curve at matched training steps (not against itself — the mistake
that hid v3's failure for 200k steps), and found the actual cause: v3's 27%
English replay ratio was starving the Nepali budget, not the grafted tokenizer.
Dropping replay to 5% and leaving the inherited English embedding rows
unfrozen recovered v2's exact Nepali trajectory while learning real English.

## Results

**Bilingual 2x2**: {Nepali, English} text × {Nepali, English} voice prompt, 50
utterances per cell, disjoint from training. Nepali text scored with
`himalaya-ai/whisper-large-v3-nepali-final` (base Whisper is blind on Nepali —
it rates real human Nepali speech at 0.917 WER); English text scored with base
`whisper-large-v3-turbo`. **The two WER/CER columns are not comparable to each
other** — each is only readable against its own `real_human` row in the same
cell. `real_human` is the genuine recording of the same target utterance
through the identical scoring pipeline: the ceiling, and the check that the
metric can see what it measures.

| system | cell | WER | CER | speaker sim |
|---|---|---|---|---|
| real human | en_text/en_voice | 0.024 | 0.011 | 0.571 |
| real human | en_text/ne_voice | 0.024 | 0.011 | 0.583 |
| real human | ne_text/en_voice | 0.347 | 0.120 | 0.590 |
| real human | ne_text/ne_voice | 0.347 | 0.120 | 0.727 |
| **v4 student (this model)** | en_text/en_voice | **0.441** | **0.356** | 0.709 |
| **v4 student (this model)** | en_text/ne_voice | **0.430** | **0.369** | 0.709 |
| **v4 student (this model)** | ne_text/en_voice | **0.323** | **0.111** | 0.821 |
| **v4 student (this model)** | ne_text/ne_voice | **0.313** | **0.134** | 0.829 |
| v2 student (Nepali-only, reference) | en_text/en_voice | 1.328 | 1.030 | — |
| v2 student (Nepali-only, reference) | en_text/ne_voice | 1.030 | 0.879 | — |
| v2 student (Nepali-only, reference) | ne_text/en_voice | 0.378 | 0.157 | — |
| v2 student (Nepali-only, reference) | ne_text/ne_voice | 0.311 | 0.117 | — |

**Nepali: matches v2** (CER 0.111–0.134 vs v2's 0.117–0.157, both essentially
at the 0.120 real-human ceiling). **English: goes from unusable to functional**
— CER 0.356–0.369, nowhere near the 0.011 English human ceiling (English is
only 5% of the training mix, so this is expected, not a bug), but a categorical
change from CER > 0.87 (worse than emitting silence, because deleted `<unk>`
tokens leave uncorrected insertions) to intelligible, ASR-transcribable speech.
Sample content check on a held-out sentence: reference *"If you are liking our
videos, then like them, comment them, and subscribe."* → generated audio
transcribes as *"If you are liking our videos, then like them, comment them."*
— correct words, an EOS-truncated tail, not garbage.

**Speaker similarity** is higher for this model than the real-human
cross-speaker floor in every cell (0.709–0.829 vs 0.571–0.727) — a clone sits
unnaturally close to its own prompt, the same pattern v2 showed. Not a red
flag; not comparable across the language boundary either (see caveat above the
table).

**CPU speed**: 5.87x real-time, 4 pinned threads, one H100-trained model
running consumer-CPU inference. (Thread count matters: unpinned threads on a
loaded box previously measured 0.15–0.17x RT on this same architecture and
read as "the distillation bought nothing" — it was a benchmarking artifact,
not a result. [See CLAUDE.md's CPU bench trap.])

## The EOS threshold — read this before you load the model

pocket-tts defaults to `eos_threshold=-4.0`. At that default, **this
checkpoint's English generations truncate early** (calibration-set English WER
0.925 at -4.0 vs 0.523 at 0.0 — nearly double the error, same content).
Nepali is insensitive to this knob across the range tested (WER 0.236 flat
from -4.0 to +2.0) — only English's EOS head is miscalibrated, consistent with
English carrying 20x less training signal than Nepali in this recipe (5%
replay).

`eos_threshold=0.0` was picked by sweeping on 40 utterances (20 Nepali + 20
English) **disjoint from the eval set above**, scoring actual transcribed
content (WER), not generated duration — duration alone is a documented lying
proxy in this codebase (a badly-EOS-calibrated model can produce the "right"
duration for the wrong reason, or vice versa).

It is not settable in `config.yaml` (`Config` is a strict schema with no such
field) — pass it to `load_model` every time:

```python
from pocket_tts.models.tts_model import TTSModel
model = TTSModel.load_model(
    config="hf://milanakdj/pocket-tts-nepali-en-6l-v4/config.yaml",
    eos_threshold=0.0,
)
```

## Usage

```python
import torch, scipy.io.wavfile as wav
from pocket_tts.models.tts_model import TTSModel

torch.set_num_threads(4)  # pin this -- see the speed caveat above
model = TTSModel.load_model(
    config="hf://milanakdj/pocket-tts-nepali-en-6l-v4/config.yaml",
    eos_threshold=0.0,
)
model.to("cpu")

state = model.get_state_for_audio_prompt("my_voice_3_to_5s.wav")
audio = model.generate_audio(state, "नमस्ते, तपाईंलाई कस्तो छ?", copy_state=True)
wav.write("out_ne.wav", model.config.mimi.sample_rate, audio.detach().cpu().numpy().squeeze())

audio = model.generate_audio(state, "Hello, how are you today?", copy_state=True)
wav.write("out_en.wav", model.config.mimi.sample_rate, audio.detach().cpu().numpy().squeeze())
```

Run text through `ne_frontend.py:normalize(text, keep_latin=True)` first —
**`keep_latin=True` is required**, not optional. `normalize()` defaults to
`keep_latin=False`, which is correct for v2 (no byte fallback, so it
transliterates Latin words into Devanagari phonetics to keep them
representable — "Hello" → "हेल्लो") and **wrong for this model**: `ne_en_9682`
encodes Latin script natively, which is the entire reason this checkpoint can
speak English at all, so `keep_latin=True` keeps English words as English
instead of spelling them out phonetically in Nepali script. Digits still get
verbalized either way (`normalize()` renders "४५" as Nepali words, not read
digit-by-digit), which is what most Nepali TTS use cases want.

```python
from ne_frontend import normalize, assert_clean
text = normalize(raw_text, keep_latin=True)
assert_clean(text, model="path/to/tokenizer/ne_en_9682.model")  # raises on any <unk>
```

## Try it in Google Colab

Runnable end-to-end, GPU or CPU runtime. This repo is **private**, so the
first cell needs a Hugging Face token with read access to
`milanakdj/pocket-tts-nepali-en-6l-v4` (create one at
[hf.co/settings/tokens](https://huggingface.co/settings/tokens) if you don't
already have one) — paste it when `login()` prompts.

**Cell 1 — install**

```python
!pip install -q pocket-tts huggingface_hub
```

**Cell 2 — authenticate** (repo is private)

```python
from huggingface_hub import login
login()   # paste your HF token when prompted
```

**Cell 3 — fetch `ne_frontend.py`** (it's a repo file, not part of the
`pocket-tts` pip package, so it has to be downloaded and put on `sys.path`
before it can be imported)

```python
import sys
from huggingface_hub import hf_hub_download

REPO = "milanakdj/pocket-tts-nepali-en-6l-v4"
frontend_path = hf_hub_download(repo_id=REPO, filename="ne_frontend.py")
sys.path.insert(0, frontend_path.rsplit("/", 1)[0])

tokenizer_path = hf_hub_download(repo_id=REPO, filename="tokenizer/ne_en_9682.model")
```

**Cell 4 — upload a voice prompt** (3–5s, mono, real speech — no sample ships
with this repo; see License and data below)

```python
from google.colab import files
uploaded = files.upload()
voice_path = next(iter(uploaded))
```

**Cell 5 — load and generate**

```python
import torch, scipy.io.wavfile as wav
from IPython.display import Audio, display
from pocket_tts.models.tts_model import TTSModel
from ne_frontend import normalize, assert_clean

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    torch.set_num_threads(4)   # pin this on CPU runtimes -- see the speed caveat above

model = TTSModel.load_model(
    config=f"hf://{REPO}/config.yaml",
    eos_threshold=0.0,          # required -- see the EOS section above, default (-4.0) truncates English early
)
model.to(DEVICE)
sr = model.config.mimi.sample_rate

state = model.get_state_for_audio_prompt(voice_path)   # slow step, reusable across sentences

def speak(text, out_file, seed=0):
    # Generation is NOT seeded by default -- every call samples fresh, so the
    # SAME text + SAME voice prompt gives a DIFFERENT take every run. This is
    # most visible on English (see the Known limitations section: English
    # speaker-similarity-to-prompt is both lower on average AND has a wider
    # run-to-run spread than Nepali's). torch.manual_seed() makes two calls
    # with identical arguments byte-identical -- verified on this checkpoint --
    # so pin a seed if you want a reproducible take rather than a fresh
    # sample each time. It does not fix the underlying identity gap on
    # English, only which particular (mediocre-to-good) sample you land on.
    torch.manual_seed(seed)
    text = normalize(text, keep_latin=True)   # keep_latin=True is required, see above
    assert_clean(text, model=tokenizer_path)
    audio = model.generate_audio(state, text, copy_state=True)
    x = audio.detach().cpu().numpy().squeeze()
    wav.write(out_file, sr, x)
    display(Audio(out_file))
    return x

speak("नमस्ते, तपाईंलाई कस्तो छ?", "out_ne.wav", seed=0)
speak("Hello, how are you today?", "out_en.wav", seed=0)
# Unhappy with the English take? Try other seeds -- it changes WHICH sample
# you get, not the average quality (temperature sweeps 0.3 -> 0.05 didn't
# move the mean speaker-similarity either, see Known limitations):
# speak("Hello, how are you today?", "out_en_seed7.wav", seed=7)
```

If Colab gives you a GPU runtime (`Runtime > Change runtime type > T4 GPU`),
`DEVICE` picks it up automatically and generation is near-instant; on CPU-only
runtimes it still runs at roughly the 5.87x-real-time speed measured above
with 4 pinned threads.

## How this was trained

Two stages, both on the same grafted tokenizer as v3
(`tokenizer_v3/ne_en_9682.model`: Kyutai's original 4,000 pieces kept
byte-identical, plus 5,682 appended Nepali pieces — proven innocent of v3's
failure by a matched-step Phase-1 probe before any full run started):

1. **24-layer teacher**, finetuned from Kyutai's English pocket-tts release,
   250,000 steps, **5% English replay** (down from v3's 27%), inherited
   embedding rows left unfrozen. Final valid `flow_loss`: Nepali -0.0698,
   English 0.1887 (flat from ~10k steps — the extra 50k steps over v2's
   200k-step budget bought nothing on English).
2. **6-layer student**, depth-distilled from the teacher with
   `distill_cfg_coef 2.0` (bakes in classifier-free guidance the inference
   package cannot express at all — the reason the student beats its own
   teacher on every measured axis in the v2/v3 lineage too), 200,000 steps.
   Final valid `distill_mse`: Nepali 0.01185, English 0.01120.

Three short controlled arms (7,500 steps each) established the recipe before
the full run: 27%→5% replay recovered v2's Nepali trajectory; **pinning the
inherited English embedding rows capped English** (led early, then plateaued
while the free-to-adapt arm overtook it — the model can never fit those rows
to its own acoustics if they're frozen); and at 0% replay English got *worse*
over training even with the embedding frozen, proving the backbone forgets,
not just the embedding. Full writeup: `docs/V4_RECIPE.md` in the training
repo.

## Contents

| file | what |
|---|---|
| `model.safetensors` | step-200000 EMA export |
| `config.yaml` | inference config (paths point into this repo) |
| `tokenizer/ne_en_9682.*` | the grafted tokenizer — **must** travel with the weights |
| `ne_frontend.py` | Nepali text frontend (number verbalization) |
| `training_args.yaml` | the exact training arguments |
| `eval_results.json` | every number above, machine-readable |
| `inference.py` | runnable example, with the EOS-threshold caveat baked in |

## Known limitations

- **English is 5% of the training mix.** It works; it is not native-quality.
  CER 0.356–0.369 against a 0.011 human ceiling on the same instrument.
- **English voice cloning is weak and unseeded generation makes it look
  worse than it is.** Measured on one prompt, 5 generations per language, no
  manual seed (matching naive usage): Nepali speaker-similarity-to-prompt
  0.833 ± 0.029 (resemblyzer cosine); English 0.535 ± 0.049 — both a much
  lower mean *and* a wider run-to-run spread. Two separate things are
  bundled in what that looks like to a user: (1) **no seed → a different
  sample every run**, confirmed by `torch.manual_seed(N)` before
  `generate_audio()` making two runs byte-identical — set a seed if you want
  reproducibility; (2) **the mean itself is low and doesn't move** — sweeping
  `temp` from 0.3 down to 0.05 left English sim flat at 0.50–0.51, so this is
  not a sampling-temperature problem to tune away, it is 5% English data not
  being enough to learn strong speaker conditioning for that language. The
  bilingual 2x2 table above already shows this shape (sim keys off
  *text* language, not voice-prompt language: ~0.82–0.83 for any ne_text
  cell, ~0.71 for any en_text cell) — this section just makes explicit what
  that means for a single voice heard across repeated runs. Fixing it needs
  more English training data, not a different inference setting.
- **Code-mixed input is untested by construction.** The training corpus holds
  only 1.2h / 371 rows (0.05%) of naturally code-mixed speech, so any
  code-switching ability is a generalization test, not a trained capability.
- **Not yet checked against Nepal-native speaker accent.** Same open question
  as v2 — part of the training corpus (IndicVoices-R Nepali, Rasa) is West
  Bengal Nepali speech, not Nepal-native, and the filter to exclude it was
  never applied to this training mix either.
- **No human listening pass yet.** These are ASR/embedding-similarity metrics
  only; promoting this model to replace v2 in production should wait for a
  blind listen, especially on the English cells.

## License and data

Trained on a Nepali+English corpus that is **not redistributable** — the
Nepali portion is 86.6% YouTube audio, only 296.6h of AI4Bharat material is
CC-BY; the English portion is likewise sourced from YouTube. The weights are
shared privately for research and record-keeping; no voice-prompt audio ships
with this repo, because the training corpus contains real people who did not
consent to having their voices redistributed as cloning prompts.
