# Nepali Pocket-TTS

A Nepali text-to-speech model with zero-shot voice cloning that runs **5.04x faster
than real time on CPU**. Two stages: a 24-layer teacher finetuned from Kyutai's
English release, then a 6-layer student depth-distilled from it. **The student is
the shippable model and it beats the teacher on every measured axis.**

Both runs are finished (200,000 steps each, 2026-09-07 and 2026-09-08). Nothing is
training right now.

| Document | Read it for |
|---|---|
| this file | commands, layout, traps |
| [`docs/ADR.md`](docs/ADR.md) | every decision and why it was taken |
| [`docs/INTERN_GUIDE.md`](docs/INTERN_GUIDE.md) | how the model works, from the basics |

---

## Results

100 held-out utterances, 77 speaker identities, all 6 corpus sources. The voice
prompt is a *different* clip of the same identity from the train split; the target
utterance is strictly held out. `real human` is the genuine recording of the same
utterance scored through the identical pipeline — **a ceiling and a check on the
metric, not a competitor.**

| system | WER | CER | speaker sim | CPU speed |
|---|---|---|---|---|
| real human (ceiling) | 0.343 | 0.122 | 0.807 | — |
| teacher 24L | 0.402 | 0.177 | 0.826 | 1.53x RT |
| **student 6L** | **0.322** | **0.139** | **0.841** | **5.04x RT** |

Student wins CER in all six sources individually. It beats the teacher because
guidance was baked in during distillation (`distill_cfg_coef: 2.0`) and guidance
does not exist in the inference package at all, so a deployed teacher can only ever
run unguided — [ADR-005](docs/ADR.md#adr-005).

**Two numbers that need their caveat every time they are quoted:**

- The student's WER is *below* the human row. That is **not** superhuman quality —
  the human clips are spontaneous with disfluencies and synthetic read speech is
  easier for an ASR. WER has saturated as a discriminator here.
- Speaker similarity rates both TTS systems *above* real human cross-utterance
  similarity. A clone sits unnaturally close to its own prompt. Teacher-vs-student
  only.

| model | params | on disk | weights |
|---|---|---|---|
| student 6L | 109.5M (75.5M backbone + 9.8M flow head + 4.2M emb + 20.1M Mimi) | 438 MB | `/workspace/nepali_student_6l/model_final_200k.safetensors` |
| teacher 24L | 336.1M (302.1M backbone, rest identical) | 1.34 GB | `runs/nepali_teacher_24l/model.safetensors` |

Every removed parameter came out of the backbone; the flow head and Mimi are copied
and frozen. Mimi is only ~19% of the teacher's per-frame cost, so there is no Amdahl
ceiling hiding in the speedup.

---

## Layout

```
pocket_TTS/
├── repo/                      Kyutai pocket-tts checkout + our training/configs/*.yaml
│   └── .venv/                 the interpreter for anything touching pocket_tts
├── manifests/                 train/valid jsonl; *_v2_aligned.jsonl is what training reads
├── tokenizer/                 nepali_bpe4000.model — MUST travel with the weights
├── runs/nepali_teacher_24l/   stage 1 output
├── scripts/                   manifest building, alignment, train launchers
├── infer/                     inference configs, benchmarks, generation
│   └── final/                 the evaluation harness and its results
├── docs/                      ADR.md, INTERN_GUIDE.md
└── train_{teacher,student}.log
```

Checkpoints live on `/workspace/nepali_student_6l/`, not here: `ckpt_freq 2500` over
200k steps is ~54 GB and `/` was 88% full — [ADR-011](docs/ADR.md#adr-011).

---

## Commands

**Train** (each ~36 h on one H100; both are already done):

```bash
./scripts/train_teacher.sh     # stage 1, training/configs/nepali_finetune.yaml
./scripts/train_student.sh     # stage 2, training/configs/nepali_distill.yaml
```

**Watch a run:**

```bash
tail -f train_student.log
grep "valid @" train_student.log | tail -20      # ours flattened at 0.0118
```

**Generate one clip** (CPU, ~1 s for 5 s of audio):

```bash
cd repo && OMP_NUM_THREADS=4 .venv/bin/python3 - <<'PY'
import torch, scipy.io.wavfile as wav
from pocket_tts.models.tts_model import TTSModel
torch.set_num_threads(4)
m = TTSModel.load_model(config="/root/tts/TTS_training/pocket_TTS/infer/nepali_student_6l.yaml")
m.to("cpu")
state = m.get_state_for_audio_prompt("prompt_3_to_5s.wav")   # reusable across sentences
a = m.generate_audio(state, "नमस्ते, तपाईंलाई कस्तो छ?")
wav.write("out.wav", m.config.mimi.sample_rate, a.detach().cpu().numpy().squeeze())
PY
```

**Full evaluation** — regenerates every number above, idempotent, ~20 min:

```bash
cd infer/final && bash run_final.sh
```

It waits for a training PID if one is running, snapshots the export, purges stale
audio, benchmarks, synthesizes 200 clips, scores WER twice and similarity once,
builds the blind A/B set and prints the report. Stages in isolation:

```bash
python3 build_pairs.py                                    # eval set -> pairs.json
cd ../../repo && GEN_THREADS=16 .venv/bin/python3 ../infer/final/gen_final.py
$ASR_VENV  score_wer.py        # base whisper — kept only to show it is unusable
$SIM_VENV  score_wer_ft.py     # Nepali-finetuned whisper — the real instrument
$SIM_VENV  score_sim.py        # resemblyzer speaker similarity
python3 make_listen.py summarize.py
```

`EVAL_LIMIT=2` on any scorer runs it on two utterances — use it before committing
to a long run.

**Which interpreter.** There is no single venv with everything:

| for | interpreter |
|---|---|
| anything importing `pocket_tts` | `repo/.venv/bin/python3` |
| base whisper WER | `synthetic_pipeline/asr_qc_service/.venv/bin/python3` |
| Nepali-finetuned WER, speaker similarity | `synthetic_pipeline/vc_service/.venv/bin/python3` |

---

## Traps

**Pin CPU threads for any latency measurement.** The first benchmark called
`torch.set_num_threads(os.cpu_count())` = 16 on a 16-core box the training job held
at load 13.6. Both models thrashed identically and measured 0.15x / 0.17x real
time — "the distil bought nothing". Pinned to 4: **1.53x / 5.04x**, a real 3.3x.
When a benchmark says an architectural change bought nothing, suspect the harness.
[ADR-009](docs/ADR.md#adr-009)

**The tokenizer travels with the weights.** A mismatched
`flow_lm.lookup_table.tokenizer_path` produces fluent nonsense and no error.

**Base Whisper cannot score Nepali.** `whisper-large-v3-turbo` scores *real human*
held-out Nepali at 0.917 WER. Use `himalaya-ai/whisper-large-v3-nepali-final`
(0.343 on the same clips) — private, org-owned, and it needs a **doubled**
`<|startoftranscript|>` decoder prefix or it emits nonsense above 400% WER. The
prefix cannot be expressed through `pipeline()`, faster-whisper or a plain
`generate()`; use the loop in `infer/final/score_wer_ft.py`.

**Whisper silently romanizes some Nepali clips**, giving CER ≈ 1.0 on fine audio.
Filter on a Devanagari-character ratio and discard the *measurement*, not the clip.
That is why `n` varies between rows.

**Key speaker identity on (directory, diarization label).** A bare `SPEAKER_01` is
per-video, so two videos' `SPEAKER_01` are different people and keying on the label
alone silently pairs unrelated voices.

**`gen_final.py` skips ids that already have a wav.** Audio left from an earlier
run — different weights — silently enters your results. `run_final.sh` purges
first; do the same by hand.

**Never edit a bash script while it is executing.** Bash reads by byte offset and
will resume mid-token. Copy it aside and run the copy.

**Always score real human audio as a ceiling.** This project lost weeks twice to
metrics that could not see what they measured — `emotion2vec` rates real acted
Nepali at 30.4% (sad 3.1%), and base Whisper at 0.917 WER. A metric that cannot
handle real speech cannot rank synthetic speech.
[ADR-008](docs/ADR.md#adr-008)

---

## Open questions

**The native-accent problem.** `scripts/filter_native.py` documents that the
AI4Bharat corpora are not Nepal-native — IndicVoices-R Nepali is 100% West Bengal
speakers, all 908 of them, and Rasa came from the same collection. **The filter was
never applied.** Both runs used `manifests/train_v2_aligned.jsonl`, which still
contains 296 h of those sources (~13% of the corpus), and half the 100-utterance
eval set comes from them. So "trained on Nepal-native speech" is **false** for this
checkpoint. Settling it needs a native-filtered retrain and a blind A/B judged by a
Nepal-native listener. [ADR-013](docs/ADR.md#adr-013)

**No emotion evaluation exists.** This is a voice-prompt model with no emotion
conditioning, and there is no valid Nepali emotion metric to build one against.
[ADR-014](docs/ADR.md#adr-014)

**No like-for-like comparison with Indic Parler-TTS.** The two were measured on
different axes with different instruments. Running the Parler Amrita config through
`score_wer_ft.py` and `score_sim.py` on the same 100 utterances would take about an
hour and would answer "which model ships for Nepali" on intelligibility. It would
still not settle emotion.
