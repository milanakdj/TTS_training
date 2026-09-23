# Nepali Pocket-TTS

A Nepali text-to-speech model with zero-shot voice cloning that runs **5.04x faster
than real time on CPU**. Two stages: a 24-layer teacher finetuned from Kyutai's
English release, then a 6-layer student depth-distilled from it. **The student is
the shippable model and it beats the teacher on every measured axis.**

**Status (2026-09-23).** Four generations exist; read them in this order:

| gen | what | verdict |
|---|---|---|
| v2 | Nepali-only, BPE-4000 tokenizer | **shipped**, still the production model |
| v3 | grafted ne+en tokenizer, 27% en replay | **failed**, archived (see `docs/V3_RESULT.md`) |
| v4 | same graft, 5% en replay | Nepali = v2, English works but is ~40x worse than base Kyutai; published as a candidate, **not promoted** |
| v5 | English replay replaced by audio from untouched Kyutai | **in progress**: synthetic data pipeline built, 7.5k-step probe next |

Nothing is training right now. The v2 description below is still accurate for v2;
v4 and v5 have their own sections.

| Document | Read it for |
|---|---|
| this file | commands, layout, traps |
| [`docs/V3_RESULT.md`](docs/V3_RESULT.md) | why the v3 retrain failed, and what was ruled out |
| [`docs/V4_RECIPE.md`](docs/V4_RECIPE.md) | the controlled arms behind v4's 5% replay recipe |
| [`docs/ADR.md`](docs/ADR.md) | every decision and why it was taken |
| [`docs/INTERN_GUIDE.md`](docs/INTERN_GUIDE.md) | how the model works, from the basics |
| [`../dataset/`](../dataset/) | the data: hours per corpus, GB, provenance, what may be redistributed |

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
| student 6L | 109.5M (75.5M backbone + 9.8M flow head + 4.2M emb + 20.1M Mimi) | 438 MB | `hf://milanakdj/pocket-tts-nepali-6l` (private) or `himalaya-ai/pocket-tts-nepali-6l` (public) |
| teacher 24L | 336.1M (302.1M backbone, rest identical) | 1.34 GB | `hf://milanakdj/pocket-tts-nepali-24l-teacher` (local run dir deleted) |

Every removed parameter came out of the backbone; the flow head and Mimi are copied
and frozen. Mimi is only ~19% of the teacher's per-frame cost, so there is no Amdahl
ceiling hiding in the speedup.

### v4 (bilingual candidate) and the baseline it should have been judged against

The bilingual 2x2 set (`infer/final/pairs_2x2.json`: {ne, en} text x {ne, en}
voice, 50 utterances per cell). Nepali text is scored with the finetuned Nepali
Whisper, English text with base large-v3-turbo; compare each column only to its
own `real human` row. Sim is resemblyzer cosine against an unrelated-speaker
prompt (a floor, not a ceiling).

| system | en text CER (en / ne voice) | en sim (en / ne voice) | ne text CER (en / ne voice) |
|---|---|---|---|
| real human | 0.011 / 0.011 | 0.571 / 0.583 | 0.120 / 0.120 |
| **Kyutai 6L, untouched** | **0.009 / 0.010** | **0.848 / 0.818** | 0.898 / 0.878 (cannot speak Nepali) |
| Kyutai 24L, untouched | 0.009 / 0.015 | 0.810 / 0.786 | 0.889 / 0.898 |
| v4 student 6L | 0.356 / 0.370 | 0.709 / 0.709 | 0.111 / 0.134 |
| v2 student 6L | 1.030 / 0.879 | — | 0.157 / 0.117 |

v4 fixed v2's missing English (CER ~1.0 -> 0.36) and kept Nepali at v2 level, but
the base model it was finetuned from already speaks English at CER 0.009 and
clones a Nepali voice into English at sim 0.82. The finetune **forgot** that.
Nobody measured the base model until after v4 shipped; see the v5 section.

| model | params | weights |
|---|---|---|
| v4 student 6L | 109.5M | `hf://milanakdj/pocket-tts-nepali-en-6l-v4` (private) |
| v4 teacher 24L | 336M | `hf://milanakdj/pocket-tts-nepali-en-24l-teacher-v4` (private, includes `training/checkpoint_00250000.pt`) |

Local run dirs `/workspace/v4_student_6l` and `/workspace/v4_teacher_24l` keep only
the final checkpoint; intermediates and optimizer states were deleted 2026-09-23.

---

## Layout

```
pocket_TTS/
├── repo/                      Kyutai pocket-tts checkout + our training/configs/*.yaml
│   └── .venv/                 the interpreter for anything touching pocket_tts
├── manifests/                 train/valid jsonl; *_v2_aligned.jsonl is what training reads
├── tokenizer/                 nepali_bpe4000.model — MUST travel with the weights (v2)
├── tokenizer_v3/              ne_en_9682.model — grafted ne+en tokenizer (v3, v4, v5)
├── release_v3/, release_v4/   Hub push scripts, model cards, inference examples
├── scripts/                   manifest building, alignment, train launchers
├── infer/                     inference configs, benchmarks, generation
│   └── final/                 the evaluation harness and its results
├── frontend/                  ne_frontend.py -- MUST wrap every input string
├── docs/                      ADR.md, INTERN_GUIDE.md
└── train_{teacher,student}.log
```

Checkpoints were written to `/workspace`, never here: `ckpt_freq 2500` over 200k
steps is ~54 GB and `/` was 88% full — [ADR-011](docs/ADR.md#adr-011). **Both run
dirs were deleted on 2026-09-09** once the exports were verified on the Hub; the
inference configs point at `hf://`, so both models still load. Re-distilling a new
student needs the teacher's training checkpoint, which `torch.load` cannot fetch
over `hf://` — see the comment in `configs/nepali_distill.yaml`.

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

**Bilingual 2x2 eval** (v3 onward; any pocket-tts config, including untouched
Kyutai as the baseline):

```bash
cd repo && EOS_THRESHOLD=0.0 DEVICE=cuda .venv/bin/python3 ../infer/final/gen_2x2.py <name> <config.yaml>
cd infer/final
$SIM_VENV score_2x2.py --part ne    <name> ...   # Nepali cells + all speaker sim
$ASR_VENV score_2x2.py --part en    <name> ...   # English cells
$SIM_VENV score_2x2.py --part merge <name> ...   # table
```

Every scorer pass rewrites `eval_2x2_{ne,en}.json` with only the names you pass, so
list every system you want in the final table on every call.
`calib_eos_v4.py` + `score_calib_v4.py` sweep `eos_threshold` on `calib_v4.json`
(40 utterances, disjoint from the 2x2) and score content, not duration.

**Which interpreter.** There is no single venv with everything:

| for | interpreter |
|---|---|
| anything importing `pocket_tts` | `repo/.venv/bin/python3` |
| base whisper WER | `synthetic_pipeline/asr_qc_service/.venv/bin/python3` |
| Nepali-finetuned WER, speaker similarity | `synthetic_pipeline/vc_service/.venv/bin/python3` |

---

## Traps

**Measure the base model first.** v4's English was judged by `flow_loss [en]` on
en-in YouTube valid data, which only says how well the model fits *that* data.
Untouched Kyutai 6L on the 2x2 scores English CER 0.009 / sim 0.85; v4 scores
0.356 / 0.71. Any retrain gets the base model's 2x2 row in the same table.

**The replay ratio is not the English lever.** At 15% en replay (`configs/v4e_en15.yaml`)
English valid loss matched 5% at step 2.5k (0.2730 vs 0.2743) and was only a bit
better at 7.5k (0.1627 vs 0.1736), while Nepali paid a little (0.0026 vs -0.0018).
More of the wrong English does not bring back the right English.

**v4 inference needs three non-defaults**, all baked into the v4 model card's
`inference.py` and Colab cells:
- `eos_threshold=0.0` in `load_model` (not settable in YAML). At the default -4.0
  English stops early: calib English WER 0.925 vs 0.523. Nepali doesn't care.
- `normalize(text, keep_latin=True)`. The default `keep_latin=False` spells
  English out in Devanagari ("Hello" -> "हेल्लो"), which is right for v2's tokenizer
  and wrong for `ne_en_9682`.
- `assert_clean(text, model=<ne_en_9682.model>)`. With no `model=` it looks for
  v2's `nepali_bpe4000.model`, which checks the wrong tokenizer and does not exist
  off this box. `gen_2x2.py` passes raw text and never calls the frontend, so the
  eval numbers are unaffected.

**Generation is unseeded.** The same text and voice give a different take every
call; `torch.manual_seed(n)` before `generate_audio` makes it byte-identical on
one machine. On v4 this is loudest on English, whose voice match is both lower
(0.54 vs 0.83 on one prompt) and more spread out. Lowering `temp` to 0.05 did not
raise it.

**Mid-utterance code-switching is untrained** (0.05% of the corpus). On v4 the
same mixed sentence rendered its English fine after `?` and as gibberish after
`।`. Until v5, split at the language boundary and make one call per language.

**`pgrep -f <pattern>` inside a wait loop matches the loop's own command line**,
so the loop never exits; `pkill -f` with a pattern from the same shell kills that
shell. Wait on a PID with `kill -0`, or poll the output file.

**v3 ran and FAILED — see [`docs/V3_RESULT.md`](docs/V3_RESULT.md).** Both stages
finished 200k steps and neither produces intelligible Nepali (student CER 0.578
against v2's 0.139 on the same 100 utterances; speaker similarity 0.635 against
0.841). v2 is still the shipping model. The checkpoints are archived private at
`milanakdj/pocket-tts-nepali-{6l,24l-teacher}-v3-failed`. The design it was
testing is in [`docs/V3_PLAN.md`](docs/V3_PLAN.md); do not re-run it unchanged.

**A flat loss curve is not a healthy one.** v3's valid `flow_loss` was flat from
~100k and that read as convergence. It had converged to +0.0831, worse than v2
reached at step 5,000 (+0.0453), and v2 finished at -0.0746. The comparison that
would have killed the run 34 h in was never made. Plot a retrain against the
previous generation's curve at the same step, before stage 2.

Everything below describes the SHIPPED v2 model and still applies to it.

**The tokenizer cannot represent English, digits or a hyphen.** `nepali_bpe4000`
has no `byte_fallback`, so every Latin word, every ASCII digit, `४`-`९`, `-`, `(`
and `;` encode to one `<unk>` — and the model *deletes the word or truncates the
rest of the utterance* rather than mispronouncing it. `gen_samples.py`'s
`05_numeric` case was written pre-verbalized ("सन् दुई हजार पच्चीसमा"), which is
why 200k steps of training never surfaced this. Run every string through
`frontend/ne_frontend.py:normalize()` before `generate_audio()`; fixing it
properly needs a new tokenizer and a retrain.

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
(private, org-owned).

**Corrected 2026-09-08 — that model takes the STANDARD single-sot prefix.** Its
card warns a doubled `<|startoftranscript|>` is required; measured on 100 unseen
mahadhwani clips it is *better* without one — **0.245/0.090 with 1x sot against
0.282/0.106 with 2x**. Neither published repo was realigned, so the pathology
applied to intermediate checkpoints, not the release. It therefore works with
`faster-whisper` and `pipeline()` directly. Check both conventions on any new
checkpoint with `infer/asr_test/asr_ab.py FT_NSOT=1|2`: the wrong one does not
error, it silently returns worse text.

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

## v5 (in progress): replace the English replay with Kyutai's own English

Kyutai cannot speak Nepali, so it voices **only English**, cloned from a real clip
of our corpus; every Nepali word stays real human audio. Training cuts each clip at
a word inside its first 5 s and uses that head as the voice prompt
(`repo/training/dataloader/loader.py`), so a clip that starts with a real Nepali
speaker teaches "this Nepali voice, now in English". Three clip kinds:

| kind | clip | teaches |
|---|---|---|
| plain | Kyutai English alone | English in Kyutai's own quality |
| concat | [real Nepali 3-5 s][0.25 s][Kyutai English, same voice] | Nepali voice -> English |
| insert | [real Nepali ..word k][Kyutai phrase][real Nepali word k..] | mid-sentence code-switching |

English text was written by Claude Haiku subagents (no API key on this box), under
`/workspace/v5_synth/text/parts/`: `a*` (18k sentences, ~90% under 10 words),
`b*` (medium/long, 10-30 words, length-checked), `p*` (2.9k code-mix phrases).
Haiku self-reports claimed a length mix that the files did not have; always
measure.

Pipeline (each step resumable, run from `pocket_TTS/`):

```bash
python3 scripts/plan_v5.py                                   # -> /workspace/v5_synth/jobs.jsonl
cd repo && SHARD=i NSHARDS=3 .venv/bin/python3 ../scripts/gen_synth_v5.py   # Kyutai 6L, ~0.3 s/clip
RAW=/workspace/v5_synth/audio/raw $ASR_VENV scripts/qc_align_v5.py          # Whisper word times; ~85% pass
RAW=/workspace/v5_synth/audio/raw repo/.venv/bin/python3 scripts/assemble_v5.py  # -> train_v5_synth.jsonl
```

Job ids are positions in `jobs.jsonl`: re-running `plan_v5.py` re-maps them to new
text, so any audio made against an older plan must go to a fresh `OUT`/`RAW` dir
or it enters the set under the wrong transcript. Pilot output (142 jobs) is in
`/workspace/v5_synth/pilot/`.

Still to do: full generation, latents (`repo/training/scripts/precompute_latents.py`),
a mix manifest, a 7.5k-step probe judged on the 2x2 against **untouched Kyutai 6L**
(English) and v2's curve (Nepali), then teacher + student only if it passes.
Check the licence terms of the gated `kyutai/pocket-tts` voice-cloning weights
before shipping anything trained on their output.

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
