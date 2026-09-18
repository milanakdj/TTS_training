# Nepali + English Streaming ASR — what we are building

Last updated: 2026-09-12

## The goal

A **bilingual (Nepali + English) streaming speech-to-text model**, good enough on
Nepali to be usable in production, and small/fast enough to serve.

Two separate routes to that goal. They are not alternatives of equal weight — one
is running, the other is a decision we have already analysed and mostly declined.

1. **Finetune** `nvidia/nemotron-3.5-asr-streaming-0.6b` on our Nepali corpus with
   English replay. **This is what is running now.**
2. **Pretrain from scratch** (the reason this folder is named `pretrain`).
   Analysed, costed, and currently **not recommended for accuracy** — see below.

---

## Route 1 — the finetune (ACTIVE)

Working dir: `/workspace/asr_bilingual/` (not in this repo — it is on the scratch FS)

| | |
|---|---|
| base model | `nemotron-3.5-asr-streaming-0.6b` (RNN-T, prompted, streaming) |
| data | `manifests/train_mix.jsonl` — 773,780 utts / **2,207.1 h** |
| schedule | 6 epochs = 418,187 steps, AdamW, cosine, peak lr 1e-4, warmup 2k |
| hardware | 1x H100 80GB (Slurm job 22093, GPU 3 of 8 on bodhanai-node011) |
| log | `logs/train_6ep.log`, tmux session `train` |
| checkpoints | `ckpt/nemotron_ne_en/` — top-3 on `val_wer`, every 10k steps |

### Why this base model

It already has a **streaming** encoder and a **prompted** language-selection
mechanism (`prompt_field: target_lang` against a 128-entry `prompt_dictionary`).
Nepali occupies slot `ne-NP` = 46. That slot **exists but was never trained** —
baseline Nepali WER was **1.440 / CER 0.878**. So this is teaching the model a
language from scratch, not adapting one it already knows.

### Result so far

| step | Nepali `val_wer` |
|---|---|
| baseline (untrained slot) | 1.440 |
| 20,000 | **0.2687** (best) |
| 30,000 | 0.2763 |
| 40,000 | 0.2795 |

The slot went from unusable to functional. Drift after 20k is expected with the
cosine LR still near peak; the real test is the annealing phase in the last third.

### Data composition — READ THIS BEFORE TRUSTING ANY NUMBER

| slice | hours | labels |
|---|---|---|
| `ne` (ans_snr*, mahadhwani — YouTube) | 1,647.4 | **pseudo-labels (canary/saaras)** |
| `ne-NP` (indicvoices-r, rasa) | 258.9 | human |
| `en-US` (MLS replay) | 300.8 | human |

**Only 13.6% of our Nepali is human-labeled.** 86% carries canary's transcripts,
and canary itself scores **0.155 WER on our 600-clip human gold set**. Training
predominantly on those labels converges the student toward the teacher — you
cannot beat a teacher you are imitating.

Corroborating evidence already in hand: the `flex` gold finetune moved WER
0.155 -> 0.157 with 289 h of gold. The ceiling is sticky.

---

## Route 2 — pretraining from scratch (ANALYSED, NOT RECOMMENDED FOR ACCURACY)

### Hours we would actually have

Nepali is nearly capped. Open-source additions move the total ~9% but add ~64%
more *human* supervision, which is the part that matters:

| source | hours | labels |
|---|---|---|
| current `ne` | 1,647.4 | pseudo |
| current `ne-NP` | 258.9 | human |
| + OpenSLR SLR54 | ~157 | human, read |
| + FLEURS ne_np | ~7 | human |
| + Common Voice ne | ~2 | human |
| **Nepali total** | **~2,070** | **~425 h human** |

English is effectively unbounded (MLS-en 44.5k, People's Speech 30k, GigaSpeech
10k, ...) — you pick a cap, not a maximum. IndicVoices Nepali is already in the
mix; YODAS-ne would add a few hundred h with the same YouTube provenance.

### Cost on this hardware

From-scratch conformers need **50k h seen minimum, 150–300k to converge**.
Anchored on our own measurement (39x realtime for the 0.6b RNN-T — see the
throughput caveat below):

| audio seen | 120M CTC (~150x, est.) | 32M CTC (~300x, est.) |
|---|---|---|
| 50k h (bare floor) | 14 d | 7 d |
| 150k h (converged) | **42 d** | 21 d |
| 300k h (well-trained) | 83 d | 42 d |

Realistic range for something worth evaluating: **3–6 weeks**, versus ~1.9 days
for the finetune. The 150x/300x figures are **extrapolated, not measured**.

### Why we are not doing it for accuracy

A from-scratch model trained on this corpus is **distillation from canary**. It
would cost 2–6x the finetune's GPU time and land at or below it. We would be
spending weeks of H100 to reproduce a model we can already download.

Shrinking the model does not rescue this. The schedule is denominated in **audio
hours seen**, not FLOPs, and the dataloader feeds the same bytes/sec whether the
model is 600M or 32M — so below a point, shrinking buys accuracy loss for zero
wallclock gain.

### When pretraining IS the right call

Pretrain if we want something the finetune structurally cannot give:

- **on-device size** (0.6b is too big for phones)
- **streaming latency** below what the base encoder allows
- **license independence** from NVIDIA's model
- **fixing the tokenizer** — the shipped BPE is 5.12 tok/word on Nepali (near
  character level) vs 2.39 for a proper Indic BPE, inflating RNN-T targets ~2.1x.
  `ne_en_bpe.model` (2.39 tok/word) is already built at
  `/workspace/asr_bilingual/ne_en_bpe.model` if we go this way.

Those are real goals worth weeks. **Nepali accuracy is not one of them** — for
that, the binding lever is human-labeled hours, not GPU-days or parameters.

---

## Hardware reality

- **1x H100 is fixed.** The node has 8, but Slurm granted this job exactly one
  (job 22093 is a 365-day 1-GPU dev box). No `srun`/`sbatch`/munge inside the
  container, so multi-GPU cannot be arranged from this shell.
- The ~7x multi-GPU speedup is real but is an **organizational ask**: someone must
  submit `--gres=gpu:8` from `bodhanai-node001` as user `ttsteam`.
- **The GPU thermally throttles.** Shared chassis; we have seen SM clocks swing
  1065–1980 MHz and throughput swing **1.26–3.67 it/s** with no change on our
  side. Quote a cumulative average, never an instantaneous rate.

---

## Hard-won gotchas (each of these cost real time)

1. **The `--bench` path lies by 10x.** It reported 39x realtime; true steady state
   is 384x. It times ~40 steps from cold start, which are dominated by
   warprnnt_numba JIT-compiling a kernel per bucket shape (`num_buckets: 30`) plus
   CUDA graph capture. Never size a run from a short bench — measure past step 1000.
2. **`ne` is not a valid prompt key.** Only `ne-NP` exists. A manifest tagged `ne`
   raises `ValueError: Unknown prompt key` — but only at **step ~1955**, when the
   bucket buffer first serves one. Validate distinct `target_lang` values against
   `cfg.prompt_dictionary` before launching.
3. **`exp_manager` fights the Lightning Trainer.** It creates the logger and the
   checkpoint callback itself, and raises if the Trainer already made either. Pass
   `logger=False, enable_checkpointing=False`.
4. **Lhotse datasets have no `__len__`.** The cosine scheduler tries
   `len(train_dataloader.dataset)`, so `max_steps` must be set explicitly and the
   run driven by steps, not epochs. `val_check_interval` must be an int, and
   `every_n_epochs` must be 0 or no checkpoint is ever written.
5. **The baseline's prompt was half-random.** `prompt_mode` defaults to `unified`,
   which takes the language-agnostic `auto` slot (101) with p=0.5 per cut. The
   1.440 baseline is therefore not reproducible. Pin `prompt_mode=langID`.
6. **UNK targets leak into checkpoint selection.** The tokenizer has no token for
   `( ) : ; " ' —` , ZWJ, or Devanagari digits — they become `<unk>`, a *trainable*
   label. 1.36% of training rows and **6.5% of `ne_val.jsonl`** are affected, which
   puts a constant floor under the `val_wer` that `save_top_k` ranks on.
7. **`U >= T` is NOT a crash.** RNN-T permits multiple emissions per timestep;
   verified against NeMo's reference loss at `T=3, U=100`. 21% of our rows have
   tokens >= frames and that is fine — a consequence of the 5.12 tok/word tokenizer.
8. **Source audio is 24 kHz, the model wants 16 kHz.** NeMo resamples on CPU every
   batch. Correct, but it is the obvious lever if the dataloader ever becomes the
   bottleneck (it is not currently — workers sit at ~1.4% CPU).

## Evaluation

- Harness: `/workspace/asr_bilingual/eval_ckpt.py` (pins the prompt, reports raw +
  normalized WER/CER, compares against the 1.440/0.878 baseline).
- Test sets: `manifests/ne_test.jsonl` (600, human) and `en_test.jsonl` (600).
- **Never evaluate on a held-out split of the training corpus.** A model scoring
  ~3-6% there is measuring ASR-to-ASR agreement with canary, not accuracy — the
  same artifact that made `flex` look like 0.013.
- Gate real claims on the himalaya-ai whisper instrument with the doubled-SOT
  prefix against gold, plus human spot-checks. `emotion2vec`-style automatic
  scorers are not valid instruments for Nepali.
  






