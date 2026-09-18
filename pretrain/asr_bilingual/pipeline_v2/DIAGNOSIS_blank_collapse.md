# Five from-scratch runs, one failure: all-blank CTC collapse

Opened 2026-09-15. Live document — the conclusion section is the only part that
changes; the measurements above it are fixed.

## The failure, measured

`run_v5` trained 197,299 steps (100,000 h of audio seen, ~7.5 h wallclock) and
learned nothing at all. Not "converged badly" — nothing:

| | |
|---|---|
| train_loss, steps 0–10k | mean 342.7 |
| train_loss, steps 190k–200k | mean 378.3 |
| `val_wer` | 1.0000 at every one of 39 validations |
| blank frames at step 5k | **99.13%** |
| loss/token at step 5k | **8.144** — `ln 4001` = 8.294 is the uniform prior |

`pilot_ctc_32m`, `pilot_v3`, `run_v4` and all eleven probe arms end the same way.
The `bigbatch` arm (batch 2,400 s, lr 5e-4) reached 97.56% blank / 7.003.

**The logged `train_loss` is not a usable instrument here.** Lhotse buckets vary
the token count per batch by an order of magnitude, so the scalar swings 40–1800
on a model doing nothing at all. Everything below is loss *per target token* on a
fixed set (`eval_probe.py`), which is comparable across arms and checkpoints.

## Ruled out, each by measurement

| suspicion | how it was killed |
|---|---|
| audio ↔ text pairing | 220 sampled rows re-checked against the source corpus manifests: every mismatch is normalisation (verbalised numbers, stripped punctuation, lowercasing), none is a different utterance |
| tokenizer | 4,000 pieces, max id 3,997 over 1,261 sampled rows, blank = 4,000, UNK rate 0.000, 1.86 tok/word |
| dataloader | `diag_batches.py` on the real `input_cfg` stream: 1,889 utts, **0 empty targets**, **0 rows with T<U**, targets decode back to the correct text |
| 24 kHz source audio | NeMo's lhotse path applies `resample(config.sample_rate)` (`dataloader.py:520`); cuts arrive at 24 kHz and are resampled |
| bf16 | fp32 arm is identical: 6.91 → 0.45 vs 6.90 → 0.71 loss/token over 2,000 steps |
| weights not updating | 654 of 656 tensors differ between random init and step 5k, by 3–6x their own mean magnitude |
| an incoherent corpus | gradients from 6 independent batches agree at **cosine 0.93** (noise floor 1.9e-4). Every batch pulls the same way — toward blank |
| T/U too tight for CTC | at 12.5 frames/s: median 3.01, p01 1.58, **0.00% of rows below 1.0**. English is tighter than Nepali (median 2.22 vs 3.45) but still fine |
| the memorise control being fake | rotating the audio against the text takes the memorised model from 1.53 to 10.80 loss/token, and pooled encoder outputs for different clips sit at cosine 0.73, not 1.0. It is reading the speech |

## The one thing that works

64 utterances (human Nepali, 4–9 s) go **6.91 → 0.45 loss/token in 2,000 steps**.
Same script, same model, same tokenizer, same optimiser.

2,000 utterances, identical settings, 3,000 steps: **6.95, 94.9% blank**.
50,000: 7.13. The full 4,201 h mix: 7.12.

So the model can fit 64 utterances and cannot fit 2,000. That is the whole
problem, stated as small as it will go.

## The collapsed point is flat in every direction

`diag_linesearch.py` on `run_v5_step5k.nemo`: average the gradient over 4 batches,
then evaluate loss/token on 3 *different* batches at `w - eta*g`.

| eta | 1e-6 | 1e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 1e-1 |
|---|---|---|---|---|---|---|---|---|---|
| eval loss/token | 6.861 | 6.869 | 6.862 | 6.852 | 6.862 | 6.859 | 6.894 | 7.243 | 9.167 |

Baseline 6.8635, gradient norm 31.66. **Nothing descends anywhere.** Five orders
of magnitude of step size move the loss by less than 0.02 nats/token, and past
1e-2 it climbs. All-blank is a genuine flat attractor here, not an optimiser
failing to take a step that exists.

The direct consequence: **no learning rate, clipping or optimiser change can
recover a run that has already collapsed.** Corroborated by the sweep on a 6 h
testbed, where every arm is flat over 4,000 steps:

| arm | lr | steps 0-500 | 3000-3500 |
|---|---|---|---|
| fit2k_lr1e-3 (= scale_sub2k) | 1e-3 | 308 | 224 |
| fit2k_lr3e-4 | 3e-4 | 370 | 328 |
| fit2k_lr1e-4 | 1e-4 | 476 | 340 |
| fit2k_lr3e-5 | 3e-5 | 502 | 358 |
| fit2k_sub4x (4x subsampling) | 1e-3 | 390 | 407 |

So the only question worth asking is what stops it falling in.

## The loop is sound; random init is the wall

`nvidia/parakeet-tdt_ctc-110m` was already on the box, and this recipe's topology
came from its config. Load **only its encoder** (692 tensors, 0 missing, 0
unexpected), leave the 4,000-class CTC head random, and train on the same 6 h
every from-scratch arm has failed to move:

| arm | init | train_loss 0-500 -> 3500-4000 | held-out loss/token | held-out WER | blank% |
|---|---|---|---|---|---|
| `initenc` | parakeet encoder | 339.8 -> **3.8** | **3.589** | **0.632** | 73.6 |
| every from-scratch arm | random | 300-500, flat | 6.9-8.0 | 0.99-1.00 | 95-99 |

So the training loop, the data path, the loss, the tokenizer, the dataloader and
the evaluation are all sound. **The wall is specifically a randomly initialised
encoder.**

## Held-out scoring changes what the old probes mean

The previous session judged eleven arms on `train_loss` alone and read them all
as null. Re-scored on 200 held-out human-Nepali clips (`hum_heldout.jsonl`,
disjoint from every training subset), they are not equivalent:

| arm | update size | steps | held-out loss/token |
|---|---|---|---|
| `bigbatch` | 2,400 s | 4,000 | **6.869** |
| `clip0_spec` | 600 s | 3,000 | 7.084 |
| `nepseudo` | 600 s | 3,000 | 7.448 |
| `ladder_hum2k` | 240 s | 4,000 | 7.591 |
| `mixcfg` | 600 s | 3,000 | 7.799 |
| `run_v5` @ 5k | 600 s | 5,000 | 7.971 |
| `run_v5` final | 600 s | **197,299** | 7.988 |

Two things fall out. **run_v5 gained nothing between step 5,000 and step
197,299** — 190,000 steps and 37x more audio moved held-out loss by 0.017. And
the single best from-scratch arm is the one with the largest update, at 4,000
steps.

The scale ladder points the same way. Same population (human Nepali, 4-9 s), same
4,000 steps, only the number of distinct utterances moves:

| distinct utts | train_loss end | held-out loss/token |
|---|---|---|
| 64 | 10.6 | 19.87 (worse than the 8.29 prior) |
| 512 | 72.4 | 10.74 |
| 2,000 | 147.2 | 7.591 |

Progress is ~1/N in the number of distinct utterances and the small arms are
memorising, not generalising — held-out loss *above* the uniform prior is a model
that has learned 64 specific answers and nothing about speech.

## The mechanism: batches agree at init and disagree once collapsed

`grad_agree.py`, mean pairwise cosine between the full gradients of independent
240 s batches:

| model state | mean pairwise cosine |
|---|---|
| random init | **0.9263** |
| `run_v5` @ 5k (collapsed) | **0.0018** |

At random init every batch wants the same thing -- raise the blank probability --
so the model marches straight into the attractor at any learning rate. Once
there, the remaining signal is per-batch cosine ~0.05 against a noise floor of
1.9e-4, with individual pairs ranging -0.25 to +0.13. **The averaged gradient of
a 240-600 s update is mostly noise**, which is why the loss sits still while the
weights move 3-6x their own magnitude.

That gives the required update size directly. With per-batch correlation rho at a
batch size B, the gradient noise scale is about `B * (1 - rho) / rho`:

    240 s * (1 - 0.05) / 0.05 = ~4,600 s of audio an update

below which noise dominates signal. run_v5 ran at 600 s -- an eighth of it.
NVIDIA trains this topology at 600 s x 16-64 GPUs = 10,000-38,000 s, comfortably
above it. **This is a batch-statistics problem, and gradient accumulation fixes it
on one GPU**: accumulation buys exactly the same update statistics at exactly the
same throughput in audio/second. It costs wall-clock *per update* and nothing in
samples/second, and a larger update needs proportionally fewer of them. The
8-GPU ask is a ~7x wall-clock speedup, not a precondition.

## Open arms

- `probe_struct.sh` — 4x subsampling (halves the alignment pressure), then lr
  3e-4 / 1e-4 / 3e-5 on a 6 h testbed that any working recipe must fit.
- `probe_ladder.sh` — 64 → 512 → 2,000 distinct utterances from one homogeneous
  population at a fixed step budget, to separate "count" from "heterogeneity".

## Conclusion so far

From-scratch is not blocked by a bug, a hyperparameter or the corpus. It is
blocked by the update size this hardware can reach: 1 GPU at 600 s an update
against the 10,000-38,000 s that NVIDIA trains this topology with. `bigbatch3`
(7,200 s an update, otherwise identical to `bigbatch`) is the last test of that,
and `arch110m` says whether 27M was also too small.

Do not start a long run until an arm shows held-out loss/token off 8.29 and
blank% off 95.
