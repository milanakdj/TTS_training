# v3 result: the run failed. v2 remains the shipping model.

**Status: dead run, 2026-09-20.** Both stages trained to completion (200,000
steps each) and both produce unintelligible Nepali. Nothing here is shippable and
nothing in it is recoverable by tuning inference.

## The canonical eval: 100 held-out utterances

Same set, same harness and same instrument as the shipped v2 table. All 100
targets were verified still present in `valid_v3` and absent from `train_v3`,
so the rows are directly comparable. **The v2 rows reproduce the published table
exactly** (0.322/0.139/0.841 for the student), which is the check that nothing
in the harness drifted.

| system | WER | CER | speaker sim | CPU speed |
|---|---|---|---|---|
| real human (ceiling) | 0.343 | 0.122 | 0.807 | — |
| **v2 student 6L (shipped)** | **0.322** | **0.139** | **0.841** | **5.88x RT** |
| v2 teacher 24L | 0.402 | 0.177 | 0.826 | 2.10x RT |
| v3 teacher 24L (unguided, eos 1.0, t 0.3) | 0.812 | 0.632 | 0.701 | 2.10x RT |
| v3 student 6L (eos 0.0, t 0.3) | 0.724 | 0.578 | 0.635 | 5.93x RT |

Worse on every source individually. **Speaker similarity collapsed as well** —
0.635 against v2's 0.841, and below the real-human cross-utterance ceiling of
0.807 — so v3 fails at voice cloning too, not just at pronunciation. Speed is
the one thing that did not change (5.93x vs 5.88x, measured in the same run):
v3 costs everything and buys nothing.

The four RTF numbers were all measured in one run under the same load, so they
are comparable with each other but not with the published v2 figures (1.53x /
5.04x), which were taken on a differently loaded box.

## What was measured to choose the settings

40 held-out utterances (`infer/final/calib_v3.json`), drawn from the same
`valid_v3` split as the headline eval set and **disjoint from `pairs.json`**, so
the threshold search below could not fit the test set. Scored with
`himalaya-ai/whisper-large-v3-nepali-final`, the same instrument as every other
number in this project. `real human` is the genuine recording of the same
utterance through the identical pipeline — the ceiling and the check on the
metric.

| system | settings | WER | CER |
|---|---|---|---|
| real human (ceiling) | — | 0.418 | 0.149 |
| **v2 student 6L (shipped)** | defaults | **0.330** | **0.107** |
| v3 student 6L | temp 0.3, eos 0.0 (its best) | 0.685 | 0.506 |
| v3 student 6L | temp 0.7, eos 0.0 | 0.733 | 0.548 |
| v3 teacher 24L, unguided | eos 1.0 (its best) | 0.850 | 0.659 |
| v3 teacher 24L, guided | cfg 3.0, temp 0.3 (its best) | 0.630 | 0.407 |

v2 scores 0.107 on these same clips, so the set is not hard. v3 is five times
worse at its own best settings, and its *teacher* — sampled exactly the way it
was trained to be sampled — is worse than the v2 student by a factor of four.

## The training objective said so all along

| teacher, valid @ 200k | flow_loss | eos_loss | flow_diag |
|---|---|---|---|
| v2 | **-0.0746** | 0.0078 | 14.43 |
| v3 | **+0.0831** | 0.0276 | 16.93 |

v3 at step 200,000 is worse than v2 was at step **5,000** (0.0453). The v3 curve
is flat from ~100k, which reads as convergence only if you do not have the v2
curve next to it: it converged to a much worse point. **A flat loss curve is not
evidence of a healthy run. Always plot it against the previous generation's.**

## Two symptoms that were nearly misdiagnosed

**Early EOS is a symptom, not the disease.** At pocket-tts's shipped
`eos_threshold=-4.0` the v3 student emits 0.26x the reference duration — 0.32 s
where v2 gives 4.72 s on the same prompt and text. That looks like a
configuration bug, and `eos_threshold` does move it: +1.0 restores the duration.
But restoring the duration does **not** restore the content (CER 0.548 -> 0.557
-> 0.558 -> 0.576 across thresholds 0/1/2/4 — flat). The 3.5x higher eos_loss in
the table above is the real story; the head is badly fit, and so is everything
else.

**The 140k probe was read too charitably.** `score_oov_v3.log` was recorded as
"OOV deletion fixed, Devanagari digits regressed". Re-reading it, half the
transcripts are nonsense (`सन् २०२४ मा ४५ जना विद्यार्थी थिए।` ->
`सन् त्युरिकल पत्कल्कल्पन मापन उसमार्ने काम गरे`) and tail retention was 7-9/22
against v2's 10/22 raw. The model was already broken at 140k. It was scored
against the OOV-deletion hypothesis and the general collapse was not noticed,
because there was no v2 row and no human row in that table.

## Probable cause

The v3 models speak **English better than Nepali**. The student renders "The
quick brown fox jumps over the lazy dog." verbatim while its Nepali is
unintelligible. That is the shape of the answer:

- v2 reset the whole text embedding and learned **4,000** Nepali pieces from
  scratch, with 100% of the training signal on Nepali. It reached CER 0.107.
- v3 inherited rows 0-3999 (English, pretrained, already good) and had to learn
  **5,682 fresh Nepali rows** — 1.4x more pieces, each seen proportionally less
  often — while ~27% of the corpus (800 h of ~3,010 h) was English replay.

So the Nepali half got materially less effective training and underfit. The
graft did exactly what [V3_PLAN](V3_PLAN.md) promised — English is representable
and works — but it was paid for out of the Nepali budget, and nobody costed that.

This is a hypothesis consistent with every measurement here; it is not proven.
Cheapest test before committing another ~72 h: re-run stage 1 only, same graft
and tokenizer, with the English replay cut to ~5% and `max_steps` raised, and
compare the valid `flow_loss` curve against v2's at the same step. If it does not
beat v2's -0.0746 trajectory by ~50k, the tokenizer graft itself is the suspect
and should be tested in isolation on a Nepali-only run.

## What was NOT the cause

Ruled out by measurement, so nobody repeats them:

- **Not the EMA export.** The training checkpoint and the safetensors export
  generate identically badly, with EMA on and off.
- **Not the inference harness.** The v2 student renders 4.72 s of correct Nepali
  through the exact same `TTSModel` path, prompt and text.
- **Not guidance dropout.** `nepali_finetune.yaml` and `nepali_finetune_v3.yaml`
  differ only in tokenizer, `n_bins`, the embedding graft, and the data paths.
  `text_dropout` and `voice_dropout` are identical.
- **Not temperature.** 0.3 beats the inherited 0.7 (CER 0.506 vs 0.548) but the
  model is unusable at both.
- **Not the distillation step alone.** The teacher is broken too, at its own
  intended sampling settings.

## Disposition

- **v2 remains the shipping model** (`milanakdj/pocket-tts-nepali-6l`).
- Both v3 checkpoints are archived private on the Hub with `-v3-failed` in the
  repo name, weights plus the stage-1 training checkpoint, so the diagnosis is
  reproducible and a future re-run has a baseline to beat.
- `/workspace/nepali_student_6l_v3` holds 81 checkpoints (55 GB) because
  `num_ckpt_keep: 999`. Once the archive upload is verified, all but the final
  one can go.
