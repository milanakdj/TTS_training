# Why this repo exists, and why everything is in it

This is not just a model drop. It is the **durable home** for a training run whose
local artifacts were deliberately destroyed. Read this before assuming something is
recoverable elsewhere — for most of it, this repo is the only copy.

## Why the run was stopped

A ne+en finetune of `nvidia/nemotron-3.5-asr-streaming-0.6b` ran 2026-09-12 to
2026-09-13 and reached global step 147,741 of a planned 418,187 (35%). It was
stopped early, deliberately, for two reasons:

1. **Nepali had flattened.** Gold-test normalised WER went 0.1800 -> 0.1864 ->
   0.1875 across the last three checkpoints — drifting *worse*, not better. The
   remaining ~270k steps were not going to buy anything.
2. **English was unrecoverable.** It had collapsed to a degenerate repetition loop
   and a mid-run fix failed to reverse it. Continuing would have burned ~88 h of
   wall clock to confirm a known negative.

The next run will be a **pretrain on an enlarged corpus** (pending a Seed-VC pass to
generate out-of-vocabulary coverage), not a resume of this one. That is why nothing
here is optimised for resumption.

## Why we are pushing this instead of keeping it on disk

- **The scratch space was shared.** The run lived on `/workspace`, a network
  filesystem used by other teams. It did not belong there and has been removed.
- **The local disk cannot hold it.** The working repo lives on a root filesystem at
  93% capacity with ~128 GB free. A 70 GB training directory is not a reasonable
  tenant, especially with a Seed-VC run and a full pretrain queued behind it.
- **Most of it was redundant anyway.** Of 70 GB: 35 GB was English MLS audio
  (re-downloadable, and English is being dropped), 31 GB was four checkpoints of
  which only the best mattered, and ~850 MB was superseded manifests kept only as
  accidents of history.

So: the parts worth keeping were pushed here, and the local copies deleted.

## What is here

| path | what it is |
|---|---|
| `nemotron_ne_nepali_best.nemo` | The model. Best checkpoint (`val_wer=0.2065`), exported from its `.ckpt`. **The source checkpoints are deleted — this is the only copy.** |
| `README.md` | Model card, results, and verified inference instructions. |
| `POSTMORTEM_english_collapse.md` | Why the English half failed, and the rules for the next run. |
| `eval_history.csv` | Per-checkpoint WER/CER for *both* languages across the run. |
| `manifests/` | Training/eval manifests: audio paths + transcripts. No audio. |
| `logs/` | Training logs (gzipped) and every evaluation JSON produced. |
| `code/` | `train.py`, `eval_ckpt.py`, `export_nemo.py`, `watch_eval.py` and helpers. |

## What is deliberately NOT here

- **The 31 GB of `.ckpt` files.** Deleted. This means **the run cannot be resumed** —
  optimizer state is gone. Intentional: the next run is a fresh pretrain.
- **Audio.** The manifests reference `/workspace/proc_data_new/...`, a shared corpus
  that is predominantly YouTube-derived and **not redistributable**. Paths will only
  resolve on the original machine.
- **The 35 GB English MLS set.** Re-fetch with `code/get_mls_eng.py` +
  `code/extract_mls.py` if English is ever revived.

## If you are starting the next run

1. Read `POSTMORTEM_english_collapse.md` first. It documents two real defects and
   which one actually mattered — do not re-derive them.
2. `manifests/ne_corpus_train.jsonl`, `ne_train.jsonl` and `en_train.jsonl` are the
   source components; `train_mix_langid.jsonl` is the assembled mix with correct
   `lang`/`prompt_mode` fields. The two earlier, broken mixes were deleted.
3. `code/watch_eval.py` scores every new checkpoint on *both* languages. A
   single-language `monitor` hid a total collapse for 21 hours on this run. Use it.
