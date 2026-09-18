# Why this repo exists

The 444 MB `.nemo` is the artifact. Everything else here is here because the
*negative* results cost more to obtain than the model did, and they are the part
that will not survive in anyone's memory.

## What this run was for, decided before it started

`code/run_v7_encinit.sh` states the goal in its header, because
`PRETRAINING_FROM_SCRATCH.md` Step 0 requires writing it down before spending the
GPU: pretraining is 3–6 weeks versus ~2 days for a finetune and lands at or below
it on accuracy, so "better Nepali WER" is explicitly **not** a valid reason to
pretrain. The valid reasons are on-device size, streaming latency, license
independence, or tokenizer.

**This run pursued on-device size (110M vs the finetune's 0.6B) and the
tokenizer (2.39 vs 5.12 tok/word). It did not pursue license independence —
the encoder is NVIDIA's, so that goal was forfeit the moment encoder-init was
chosen.**

## The two things worth keeping

1. **From-scratch CTC on this corpus is dead on 1 GPU.** Five arms, up to 97,500 h
   seen, all landed on or beside the all-blank attractor — `run_v6` finished at
   loss/token 8.291 against a uniform prior of `ln(4001)` = 8.294, i.e. 0.003 nats
   from knowing nothing. A line search from 1e-6 to 3e-3 found no descent
   direction. `docs/DIAGNOSIS_blank_collapse.md` is the full ruled-out list.
2. **The plateau is real and label-bound.** Human-Nepali WER sat at 0.177 → 0.178
   → 0.176 across steps ~35k → 40.8k → 50k, through a full cosine anneal to 1e-6.
   The remaining budget bought accuracy on English and on canary-agreement, which
   the logged mixed `val_wer` (0.0756 → 0.0742) faithfully reports and which is
   exactly the part that was never the problem. **Scale the recipe further on
   this corpus and expect nothing.**

## What is not recoverable elsewhere

- The 31 GB of intermediate `.ckpt` files have been deleted. Only the final
  `.nemo` and the four periodic checkpoints still on disk under
  `/workspace/asr_pretrain_v2/` remain.
- Training audio is **not** in this repo and cannot be: 86.6% of the Nepali
  portion is YouTube-derived. Manifests carry paths only, and those paths point
  at a filesystem this repo does not ship.
- The training manifests themselves (`train_mix.jsonl`, 709 MB, and the four
  per-bucket files, ~1.3 GB total) are **not** included. `mix_report.json` and
  `train_input_cfg.yaml` capture the weights, hours and label provenance that
  actually matter for interpreting the result.
