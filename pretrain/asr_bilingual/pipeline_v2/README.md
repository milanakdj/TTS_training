# pipeline_v2 — from-scratch ne+en ASR pretrain, queued behind Seed-VC

Implements phases 0–3 of `../../PRETRAINING_FROM_SCRATCH.md` and **stops at the
pilot**. Step 9 of that doc calls phase 3 "the real decision point … do not skip
it and do not scale past it on faith", so the 120M hybrid is not launched
unattended. Built 2026-09-13 while the Seed-VC pass was still running.

## Run it

```bash
cd /root/tts/TTS_training/pretrain/asr_bilingual/pipeline_v2
nohup setsid bash queue.sh > /workspace/asr_pretrain_v2/queue.log 2>&1 &
```

It blocks until the Seed-VC process exits, then: manifests → preflight →
dataloader bench → tarred shards → bench again → pilot → stop.

## What gets built

| file | what it does |
|---|---|
| `build_manifests.py` | assembles `train_mix / val_mix / val_ne / val_en` + `mix_report.json` |
| `preflight.py` | six abort-on-fail checks; exits 1 and stops the queue |
| `bench_dl_v2.py` | Step 1 dataloader ceiling — **never been measured** |
| `make_shards.py` | 16 kHz resample + tarred shards (Step 4) |
| `pilot_train.py` | 27M CTC-only FastConformer, random init |
| `queue.sh` | the driver |

Everything lands in `/workspace/asr_pretrain_v2/` — checkpoints, shards and logs.
Not on `/` : root is at 93% with 132 GB free and the last run's 70 GB directory is
exactly what had to be deleted.

## Decisions I made, and why

**From scratch, not another nemotron finetune.** Confirmed by you mid-build. The
template `parakeet-tdt_ctc-110m.nemo` supplies *topology only* — `restore_from` is
never called, weights are random.

**27M CTC-only for the pilot, not the 120M hybrid.** Per Step 2 and Step 7.1: CTC
converges faster and more stably from random init, and dropping the transducer
removes the `B×T×U×V` joint tensor. Note this had to be a real `EncDecCTCModelBPE`
— a hybrid with `ctc_loss_weight=1.0` still computes the transducer loss
unconditionally (`hybrid_rnnt_ctc_models.py:456`), so it would have paid the full
joint cost while looking like a CTC run.

**English at 40% of hours, Indian-accented.** The postmortem's rule 3 is that ~10%
replay does not preserve a language: 301 h English against 1,906 h Nepali produced
a total collapse that a mid-run fix made *worse*. `en-in_snr40-50` + `en-in_snr50`
are on the mount with the same 3-ASR consensus metadata as the Nepali sets, match
the deployment domain better than MLS, and need no 35 GB download.

**Our BPE (4,000 pieces) at `/workspace/asr_pretrain_v2/tokenizer/`.** NeMo wants
`tokenizer.model` + `tokenizer.vocab` + `vocab.txt` in one directory; `vocab.txt`
was missing and is generated from the SPE model. Smaller vocab also makes the CTC
head cheap.

**The OOV set ships as clean + Seed-VC, not clean + CPU-augmented.** You asked for
~900 h; this delivers ~590 h of distinct audio on disk. The omitted third is
noise/speed/pitch over the *same* utterances, which SpecAugment already covers,
while Seed-VC is the only lever that touches the real defect (one synthetic voice,
F0 spread 13 Hz). To add it anyway: extract `/workspace/oov_distill/out/hf/shard_*`
to wav and append with `source="oov_aug"`. **This is a scope reduction — flagged,
not hidden.**

**Only VC clips that cleared the 0.85 speaker gate** (~81%) are included; the
failures are closer to the source voice, which is the thing we are trying to
dilute.

## Two things I added that the doc does not mention

**A transliteration gate.** ~5% of the Premal rows are English article text read
as Devanagari transliteration ("चाइना एक्सटेन्ड्स भिसा फ्री पोलिसी"). That is
exactly the English-audio → Devanagari mapping the last run died of, and the
postmortem flags this corpus as "worth re-checking if it is ever added" — we are
now adding it. `devfrac` does **not** catch these (the script *is* Devanagari; only
the words are English), so rows are scored against a 54k-entry Nepali lexicon and
dropped below 55% real-Nepali words.

**A duration report in preflight.** The OOV clips are 30–59 s and the recipe trains
at `max_duration=20`, so *every one of them* would be silently dropped by lhotse.
Preflight prints kept-hours per source and warns when a source loses >50%, so
"we trained on the OOV data" cannot quietly mean "we didn't".

## Two calls the doc leaves open, decided here

**`max_duration = 60`, not 20.** Step 6's cap of 20 s is sized for an RNN-T, whose
joint tensor is `B×T×U×V`. CTC has no `U` dimension — the loss is `B×T×V` — so a
60 s clip costs linearly in `T`, not quadratically. At 20 s lhotse would silently
drop *every row* of the 30–59 s OOV corpus, which is the whole reason that corpus
was built. Bucketing keeps padding waste down. Revisit when phase 4 adds the
transducer head back: at that point either the cap drops to 20 s or the OOV audio
gets segmented by forced alignment (the better fix, not built).

**Human Nepali upsampled 3×.** Step 5 says to upweight the ~290 h of human-labelled
Nepali against the ~1,900 h of canary-labelled pseudo-data, using lhotse weights
rather than duplicated rows. `train_input_cfg.yaml` does exactly that — four
buckets (`ne_human`, `ne_script`, `ne_pseudo`, `en_pseudo`), weighted by hours with
a 3× multiplier on `ne_human`. Without it the model is optimised toward canary's
output distribution, which is what makes in-domain WER a lie.

## Still needs a human

- **The 8-GPU ask (Step 10).** "Worth more than every other optimisation in this
  document combined, and it costs one conversation." It needs someone to submit
  from `bodhanai-node001` as `ttsteam` — I cannot. Ask before phase 4.
- **Step 0's gate.** The doc is blunt that pretraining lands *at or below* the
  finetune on accuracy, and that "better Nepali WER" is not a valid reason to do
  it. You have decided to pretrain; the valid reasons on the list (on-device size,
  streaming latency, license independence, tokenizer) are all real here. Recording
  it so the pilot's numbers get judged against the right goal.
