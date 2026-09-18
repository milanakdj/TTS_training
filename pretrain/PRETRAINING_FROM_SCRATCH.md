# How to pretrain a Nepali + English ASR model from scratch

Last updated: 2026-09-18
Companion to `WHAT_WE_ARE_BUILDING.md` — read that first for *whether* to do this.
This file is the *how*.

> **2026-09-18 — corrections after run_v7 shipped.** Three claims in this doc were
> wrong and are fixed below: the tokenizer numbers in Step 0/Step 3, and Step 8's
> treatment of held-out sets, which named the right trap but did not say how to
> avoid it — and run_v7 walked into it anyway. Details at each site.

---

## Step 0 — The gate. Do not skip.

Pretraining is 3–6 weeks of H100 versus ~2 days for the finetune, and on our data
it lands **at or below** the finetune on accuracy, because 86% of our Nepali
carries canary's labels and canary is 0.155 WER on gold. A from-scratch model
trained on those labels is distillation from canary.

**Write down which of these is the goal before starting.** If none of them is,
stop and use the finetune.

| valid reason | why the finetune can't do it |
|---|---|
| on-device size | 0.6b is too big for phones; we want ~30–120M |
| streaming latency | base encoder's att-context is fixed |
| license independence | not bound to NVIDIA's weights |
| tokenizer | the shipped BPE **cannot encode Nepali at all** (48.9% UNK); ours is 1.70 tok/word at 0% UNK — see Step 3 |

"Better Nepali WER" is **not** on this list. The binding lever there is
human-labeled hours, not GPU-days or parameters.

---

## Step 1 — Measure the dataloader ceiling FIRST

**This single measurement decides whether the whole plan is 3 weeks or 3 months,
and it has never been run.**

The schedule is denominated in *audio hours seen*, not FLOPs. The dataloader feeds
the same bytes/sec whether the model is 600M or 32M, so if I/O caps at ~80x
realtime, shrinking the model buys nothing and every estimate below roughly
doubles.

```bash
source /workspace/venvs/nemo/bin/activate
cd /workspace/asr_bilingual
python bench_dl.py 8     # workers=8
python bench_dl.py 16    # workers=16 — we have 16 cores
```

`bench_dl.py` already exists and iterates the real training dataloader with no
model attached. Interpretation:

- **lands near the model's step rate** -> I/O bound. Fix the data layout (Step 4)
  before anything else; model surgery is wasted.
- **far above** -> compute bound. Shrinking the model and dropping the RNN-T joint
  will actually pay off.

Run this while the GPU is otherwise idle, and do not trust a short window — the
`--bench` lesson applies (see `WHAT_WE_ARE_BUILDING.md`, gotcha 1).

---

## Step 2 — Architecture

**Recommendation: cache-aware streaming FastConformer, Hybrid CTC + RNN-T, ~120M.**

| choice | why |
|---|---|
| **FastConformer** over Conformer | 8x depthwise subsampling instead of 4x — halves encoder sequence length, the single biggest throughput win available |
| **Hybrid CTC+RNNT** (`EncDecHybridRNNTCTCBPEModel`) | train both heads on one encoder. CTC gives a cheap, stable early signal and fast greedy eval; RNN-T gives the better final WER. You can decode with either. |
| **cache-aware streaming** | if streaming latency is the goal, it must be trained in — you cannot bolt it on afterwards. Set `att_context_size` to the target, e.g. `[70,13]`, and train with `att_context_probs` for multi-latency. |
| **~120M** | NeMo Conformer-CTC small (~13M) already reaches low single-digit WER on LibriSpeech with 960 h. 13–32M is legitimate for Nepali; 120M is the safer bet for open-domain bilingual with a Devanagari vocab. Below ~10M is keyword-spotting territory, not free-form speech. |

If the *only* goal is on-device size, drop to 32M CTC-only and skip the RNN-T head
— that removes the `B x T x U x V` joint tensor entirely, which is where most of
the memory and a large share of the time goes.

NeMo config to start from:
`examples/asr/conf/fastconformer/hybrid_cache_aware_streaming/fastconformer_hybrid_transducer_ctc_bpe_streaming.yaml`

---

## Step 3 — Tokenizer

**Already done** — `/workspace/asr_bilingual/ne_en_bpe.model`, built from
`tok_corpus.txt`. (Identical file to `/workspace/asr_pretrain_v2/tokenizer/tokenizer.model`,
md5 `17ea43f5` — they are the same tokenizer under two names.)

> **Corrected 2026-09-18.** This section used to say "**2.39 tok/word** vs the
> shipped 5.12", and claimed a ~2.1x shortening of RNN-T targets. Both numbers
> were wrong. Measured on 20k rows each of `train_ne_pseudo` and `train_ne_human`:

| tokenizer | vocab | tok/word (ne) | UNK rate |
|---|---|---|---|
| shipped `parakeet-tdt_ctc-110m` BPE | 1,024 | 2.06 — *meaningless* | **48.9%** |
| ours (`ne_en_bpe.model`) | 4,000 | **1.69–1.74** | **0.0%** |

The shipped BPE is 1,024 English pieces with **no byte fallback**, so essentially
every Devanagari word collapses to one `<unk>`. Its apparent 2.06 tok/word is the
cost of *discarding* Nepali, not encoding it — the tokenizer is unusable for this
language at any token budget, which is a stronger argument for building our own
than the old framing gave.

And **5.12 was never the shipped tokenizer's output**: it is simply the character
count per Nepali word (measured 5.09), i.e. what character-level fallback costs.
So the real gain is ~3x over character fallback, not 2.1x over a working BPE.

The conclusion is unchanged and now better supported: in a finetune you cannot
swap the vocab (it reinitialises the joint + prediction net and forces the English
head to be relearned). **From scratch, it is free.** Use it.

Sanity checks before committing:
```bash
# tok/word on a held-out Nepali sample, and UNK rate
python - <<'PY'
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="/workspace/asr_bilingual/ne_en_bpe.model")
# assert: no text tokenizes to zero tokens; UNK rate ~0 on gold text
PY
```
Fix the normalization the current tokenizer gets wrong: strip `( ) : ; " ' —`,
ZWJ/ZWNJ, and map Devanagari digits `०-९` -> `0-9`. Those currently become
trainable `<unk>` targets (gotcha 6) and there is no reason to inherit that.

---

## Step 4 — Make the data fast (this is the real engineering)

773,780 small files on a network filesystem is **seek-bound**. This does not
matter much for a 2-day finetune; over 50–150k hours seen it dominates.

1. **Resample to 16 kHz offline, once.** Source is 24 kHz and NeMo resamples on
   CPU every single batch, for every epoch. Doing it once on disk removes that
   permanently.
2. **Convert to tarred / Lhotse Shar shards.** Sequential reads instead of 773k
   seeks. This is the change that moves the dataloader ceiling.
   ```bash
   python <NeMo>/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \
     --manifest_path=manifests/train_mix.jsonl \
     --target_dir=/workspace/asr_pretrain/tarred \
     --num_shards=2048 --max_duration=20 --min_duration=0.3 \
     --shuffle --shuffle_seed=42 --workers=16
   ```
   Then `is_tarred: true`, `tarred_audio_filepaths`, `shuffle_n: 2048`.
3. **Keep bucketing on** (`use_bucketing: true`, `num_buckets: 30`,
   `batch_duration`) to minimise padding waste.
4. Budget disk: ~2,200 h of 16 kHz mono 16-bit ≈ **250 GB**. With English scaled
   to 1:3, plan for ~1 TB. `/workspace` has 839 TB free — not a constraint.

Re-run Step 1 after this. If the ceiling has moved, every timeline below improves.

---

## Step 5 — Data assembly

Nepali is nearly capped; the open-source additions add ~9% hours but **~64% more
human supervision**, which is the part that matters.

| source | hours | labels | how |
|---|---|---|---|
| current `ne` | 1,647 | pseudo | already on disk |
| current `ne-NP` | 259 | human | already on disk |
| OpenSLR SLR54 | ~157 | human | openslr.org/54 |
| FLEURS ne_np | ~7 | human | HF `google/fleurs` |
| Common Voice ne | ~2 | human | HF `mozilla-foundation/common_voice_*` |
| **Nepali total** | **~2,070** | **~425 h human** | |

English is unbounded — you pick a cap, not a maximum. `gigaspeech`, `libriheavy`,
`commonvoice` are already local; MLS-en (44.5k h), People's Speech (30k h) are
available. Ratios: 1:1 -> ~4.2k h total; 1:3 -> ~8.3k h total.

**Weighting matters more than volume.** Upsample the 425 h of human Nepali
relative to the 1,647 h of pseudo-labeled, so the model is not overwhelmingly
optimised toward canary's output distribution. Lhotse supports per-source weights
in a multi-source config — use them rather than physically duplicating rows.

---

## Step 6 — Training recipe

```yaml
model:
  sample_rate: 16000
  tokenizer: { dir: /workspace/asr_bilingual, type: bpe }   # the 1.70 tok/word one (Step 3)
  train_ds:
    is_tarred: true
    use_lhotse: true
    use_bucketing: true
    num_buckets: 30
    batch_duration: 600          # tune to fit ~70GB; CTC-only fits far more than RNN-T
    max_duration: 20
    min_duration: 0.3
    num_workers: 16
  optim:
    name: adamw
    lr: 2.5e-3                   # from scratch wants ~25x the finetune's 1e-4
    betas: [0.9, 0.98]
    weight_decay: 1e-3
    sched:
      name: NoamAnnealing        # or CosineAnnealing with explicit max_steps
      warmup_steps: 15000        # from scratch needs a LONG warmup; 2k will diverge
      min_lr: 1e-6
      max_steps: <set explicitly>
```

Non-negotiables learned the hard way (all in `WHAT_WE_ARE_BUILDING.md`):

- `logger=False, enable_checkpointing=False` on the Trainer — `exp_manager` owns both.
- `max_steps` **explicit** in the sched config; lhotse datasets have no `__len__`.
- `val_check_interval` an **int**; `every_n_epochs: 0` or nothing ever checkpoints.
- Validate every manifest's distinct language tags against the prompt/vocab config
  before launch — a bad key surfaces thousands of steps in, not at startup.
- `precision: bf16-mixed`.
- Measure throughput **past step 1000**, never from a cold-start bench.

---

## Step 7 — Curriculum

1. **CTC-only warmup, ~20–30k steps.** CTC converges faster and more stably from
   random init than RNN-T, and gives a usable signal early. With a hybrid model,
   weight the CTC loss high initially (`ctc_loss_weight: 1.0`) then anneal toward
   the transducer (`0.3`).
2. **Short utterances first.** Cap `max_duration` at ~10 s for the first few
   thousand steps, then lift to 20 s. Long sequences early destabilise attention.
3. **English-heavy early, Nepali-heavy late.** English has clean human labels and
   teaches the encoder general acoustics; shift the sampling weights toward Nepali
   as it stabilises.
4. **SpecAugment on from the start** (`freq_masks: 2, time_masks: 10`). From
   scratch on 2k h will overfit without it.
5. **Full-context first, streaming last** if targeting streaming: train with full
   context, then finetune with cache-aware masks. Converges faster than
   streaming-from-random-init.

---

## Step 8 — Eval gates, and the trap

Use `/workspace/asr_bilingual/eval_ckpt.py`. Test sets: `ne_test.jsonl` (600,
human), `en_test.jsonl` (600).

**Trap 1 — the in-domain number.** A from-scratch model will score ~3–6% WER on a
held-out split of the training corpus, and **that number is meaningless** — it
measures agreement with canary's labels, not accuracy. This is the same artifact
that made `flex` look like 0.013 when it was really 0.155. Never quote an
in-domain number.

**Trap 2 — the "held-out" set that isn't. Added 2026-09-18, because run_v7 fell
into it while this doc was open.** The version of this section that only warned
about trap 1 was not enough. run_v7's eval set, `hum_heldout.jsonl`, was built by
*sampling* 200 human rows — and never *removing* them from
`train_ne_human.jsonl`, which then trained at weight 0.2115. All 200 clips were
seen ~61 times. The published WER of 0.176 was a memorisation score; the real
figure on a genuinely unseen human set is **0.302** (FLEURS `ne_np`), a 1.7x gap.
Nobody noticed for a month, because the filename said `heldout`.

**So: build eval sets by subtraction, and prove it.** Never by sampling.

```bash
# 1. carve the eval set OUT of the training manifest, and rewrite training
python - <<'PY'
import json, random
rows = [json.loads(l) for l in open("train_ne_human.jsonl")]
random.Random(0).shuffle(rows)
held, keep = rows[:200], rows[200:]
for name, rs in (("hum_heldout.jsonl", held), ("train_ne_human.jsonl", keep)):
    with open(name, "w") as f:
        for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
PY

# 2. PREFLIGHT, as a hard gate before any launch: zero overlap, or do not train
python - <<'PY'
import json, sys, glob
held = {json.loads(l)["audio_filepath"] for l in open("hum_heldout.jsonl")}
for m in glob.glob("train_*.jsonl"):
    n = sum(json.loads(l)["audio_filepath"] in held for l in open(m))
    if n: sys.exit(f"ABORT: {n} eval rows are in {m}")
print("preflight ok: eval set disjoint from all training manifests")
PY
```

Path identity is the cheap check; also compare **canonical text**, since a
re-scrape under a different filename is the realistic way contamination enters.
`release_v7/gold_eval/verify_disjoint2.py` is the worked version (path, basename,
exact text, and Jaccard near-duplicates). Two lessons from writing it: a
">=3 shared 5-grams" near-dup rule fires on ~5% of a news corpus and tells you
nothing — use Jaccard ≥ 0.6; and make the canonical form *lossier* than the real
text, so it over-collides and a zero-overlap verdict is conservative.

Gates, checked at every eval:

| gate | threshold | action if failed |
|---|---|---|
| eval set disjoint from training | **0 overlapping rows** | abort; this is a preflight, not an eval-time check |
| Nepali WER on **human gold** | improving | if flat 3 evals running, stop |
| English WER | not regressing >10% relative | rebalance sampling |
| in-domain vs gold WER gap | < 3x | you are fitting canary, not speech |
| loss | finite | NaN -> lower LR, lengthen warmup |

The "flat 3 evals -> stop" gate is only as good as your willingness to obey it.
run_v7 was flat from step ~35k to 50k — six consecutive evals — and ran to 50k
anyway. Write the stop into the driver script, not into your intentions.

Final claims go through the himalaya-ai whisper instrument with the doubled-SOT
prefix, plus **human spot-checks**. Automatic scorers are not valid instruments
for Nepali on their own.

Average the top 5–10 checkpoints before the final eval (`scripts/checkpoint_averaging`)
— usually worth ~3–5% relative for free.

**But check, don't assume.** On run_v7 it was a null: averaging all four saved
checkpoints scored 0.3039 WER on gold against the final step's 0.3021 — slightly
*worse*. With `save_top_k: 3` on a run that had already plateaued and annealed,
every saved checkpoint sits in the last ~5k steps, i.e. at effectively the same
point in weight space, so there is nothing to average away. If you want this to
pay, save checkpoints further apart than the plateau is wide.

---

## Step 9 — Phased plan with kill criteria

| phase | what | wallclock (1x H100, est.) | kill criterion |
|---|---|---|---|
| 0 | Step 0 gate + `bench_dl.py` | 1 day | no valid non-accuracy goal -> stop |
| 1 | 16 kHz resample + tarred shards | 1–2 days | ceiling doesn't move -> reconsider scale |
| 2 | Tokenizer re-check + config | 1 day | — |
| 3 | **Pilot: 32M CTC, 5k h seen** | 2–3 days | loss not converging -> fix before scaling |
| 4 | Scale to 120M hybrid, 50k h seen | ~2 weeks | gold WER worse than finetune -> stop |
| 5 | To 150k h seen + streaming finetune | ~4 weeks | gold WER flat 3 evals -> stop, average, ship |
| 6 | Checkpoint averaging, export, eval | 2 days | — |

**Phase 3 is the real decision point.** A 2–3 day pilot tells you whether the
recipe converges at all, for ~5% of the total cost. Do not skip it and do not
scale past it on faith.

---

## Step 10 — The lever that actually matters

Everything above is single-GPU engineering around a constraint that is
**organizational, not technical**: this Slurm job has 1 GPU of the node's 8, for
365 days. Eight GPUs divides every timeline by ~7 — turning phase 5 from 4 weeks
into 4 days.

That requires someone to submit from `bodhanai-node001` as `ttsteam`:

```bash
srun -p defq --gres=gpu:8 -N1 --cpus-per-task=64 --mem=256G --pty bash
# or sbatch with #SBATCH --gres=gpu:8
```

**Ask for the allocation before starting phase 4.** It is worth more than every
other optimisation in this document combined, and it costs one conversation.
