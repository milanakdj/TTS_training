# v4: the bilingual recipe, and the evidence for it

**Status (2026-09-23): both stages done and published private** as
`milanakdj/pocket-tts-nepali-en-{6l,24l-teacher}-v4`. Nepali matches v2; English
works but is ~40x worse than untouched Kyutai (2x2 CER 0.356 vs 0.009). The
recipe below fixed v3's Nepali collapse and did not fix English forgetting; see
the v4/v5 sections of `../CLAUDE.md`.

v3 ([`V3_RESULT.md`](V3_RESULT.md)) changed the tokenizer, the embedding init, the
data mix and the alignment pipeline in one 72 h run, and produced an unusable
model with no way to attribute the failure. v4 changed one thing at a time and
measured each against **v2's own logged curve at matched steps** rather than
against itself.

Every number below is valid `flow_loss` on the same Nepali valid set v2 used
(`manifests/valid_probe_ne.jsonl` = `valid_v2_aligned` verbatim; verified 0 of
999 rows leak into v3/v4 training data).

## Phase 1 — the graft is innocent

Nepali-only on v3's exact tokenizer and embedding graft, nothing else changed:

| step | v2 | v3 (27% en) | probe |
|---|---|---|---|
| 2,500 | 0.1843 | 0.2815 | **0.1827** |
| 5,000 | 0.0453 | 0.1771 | **0.0450** |
| 7,500 | -0.0034 | 0.1477 | **-0.0029** |

`eos_loss` agreed (0.0090 vs v2's 0.0087, against v3's 0.0272). So
`tokenizer_v3/ne_en_9682` and `graft_text_embedding: 4000` are sound and stay.
**v3 died of its recipe, not its tokenizer.**

## Phase 2 — what the recipe should be

Three arms, 7,500 steps each, validated per language:

| arm | English replay | inherited rows | Nepali | English |
|---|---|---|---|---|
| **v4b** | **5%** | **free (wd 0.1)** | **-0.0018** | **0.1736** |
| v4a | 5% | pinned, wd 0.0 | 0.0031 | 0.1849 |
| v4c | 0% | pinned, wd 0.0 | 0.0022 | 0.2890 |

**1. The replay ratio was the entire failure.** 27% -> 5% moves Nepali by ~0.15
and restores v2 parity. Consistent with the VLM-forgetting result that replay
gains saturate by 3-10%: everything v3 spent above that came out of the Nepali
budget for nothing.

**2. Pinning the inherited rows caps English.** v4a led at 5k (0.1845 vs 0.2004)
and then *plateaued* (0.1849 at 7.5k) while v4b kept improving and overtook it.
Freezing buys a head start and pays for it with a ceiling: the model can never
adapt those rows to its own acoustic model and speaker distribution. It also cost
Nepali at every step. **Do not pin.**

**3. Replay is required, not optional.** At 0% English, English got *worse* over
training (0.2824 -> 0.2890) with `eos_loss` climbing 0.0319 -> 0.0467 — forgetting
caught live, **with the English embedding frozen**. The backbone is what forgets.

That last point refutes V3_PLAN's central claim:

> English arriving as inherited weights is a structural guarantee. A replay ratio
> is a hope.

The embedding is half the model. The true, weaker, useful statement is that **the
graft makes English cheap (5% suffices), not free.**

## What to do differently, permanently

- **Split valid by language.** `valid_jsonl: ne=…,en=…`. A pooled bilingual loss
  cannot distinguish "good at English, broken at Nepali" from "mediocre at both",
  which is exactly how v3 hid for 200k steps.
- **Judge against the previous generation, at matched steps.**
  `scripts/curve_vs.py <log>` and `SET=en scripts/curve_vs.py <log>`. A flat curve
  is not a healthy one.
- **Sample both languages in-loop.** v3 sampled three stock English sentences for
  200k steps. v4 samples Nepali, English and code-mixed.
- **Stop early.** 250k is an upper bound with `ckpt_freq: 5000`. v2 was flat from
  ~50k. Watch the curve and stop; do not run blind to max_steps.

## Code-mixed cannot be trained

The corpus contains **1.2 h / 371 rows (0.05%)** of naturally code-mixed speech,
and those are IndicVoices translation drills, not Nepali carrying English brand
names. Code-mixed ability has to come from representability plus transfer, so it
is a **generalization test** (`infer/final/oov_probes.json`, and the cross cells
of `pairs_2x2.json`), never a data target.

## The baseline v4 has to beat

v2 across the bilingual 2x2 (50 utterances per cell; Nepali scored with the
finetuned Nepali Whisper, English with base large-v3-turbo — the two columns are
not comparable to each other):

| system | cell | WER | CER |
|---|---|---|---|
| real human | en_text/en_voice | 0.024 | 0.011 |
| real human | ne_text/ne_voice | 0.347 | 0.120 |
| v2 student | ne_text/ne_voice | 0.311 | 0.117 |
| v2 student | ne_text/en_voice | 0.378 | 0.157 |
| **v2 student** | **en_text/en_voice** | **1.328** | **1.030** |
| v2 student | en_text/ne_voice | 1.030 | 0.879 |

**v2 cannot speak English at all** (CER 1.03 — worse than silence, because
deleted `<unk>` words leave insertions). That gap is what the graft exists to
close, and it is now measured rather than assumed.
