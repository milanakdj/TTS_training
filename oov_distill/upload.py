"""Push the distilled + augmented shards to a gated dataset repo on the Hub.

Token comes from /tmp/hf_tok. The card states plainly that the source is a
single synthetic TTS voice -- anyone training on this needs to know that before
they budget speaker diversity, and it is the first thing a consumer would
otherwise have to rediscover the hard way.
"""
import argparse, json, os, glob
from huggingface_hub import HfApi

CARD = """---
license: other
language: [ne]
task_categories: [automatic-speech-recognition]
size_categories: [10K<n<100K]
extra_gated_prompt: >-
  This dataset is derived from a single synthetic TTS voice. Please confirm you
  have read the limitations section before requesting access.
extra_gated_fields:
  Intended use: text
---

# Nepali OOV-distilled subset ({hours:.0f} h)

An OOV-dense distillation of `Premal-12/c9nepali-audio-dataset2` (used with the
author's permission), plus CPU augmentations, for Nepali ASR and TTS work.

## What this is

The source corpus is 1,092 h / 100,000 rows of Nepali news articles read aloud.
This subset selects **{rows_clean} clips ({hours_clean:.0f} h)** by greedy set
cover over a list of {n_lex} out-of-vocabulary *lexical* terms, then fills the
remaining budget with OOV-dense rows. Coverage achieved: **{covered}/{n_lex}
terms ({cov_pct:.0%})**.

Each clean clip is shipped alongside {n_aug} augmented variant(s), giving
**{rows_total} rows / {hours:.0f} h** in total.

## Read this before using it

**The source is a single synthetic TTS voice, not human speech.** Measured over
100 clips sampled across the source corpus:

| measure | value | what a real corpus looks like |
|---|---|---|
| F0 median, p5 / p50 / p95 | 211 / 218 / 224 Hz | spans ~80-250 Hz |
| pairwise MFCC-mean cosine | mean 0.998, **min 0.965** | far lower, far wider |
| KMeans silhouette (best, k=2) | 0.192 | >0.5 with distinct speakers |

Consequences you should plan around:

- **Speaker diversity is one.** The noise, pitch and speed augmentations here
  change channel, F0 and formants; none of them create speaker identity. Voice
  conversion is the only thing that would.
- **Text is not verbatim.** Digits are written as digits but *spoken verbalized*
  (`२१ भदौ` is read "एक्काइस भदौ"; `२०%` is read "बिस"). Useful as free ITN
  supervision for ASR, and as number-pronunciation data for TTS; wrong if you
  need literal transcripts.
- **Article furniture is audible.** The voice reads headlines, section names and
  clock timestamps aloud, so the text retains them on purpose. Stripping them
  from the text would create a misalignment, not fix one.
- Rows whose text was predominantly English (~5% of the source) were dropped.
- Clips are long (30-59 s). Segment before training at `max_duration <= 30`.

## Fields

| field | meaning |
|---|---|
| `id` | unique; augmented rows are `<source_id>_aug<k>` |
| `source_id` | clean row this came from -- **keep pairs on the same split** |
| `text` | article text, whitespace-normalised only |
| `audio` | 16 kHz mono FLAC |
| `variant` | `clean` or `aug<k>` |
| `aug_params` | JSON: speed, semitones, snr_db, band, gain |
| `oov_hits` / `n_oov` | OOV terms present in this row |
| `oov_core` | in the 17.7 h high-redundancy core (see below) |
| `n_oov_learnable` | OOV terms in this row that occur 5+ times corpus-wide |

## Coverage by presence is misleading -- use `oov_core`

The headline "{covered}/{n_lex} terms covered" counts a term as covered if it
appears *once*. Measured on this selection:

- **21% of covered terms appear exactly once**; 44% appear fewer than 5 times.
- The most common term (`सेयर`) appears 10,546 times. The distribution is brutally
  skewed.
- Across the **whole 1,092 h source**, only **263 terms occur 5+ times** -- the
  rest are not learnable from this corpus at any selection size.

So `oov_core = true` marks the **17.7 h** subset (800 clean + 800 augmented rows)
that carries every learnable term at >=5 occurrences. If you want OOV coverage,
train on that; the remaining ~284 h is general-purpose Nepali audio that adds
volume, not vocabulary.

Terms needing synthesis rather than selection: ~85 appear in no corpus at all
(company/brand names -- Foodmandu, CloudFactory, Hamro Patro -- plus acronyms
and four place names).

## Voice-converted variants (`vc`, `vc_aug0`)

The source corpus is a single synthetic voice, so **707 clips of the `oov_core`
subset were re-rendered with Seed-VC v1** against real human references drawn
from AI4Bharat IndicVoices-R (1,076 distinct Nepali speakers). This is the only
thing in this dataset that changes *speaker identity* -- the noise, pitch and
speed augmentations below change channel and F0, not who is talking.

Measured over the 800 converted clips (CAMPPlus speaker embeddings; flex ASR for
intelligibility):

| measure | value |
|---|---|
| cos(output, its reference) | mean **0.890**, p10 0.842, p90 0.928 |
| cos(output, original source voice) | 0.304 |
| clips clearing a 0.85 speaker gate | **707/800 = 88%** |
| CER, flex(converted) vs flex(original) | mean **0.0368**, p90 0.089 |
| clips with >10% of characters changed | 7% |

**Only gate-passing clips are shipped.** `aug_params` on these rows carries
`vc_ref`, `cos_out_ref` and `vc_gate_passed`. Intelligibility cost is small but
real (~3.7% of characters differ); if you need pristine transcription fidelity,
filter to `variant in ('clean','aug0')`.

Two settings mattered and are worth reusing: references need **>=8 s of speech
after VAD trimming** (only 169 of 772 pool clips qualified), and longer source
audio converts far better -- the same setup on 3-4 s clips passed only 30%.

## Augmentation

Speed perturbation (0.9-1.1, resampling so formants move too), pitch shift
(-2..+3 semitones, upward-biased), additive AudioSet noise (SNR 5-25 dB),
telephone-band simulation on 25% of rows, and random gain. Parameters are
recorded per row and seeded from the row id, so the set is reproducible.

## Provenance and licence

Derived from `Premal-12/c9nepali-audio-dataset2` with the author's permission.
The underlying article text was scraped from Nepali news sources and the audio
is synthetic; no licence is asserted over the source text. Gated for that
reason. Not for redistribution without checking the upstream position.
"""

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='milanakdj/nepali-oov-distilled')
    ap.add_argument('--dir', default='/workspace/oov_distill/out/hf')
    ap.add_argument('--report', default='/workspace/oov_distill/out/selected_report.json')
    ap.add_argument('--n-aug', type=int, default=1)
    ap.add_argument('--private', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    tok = open('/tmp/hf_tok').read().strip()
    rep = json.load(open(a.report))
    import pyarrow.parquet as pq
    files = sorted(glob.glob(f'{a.dir}/shard_*/*.parquet'))
    rows_total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    hours = 0.0
    for f in files:
        t = pq.read_table(f, columns=['duration']); hours += sum(t.column('duration').to_pylist())/3600
    card = CARD.format(hours=hours, rows_clean=rep['rows'], hours_clean=rep['hours'],
                       n_lex=rep['oov_lexical'], covered=rep['oov_covered'],
                       cov_pct=rep['oov_covered']/rep['oov_lexical'],
                       n_aug=a.n_aug, rows_total=rows_total)
    open(f'{a.dir}/README.md', 'w', encoding='utf-8').write(card)
    print(f"{len(files)} shards | {rows_total} rows | {hours:.1f} h")
    if a.dry_run:
        print("--dry-run: card written, NOT uploading"); print(card[:1200]); raise SystemExit
    api = HfApi(token=tok)
    api.create_repo(a.repo, repo_type='dataset', private=a.private, exist_ok=True)
    api.upload_large_folder(repo_id=a.repo, repo_type='dataset', folder_path=a.dir)
    print(f"pushed -> https://huggingface.co/datasets/{a.repo}")
    print("NOTE: gating must be switched on in repo settings (settings API), "
          "creation alone does not gate it.")
