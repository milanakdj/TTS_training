# v3: Nepali + English on a grafted tokenizer

**Status: data prep.** Nothing has trained yet.

## The problem v3 fixes

v2 shipped a tokenizer, `nepali_bpe4000`, trained from scratch on Nepali
transcripts that had already been verbalized. It has **no `byte_fallback`**, so
every Latin letter, every ASCII digit, `४`–`९`, `-`, `(`, `)`, `;` and the
typographic quotes encode to a single `<unk>`. The model does not mispronounce
`<unk>` — it **deletes the word or truncates the rest of the utterance**.

This was never a shortage of data. The Kyutai base checkpoint's own tokenizer has
`byte_fallback` **and** `split_digits`; the v2 rebuild threw away a property the
base model shipped with.

## The three changes

### 1. Graft the tokenizer instead of replacing it

`tokenizer_v3/ne_en_9682.model` = Kyutai's 4,000 pieces, unchanged, **plus** 5,682
appended Nepali pieces (`scripts/build_tokenizer_v3.py`).

| | `nepali_bpe4000` | Kyutai base | **`ne_en_9682`** |
|---|---|---|---|
| tokens / Nepali utterance | 38.7 | ~55 | **34.8** |
| utterances with `<unk>` (of 40k) | 662 | 0 | **0** |
| English representable | no | yes | **yes** |

It is cheaper per utterance than the tokenizer it replaces, not a trade.

### 2. Inherit the English embedding instead of relearning it

Because ids 0–3999 are byte-identical, row *i* of the pretrained
`flow_lm.conditioner.embed.weight` still means piece *i*. So stage 1 uses
`graft_text_embedding: 4000` rather than `reset_text_embedding: true`:

```
rows 0..3999      <- pretrained (English + the 256 byte-fallback pieces)
rows 4000..9681   <- fresh init (the appended Nepali pieces)
row  9682         <- pretrained row 4000, the padding row, moved to the new tail
```

That last line is the part that is easy to get wrong. `conditioner.embed` is
`nn.Embedding(n_bins + 1, dim)` and the **last** row is padding, not a piece — so
growing the vocabulary is not a plain append. Leave the padding row at index 4000
and every Nepali piece trains against a row the model already reads as "no text
here". `training/modules/builders.py:_graft_embedding` handles it; there is a
unit-test-shaped check in the commit that added it.

**Why this matters more than the data mix.** English arriving as inherited
weights is a structural guarantee. A replay ratio is a hope.

### 3. Add Indian English replay

~800 h from `en-in_snr{50,40-50}` — 5,397 h available, 24 kHz, already through the
same 3-ASR consensus gate as `ans_*`. Indian rather than US English on purpose:
the register this model needs is a South Asian speaker reading Nepali brand names,
not General American. `gigaspeech` was rejected despite having 9,998 h because it
carries no speaker labels; `emilia` (46,646 h) is the wrong register.

## What is deliberately NOT changed

**Numbers stay in the text frontend.** Nepali 0–99 is an irregular lookup table,
not a composition rule. A verbalizer gets it right every time; training signal
spent teaching it from audio would be strictly worse. The tokenizer change buys
*representability*, so a bypassed frontend degrades to reading digits aloud
instead of deleting the sentence. `ne_frontend.normalize()` now also runs over the
training transcripts, so train and inference see the same text distribution.

## Traps found while building this

- **`speaker_id` is a bare per-video diarization label.** Two videos' `SPEAKER_01`
  are different people. `build_manifest_v3.py:speaker_key` keys on the
  channel/video directory as well. (Harmless for training — the loader takes the
  voice prompt from the same file — but it silently corrupts eval pairing.)
- **`align_data --resume` keys on `(path, start)`, not the transcript.**
  `normalize()` changed 1.8% of Nepali transcripts, and word timings describe a
  specific transcript. Seeding those rows from v2 would make the aligner skip
  exactly the rows that need redoing. `scripts/seed_align_v3.py` compares the text
  before reusing an alignment.
- **The Nepali aligner cannot align English.** `align_data` raises on a transcript
  whose characters are outside the CTC model's alphabet, so the two languages need
  separate passes — `scripts/align_v3.sh`.
- **Alignment is optional but not free.** Without `words` the loader falls back to
  a random prompt window and skips the trailing-silence trim; 12% of utterances
  carry >1 s of trailing silence, which teaches the model to emit silence instead
  of EOS so generations never terminate.

## Run order

```bash
python scripts/build_tokenizer_v3.py                  # tokenizer_v3/
python scripts/build_manifest_v3.py --en-hours 800    # train_v3.jsonl, valid_v3.jsonl
python scripts/seed_align_v3.py                       # split + reuse v2 alignments
bash   scripts/align_v3.sh                            # two aligner passes
bash   scripts/train_teacher_v3.sh                    # stage 1, ~36 h H100
# then set distill_teacher_weights in configs/nepali_distill_v3.yaml
bash   scripts/train_student_v3.sh                    # stage 2, ~36 h H100
```

## Baseline to beat

`infer/final/oov_probes.json` — 22 probes over digits, brands, acronyms,
alphanumerics, symbols and English, scored by **tail retention**: did the last
content word of the sentence get spoken at all. WER is the wrong headline here,
because the v2 failure was *deletion*, and an average buries that in the same
range as ordinary mispronunciation.

| v2 student 6L | tail retention |
|---|---|
| raw input | **10/22 (45%)** |
| through `ne_frontend.normalize()` | **20/22 (91%)** |

Under raw input `symbol`, `phone` and `english` are 0/n, `devanagari-digits` 1/3,
`acronym` 1/2. With the frontend on, the only remaining failures are the two pure
English probes — transliteration renders them Nepali-accented
(`There are thirty-six students` → `थेरेरे, थिटेसिक्स स्टुडेन्स इन थे क्लसस`), which is
by design and exactly what the v3 English replay exists to fix.

**v3 targets: beat 10/22 raw, and move `english` off 0/2.**

Two metric traps, both hit on the first pass:

- Anusvara and candrabindu (`ं` / `ँ`) are interchangeable in Nepali, so `तिरें`
  and `तिरेँ` are one word. Not folding them scored three correct utterances as
  content loss — the first frontend number came out 17/22 instead of 20/22.
- The raw and frontend runs were writing to the same wav directory and
  overwriting each other. They are split by mode now, for the same reason
  `run_final.sh` purges before generating.

## Open

- The eval set must grow a raw-digit / raw-Latin / acronym block. v2's
  `05_numeric` probe was written pre-verbalized, which is why 200k steps never
  surfaced the defect.
- The native-accent question from ADR-013 is untouched here: `filter_native.py`
  is still not applied, so ~296 h of non-Nepal-native AI4Bharat speech remains.
