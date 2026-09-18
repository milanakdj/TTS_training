# Text frontend — `ne_frontend.py`

Call `normalize()` on every string before it reaches `generate_audio()`.

```python
import sys; sys.path.insert(0, "/root/tts/TTS_training/pocket_TTS/frontend")
from ne_frontend import normalize, assert_clean

text = normalize("मैले Foodmandu बाट 45 रुपैयाँमा अर्डर गरें।")
assert_clean(text)                 # raises if anything still encodes to <unk>
audio = m.generate_audio(state, text)
```

## Why

`nepali_bpe4000.model` has no `byte_fallback`. Every Latin letter, every ASCII
digit, the Devanagari digits `४`–`९`, the hyphen, `(`, `)`, `;` and the
typographic quotes all encode to a single `<unk>` (id 0) — and the model does not
mispronounce `<unk>`, it **deletes the word or truncates the rest of the
utterance**. Measured on the 6L student, prompt `NEP_F_CONV_00347.wav`:

| input | before | after `normalize()` |
|---|---|---|
| सन् २०२४ मा ४५ जना विद्यार्थी थिए। | `संसया त्याचमा` (2.2 s) | सन् २०२४ मा ४५ जना विद्यार्थी थिए ✅ |
| मैले 45 रुपैयाँ तिरें। | `मैले` (0.8 s) | मैले ४५ रुपैयाँ तिरेँ ✅ |
| मैले Foodmandu बाट अर्डर गरें। | मैले ⟨—⟩ बाट अर्डर गरेँ | मैले फुटमाण्डुबाट अर्डर गरेँ ✅ |
| उनले BAMS पढेका हुन्। | `उनले` (0.8 s) | उनले बिएमएस पढेका हुन् ✅ |

(ASR column is `himalaya-ai/whisper-large-v3-nepali-final`. It re-renders spoken
number words *as digits*, which is itself the proof they were spoken.)

## What it does, in order

1. Drops bracketed Latin glosses — `ट्याक्सी (Taxi)` is a transcription
   convention, and reading it back says the word twice.
2. Normalises unencodable punctuation and symbols (`%` → प्रतिशत, `₹` → रुपैयाँ).
3. Multi-word `LEXICON` hits, longest first, so `CG Group` beats `CG`.
4. Decimals → `X दशमलव Y`.
5. Per token: `LEXICON` → number → acronym speller → rule transliterator.
   Numbers use `verbalize_ne.py`'s 0–99 table and हजार/लाख/करोड scale; runs of
   ≥7 digits are read digit-by-digit, the way phone numbers actually are.
6. **Charset guard.** Anything still outside the probed-safe set is dropped
   rather than allowed to become `<unk>`.

`LEXICON` carries the 85 terms `true_gap.json` says appear in no corpus, plus
common vocabulary the rule engine would mangle. The rule transliterator is
approximate on purpose — it exists so an unseen brand is *spoken*, however
imperfectly, instead of deleted. **Anything whose pronunciation matters belongs
in `LEXICON`.**

## Verified

Zero unencodable output over 200,001 training transcripts, the full 4,193-entry
`oov_words.full.txt` (digits included) and all 85 true-gap terms. `normalize()`
is idempotent on 20,000 sampled transcripts.

```bash
echo "COVID-19 मा ९८४१२३४५६७ हो।" | ../repo/.venv/bin/python3 ne_frontend.py
```

## Limits

- Transliterated English is Nepali-accented — correct for a Nepali TTS, but it is
  not an English voice. A real English register needs the tokenizer rebuild
  (`byte_fallback` + Latin + digits) and a retrain.
- Adding a `LEXICON` entry is free and takes effect immediately. Prefer it over
  touching the rules.
