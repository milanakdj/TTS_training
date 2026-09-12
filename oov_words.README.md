# oov_words.txt

Out-of-vocabulary terms for Nepali, found by greedy-matching our corpus against a
larger one.

**This file has been redacted.** The original list was 4,193 entries, of which
**3,498 were digit strings** — years, decimals, amounts, and ~1,000 numbers in
Nepali mobile/landline format scraped from news text. Those have been removed:
publishing them as an aggregated, searchable list is a privacy risk, and they
were never an audio problem in the first place.

Numbers are handled in text by a verbalizer, not by finding audio for each one:
Nepali 0–99 is irregular (a lookup table, not a composition rule), scale is
हजार/लाख/करोड, and any run of ≥7 digits is read digit-by-digit — `७०४४८३३` becomes
"सात शून्य चार चार आठ तीन तीन". That single rule collapses ~1,052 "missing words"
into the ten digits the model already knows.

What remains here are the **695 lexical terms** that genuinely need audio:
company and brand names in Latin script (Foodmandu, CloudFactory, Hamro Patro),
acronyms (AYUSH, BAMS, DUDBC), Devanagari loanwords (डिस्कनेक्ट, इन्स्टल), and
institution names (नेपाल वायुसेवा निगम).

Of those 695: 601 appear in the Premal corpus, 9 more in our training mix, and
**85 appear in no corpus at all** — those are the synthesis targets, and
`vc_samples/` demonstrates the path.
