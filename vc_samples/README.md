# Edge TTS -> Seed-VC samples

Ten triplets demonstrating the voice-conversion path for Nepali OOV terms.
Each sentence carries at least one term from the "true gap" set — OOV terms that
appear in **neither** the Premal corpus (1,092 h) nor our 2,207 h training mix,
so synthesis is the only way to cover them.

| dir | what |
|---|---|
| `edge/` | Microsoft Edge TTS output, voice `ne-NP-HemkalaNeural` |
| `ref/`  | target speaker, one distinct real human voice per sample |
| `vc/`   | Seed-VC v1 output: Edge content in the reference speaker's timbre |

`manifest.json` has the text and OOV term per sample; `results.json` has the
CAMPPlus speaker similarities.

## Measured on these 10

mean cos(output, reference) **0.800** vs cos(output, source) **0.284** — the
timbre genuinely moves. Only 3/10 clear a 0.85 gate, and the two failures used
the two shortest references. On longer source audio (30–59 s) with references
carrying ≥8 s of post-VAD speech, the same setup reaches **88%** passing at mean
0.890, with CER changing only 3.7% of characters.

Score with CAMPPlus, not MFCC-mean cosine: MFCC placed all ten of these in
0.96–0.998 and would have passed both genuine failures.

## Attribution

`ref/` clips are from **AI4Bharat IndicVoices-R** (`ai4bharat/indicvoices_r`),
used under **CC BY 4.0**. Speaker id and gender are preserved in the filenames.
