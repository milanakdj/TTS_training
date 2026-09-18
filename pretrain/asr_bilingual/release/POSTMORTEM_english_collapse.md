# Why the English half of this run failed

Written 2026-09-13, from the ne+en finetune of `nvidia/nemotron-3.5-asr-streaming-0.6b`
that produced this checkpoint. Nepali succeeded (WER 1.440 -> 0.195). English went
from working to emitting pure garbage. This is what went wrong, so the next run
does not repeat it.

## Outcome

| en_test (600 clips) | raw WER | normalised WER | CER |
|---|---|---|---|
| base model, before finetuning | 0.283 | **0.109** | 0.130 |
| @113k steps | 0.965 | 0.963 | 0.832 |
| @147k steps (this checkpoint) | 1.000 | **1.000** | **2.898** |

Nepali over the same period improved monotonically (normalised WER 0.199 -> 0.180).
The two languages were not equally affected — English alone was destroyed.

## Two separate defects, not one

### 1. Missing `prompt_mode` routed half of every batch to the wrong prompt

`LhotseSpeechToTextBpeDatasetWithPromptIndex` reads the prompt mode from
`cut.custom["prompt_mode"]`, falling back to `default_prompt_mode='unified'` with
`unified_auto_ratio=0.5` (`nemo/collections/asr/data/audio_to_text_lhotse_prompt_index.py:80`).
Our manifests had no such field. Measured directly, 96 cuts, 8 en + 8 ne:

| manifest | en-US(0) | ne-NP(46) | auto(101) |
|---|---|---|---|
| `prompt_mode: langID` | 48 | 48 | **0** |
| field absent (what we ran) | 24 | 30 | **42 (44%)** |

So ~50% of every batch trained the language-agnostic `auto` slot. In a 90.6%-Nepali
mixture `auto` learned to mean "emit Devanagari", and the `en-US` slot saw only
~4.7% of steps.

The startup log line `default_prompt_mode=unified, unified_auto_ratio=0.5` prints
identically whether or not the field is present, so it is **not** a check. The only
reliable check is to iterate one batch and inspect the prompt-index tensor
(`batch[4]`) — expect zero 101s.

### 2. The English replay budget was far too small — this was the real killer

301 h English against 1,906 h Nepali (9.4% of rows, 13.6% of hours), sharing one
RNN-T decoder and joint. A prompt vector is too weak a conditioning signal to hold
two orthographies apart against that gradient imbalance.

**Evidence that defect 2 dominates:** we fixed defect 1 mid-run (set
`prompt_mode: langID` on all 773,780 rows, resumed at step 120,869 with LR still at
81% of peak, giving English ~297k steps of correctly-prompted data). English did not
recover — **it got worse, faster**:

| checkpoint | en normalised WER | en CER |
|---|---|---|
| pre-fix | 0.973 | 0.832 |
| post-fix +1 | 1.000 | 0.957 |
| post-fix +2 | 1.000 | 0.981 |
| post-fix +3 | 1.000 | 1.664 |
| post-fix +4 | 1.000 | **2.898** |

Plausibly the fix *accelerated* the collapse: beforehand English at least got gradient
through the heavily-exercised `auto` slot; afterwards it was confined to slot 0 while
every shared component saw pure Nepali.

## Failure signature worth recognising

The end state is an RNN-T degeneration loop, not ordinary forgetting:

- Every English clip decodes to ~540 characters of the single rare glyph `ऱ` (U+0931),
  against a ~194-character reference. CER above 1.0 means the hypothesis is longer
  than the reference — that is the tell.
- `en-US` and `auto` prompts give identical garbage, so it is a whole-model collapse,
  not a bad prompt slot.
- An earlier, milder stage is more diagnostic: at 113k steps the model **heard English
  correctly and wrote it in Devanagari** — "for timothy was a spoiled cat and he
  allowed no one to interfere" -> `फर टिमथी वाज अ स्पोइल्ट क्याट एन्ड हि लाउड नो वन टु इन्टरफेयर`.
  Near-perfect recognition, wrong output script. **The encoder was never the problem.**
  Only the decoder/joint's script prior collapsed.

## It went unnoticed for 21 hours

`checkpoint_callback_params.monitor: val_wer` validated on `ne_val` only. Checkpoint
selection was therefore blind to English, and `val_wer` kept improving while English
died. Nothing in the training logs indicated a problem.

## Rules for the next run

1. **Set `prompt_mode: langID` on every row of every manifest** — train and validation.
   Verify by inspecting `batch[4]` for one batch; do not trust the log line.
2. **Validate every language you claim to support.** Either multi-dataloader validation,
   or score the minority language on each checkpoint out-of-band (45 s per checkpoint).
   A single-language monitor will hide a total collapse of the other.
3. **Do not expect ~10% replay to preserve a language under full finetuning.** Either
   go to a genuinely balanced mixture (30-40%+, or a language-balanced sampler that
   equalises per-language steps rather than per-language hours), or stop fighting it:
   - **adapters/LoRA over a frozen base** structurally preserves the base language, or
   - **two models routed by language ID at inference** — zero compromise on either.
4. **Watch CER, not just WER.** WER saturates at 1.000 and hides how bad things get;
   CER kept climbing (0.96 -> 2.90) and was the only metric still showing the trend.
5. If the goal is monolingual Nepali, **drop the English replay entirely** rather than
   carrying a token amount of it. It bought nothing here and cost a shared decoder.

## Not a factor

The `Premal-12/c9nepali-audio-dataset2` corpus contains ~55 h of English article text
read aloud as Nepali transliteration, which would teach exactly this
English-audio -> Devanagari mapping. It is a plausible-looking culprit and it is
**innocent**: that corpus is not in this training mix. The mix is 71%
`ans_snr40-50`/`ans_snr50` web audio, plus indicvoices-r, rasa, and mahadhwani.
Worth re-checking if it is ever added.
