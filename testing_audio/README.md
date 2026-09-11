# testing_audio

Five Nepali samples from each corpus directory that is **neither already public nor
on the Hub**, pulled 2026-09-09 so the upload question can be decided by listening
rather than by reading hour counts.

Every one of these directories is **multilingual** — the first row of each manifest is
Hindi, Bengali, Assamese, Punjabi or English. Rows here are filtered to Nepali the same
way `dataset/scripts/nepali_inventory.sh` counts them (`language` in `ne/nep/nepali`, or
a `/Nepali/` path), so these clips are the Nepali subset, not a random sample of the dir.

One clip per distinct `feed_id` for the podcast folders: the first pass took five
consecutive chunks of one sermon podcast, which said nothing about the corpus.

`metadata.json` in each folder carries the full row: transcript, duration, sample rate,
speaker count, DNSMOS/SNR, feed and episode title, source URL, and the local path the
audio was read from.

| folder | Nepali h on disk | clips | audio | provenance |
|---|---|---|---|---|
| `gemini_vc` | 11.9 | 5 | 24 kHz Gemini TTS voice-converted to Rasa reference speakers | synthetic, but VC targets real Rasa speakers — gate it |
| `indicvoices-r-long` | 629.1 | 5 | 24 kHz sidon-restored long-form read/conversational | AI4Bharat IndicVoices-R, CC-BY-4.0 upstream — derivative is redistributable with attribution |
| `ncert_conversations` | 148.6 | 5 | 24 kHz synthetic 2-speaker educational dialogue (gemma_tts_codemixed) | synthetic — redistributable |
| `orpheus_snr40-50` | 16.2 | 5 | 24 kHz short clips, SNR band 40–50 | YouTube-derived — **not redistributable** |
| `podcast-index-audiobook` | 363.2 | 5 | 48 kHz, single-speaker audiobook/recitation chunks | published books read aloud — **not redistributable** |
| `podcast-index-feeds-5min` | 6,680.8* | 5 | 48 kHz, 2-speaker diarized 5-min chunks, ASR text with SPEAKER_nn labels | third-party podcasts — **not redistributable** |
| `podcast-index-feeds` | 115.7 | 5 | 44.1–48 kHz whole episodes, **no transcripts** | third-party podcasts — **not redistributable** |
| `podcast-index` | 54.0 | 5 | 24 kHz short clips, text kept where 3 ASRs agreed | third-party podcasts — **not redistributable** |

\* `podcast-index-feeds-5min`'s 6,680.8 h is inflated: `language: ne` means Nepali is the
*dominant* language of a 5-minute chunk. Sample 00 carries `languages: ['en','ne']` — read
`nepali_seconds` vs `other_language_seconds` in `metadata.json` per clip.

## Not sampled

`gemini_vc_conversational`, `_corrected` and `_pairs` (~130.5 h, count once) have **no audio
on this box**. Their manifests point at `/projects/data/ttsteam/nikhil/...` and
`/projects/data/ttsteam/avnish/...`; only `/projects/data/ttsteam/sunil` is still mounted.
Manifest-only, so nothing can be uploaded from here.

## Listening

Total 389 MB. From a laptop:

```bash
scp -r <node>:/root/tts/TTS_training/testing_audio ./
```

Sizes are as the corpus stores them — 5-minute 48 kHz chunks are ~30 MB each. Only
`podcast-index-feeds` is trimmed (60 s from 0:30), because its rows are whole 29-minute
episodes and five of them was 1.5 GB on a root disk at 94%.
