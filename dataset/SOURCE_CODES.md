# Reserve archive source codes

Maps the opaque `src` codes back to the corpus directory each row came from. Each code
is its own gated dataset repo, `milanakdj/nepali-audio-reserve-<code>`; the `src` column
inside a repo is constant and equal to its code. **Tracked on purpose** — this is the only record of what
each code means, and losing it would make the archive unreadable.

That makes this file the provenance key for the whole reserve archive, so
**the `TTS_training` GitHub remote must be private.** It was public as of 2026-09-09;
check before pushing:

```bash
gh repo view milanakdj/TTS_training --json visibility
```

Per-row `id` -> original path mapping stays out of git as data rather than docs:
`release/id_map_reserve.jsonl`, alongside `release/id_map.jsonl`. A second file,
`release/id_map_reserve_backfill.jsonl`, holds entries recovered by
`release/backfill_idmap.py` after the packager's tail drain was found to skip the last
partial batch of each source; merge the two when the run finishes.

| code | directory under `/workspace/proc_data_new` | rows | Nepali h | min/row | rate | provenance |
|---|---|---|---|---|---|---|
| `r1` | `indicvoices-r-long` (the 17,014 files NOT already in `nepali-speech-archive`) | 17,014 | 548.1 | 1.93 | 24 kHz | AI4Bharat IndicVoices-R, CC-BY-4.0 upstream |
| `r2` | `ncert_conversations` | 2,128 | 137.0 | 3.86 | 24 kHz | synthetic (`gemma_tts_codemixed`) |
| `r3` | `orpheus_snr40-50` | 5,062 | 16.2 | 0.19 | 24 kHz | YouTube-derived |
| `r4` | `podcast-index` | 24,303 | 53.9 | 0.13 | 24 kHz | third-party podcasts |
| `r5` | `podcast-index-audiobook` | 4,641 | 363.2 | 4.70 | 48 kHz | published audiobooks; a `category:books` view over `r6`'s chunks |
| `r6` | `podcast-index-feeds-5min` | 82,041 | 6,317.6* | 4.62 | 48 kHz | third-party podcasts, 2-speaker diarized 5-min chunks |
| | **total** | **135,189** | **7,436.0** | | | |

Rows and hours are MEASURED from the manifests after every filter the packager applies
(no transcript, zero duration, archived path, cross-source chunk dedupe) -- not the
nominal inventory figures, which overstated `r2` by 11.6 h and `r6` by 363.2 h. The row
counts are low relative to the hours because these are long recordings: `r5` and `r6`
average ~4.7 minutes per row.

\* dominant-language count: `language: ne` means Nepali is the *dominant* language of a
5-minute chunk. Per-row truth is in the uploaded `lang_sec` column.

## Excluded, and why

| directory | Nepali h | why not uploaded |
|---|---|---|
| `podcast-index-feeds` | 115.7 | **no transcripts** — audio only |
| `gemini_vc` | 11.9 | same audio as `r2`, voice-converted; `original_audio_filepath` points into `ncert_actor_critic_education_tts` |
| `gemini_vc_conversational{,_corrected,_pairs}` | 130.5 | **no audio on this box** — manifests point at `/projects/data/ttsteam/{nikhil,avnish}/`, and only `sunil` is still mounted |
| `indicvoices-r`, `ai4bharat___rasa`, `Rasa_v2`, `IndicVoices`, `indic_voices_long` | — | already public at source as AI4Bharat releases |

## Dedupe rules the packager applies

1. **Against the first archive**: any row whose audio path appears in
   `release/id_map.jsonl` is skipped — that removes 30,042 `indicvoices-r-long` files
   (74.0 h) which are already in `nepali-speech-archive`.
2. **`r5` before `r6`**: the audiobook rows carry the same chunk `id`s as the 5-min set
   (verified: `145428_000009_9bec40f51c97f069_0000` is in both, and all five sampled
   audiobook feeds appear in `r6`'s manifest). Whichever lands first wins, so the
   363.2 h is stored once.
3. **No transcript, no row** — enforced in `row_of()`, not just by directory choice.

## Uploaded columns

`id` (16 hex of `sha256(path#offset)`), `audio` (FLAC), `text`, `duration`, `spk`
(12 hex of `sha256(dir|speaker label)`), `src`, `sr`, `ch`, `nspk`, `lang_sec`,
`segments`, `dnsmos`, `snr_db`.

Dropped before upload: every path, URL, `feed_id`, `feed_title`, `episode_title`,
`episode_guid`, `author`, `host`, `itunes_id`, `podcast_guid`, `sample_id` and raw
diarization label. A leak audit over the first packaged shard found no path fragment,
feed id, channel id or URL.
