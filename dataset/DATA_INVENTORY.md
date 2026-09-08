# Nepali data inventory

How many hours of Nepali exist on this box, which directory each lives in, how much
of it actually reached training, and how many GB that is.

Measured 2026-09-08. Three separate numbers get confused constantly, so they are kept
apart here:

| | hours | what it is |
|---|---|---|
| **on disk** | see [inventory](#corpus-wide-inventory) | everything labelled Nepali under `/workspace/proc_data_new`, before de-duplication |
| **used for training** | **2,215.7** | what survived quality gating and went into both model runs |
| **archived to HF** | **1,993.1** | the training set minus what is already public on the Hub |

---

## Used for training

`manifests/train_v2_aligned.jsonl` — 749,610 clips, 100% word-aligned, audio
referenced in place. This is the authoritative number for anything about the models.

| source | clips | hours | ≈GB WAV | in the HF archive? |
|---|---|---|---|---|
| `ans_snr40-50` | 403,364 | 1,230.2 | 213 | yes |
| `ans_snr50` | 193,750 | 600.7 | 104 | yes |
| `indicvoices-r` | 64,029 | 168.1 | 29 | **no** — already public |
| `mahadhwani` | 26,670 | 80.8 | 14 | yes |
| `indicvoices-r-long` | 30,042 | 74.0 | 13 | yes |
| `rasa` | 29,377 | 54.5 | 9 | **no** — already public |
| `orpheus_snr50` | 2,354 | 7.4 | 1 | yes |
| `podcast-index` | 24 | ~0.0 | ~0 | yes |
| **total** | **749,610** | **2,215.7** | **383** | 1,993.1 h archived |

Size basis: measured **48,004 bytes/sec** across 3,000 sampled files — 24 kHz,
16-bit, mono, so **172.8 MB per hour**. FLAC measured at **50.3%** of WAV on 40 real
clips, so the whole training set is 383 GB as WAV or 193 GB as FLAC.

### Provenance, and what may be redistributed

| portion | hours | share | status |
|---|---|---|---|
| YouTube-derived (`ans_*`, `mahadhwani`, `orpheus_*`) | 1,919.1 | 86.6% | **not redistributable** |
| AI4Bharat CC-BY-4.0 (`indicvoices-r*`, `rasa`) | 296.6 | 13.4% | already public at source |

`mahadhwani` is YouTube too despite the corpus name — its
`original_audio_filepath` points at `yt-rechunker/chunks/yt-md/<videoId>_...`, and no
public Mahadhwani *dataset* exists on the Hub (only a pretrained conformer model).

---

## Archived to Hugging Face

`milanakdj/nepali-speech-archive` — **backup only, not a release.** Gated (manual),
FLAC, 180 shards of ~1 GB.

**1,993.1 h / 656,204 clips / ~173 GB.** Excludes `rasa` (49–54 h) and
`indicvoices-r` (168.1 h), which are already safely on the Hub as AI4Bharat releases
and would be wasted storage. `indicvoices-r-long` is **not** among those releases and
is kept.

Nothing on local disk is renamed or modified — the WAVs are read in place and the
renaming happens only inside the uploaded shards. Because a gated repo still exposes
its **file list and column schema** to anyone (verified against the live
`nepali-tts-synthetic-v2`), the uploaded columns are deliberately opaque:

| column | contents |
|---|---|
| `id` | first 16 hex of `sha256(original path)` |
| `audio` | FLAC bytes, named `<id>.flac` |
| `text` | transcript |
| `duration` | seconds |
| `spk` | `sha256(directory + diarization label)`, 12 hex — groups speakers without naming them |
| `src` | `s1`…`s6`, opaque; the mapping stays local |
| `words` | alignment, JSON |

Dropped entirely: original paths, `original_audio_filepath`, real source names, raw
diarization labels. A leak audit over 200 packaged rows found no path fragment,
channel id, video id or source name.

**The `id` → path mapping lives only at `/workspace/hf_release/id_map.jsonl`** and is
not uploaded. Without it the archive still restores audio and text, but not original
filenames — so that file needs its own backup, ideally a small private repo.

Rebuild or resume with `scripts/backup_corpus.py` (running copy lives at `/workspace/hf_release/`) (idempotent; skips
shards already in the repo).

---

## Corpus-wide inventory

Nepali hours per directory under `/workspace/proc_data_new`, counting a row as Nepali
when `language` is `ne`/`nep`/`nepali` **or** the audio path is language-partitioned
under `/Nepali/`. Produced by `scripts/nepali_inventory.sh` + `scripts/nepali_inventory2.sh` — a single `awk` pass per manifest, because the canonical
manifests total 114.8 GB of JSON and `json.loads` is far too slow at that size.

> **Status: scan was still running when this was written** — 22 of 65 canonical
> manifests plus 2 of the 4 sharded directories. Figures below are final for the
> directories listed; more directories may appear. Re-run the two scripts to refresh.

| directory | clips | hours |
|---|---|---|
| `podcast-index-feeds-5min` | 86,682 | 6,680.8 |
| `indic_voices_long` | 142,308 | 1,887.4 |
| `indicvoices-r-long` | 47,436 | 629.1 |
| `ans_snr40-50.part2of2` | 202,031 | 616.6 |
| `ans_snr40-50.part1of2` | 202,018 | 615.5 |
| `IndicVoices` | 226,055 | 419.3 |
| `podcast-index-audiobook` | 4,641 | 363.2 |
| `ans_snr50.part1of2` | 97,053 | 301.2 |
| `ans_snr50.part2of2` | 97,054 | 300.6 |
| `indicvoices-r` | 64,037 | 168.1 |
| `ncert_conversations` | 2,310 | 148.6 |
| `gemini_vc_conversational_pairs` | 1,952 | 130.5 |
| `gemini_vc_conversational_corrected` | 1,952 | 130.5 |
| `gemini_vc_conversational` | 1,810 | 120.1 |
| `podcast-index-feeds` | 285 | 115.7 |
| `mahadhwani` | 32,072 | 94.1 |
| `podcast-index` | 24,327 | 54.0 |
| `ai4bharat___rasa` | 28,113 | 49.5 |
| `Rasa_v2` | 26,824 | 49.0 |
| `orpheus_snr40-50` | 5,062 | 16.2 |
| `gemini_vc` | 182 | 11.9 |
| `orpheus_snr50` | 2,357 | 7.4 |

### Do not sum that column

Three traps, each of which inflates a naive total by hundreds or thousands of hours:

1. **The same corpus appears in several processing states.**
   `indic_voices_long` → `IndicVoices` → `indicvoices-r` are one corpus restored and
   re-chunked, so that is ~1,887 h of *unique* audio, not 1,887 + 419 + 168.
   `Rasa_v2` and `ai4bharat___rasa` are likewise ~49 h once, not twice.
2. **`gemini_vc_conversational{,_corrected,_pairs}` are three variants of the same
   ~130 h.** Count one.
3. **`podcast-index-feeds-5min`'s 6,680.8 h is not 6,680 h of Nepali speech.** A row
   with `language: "ne"` means Nepali is the *dominant* language of that 5-minute
   chunk; the rows carry `languages: ['en','ne']` and a `language_seconds` breakdown,
   and the sampled row was 229.67 s Nepali against 70.96 s English. For a true
   figure, sum `language_seconds.ne` rather than `duration`. These are also 48 kHz
   two-speaker diarized conversation chunks, so they are ASR/pretraining material
   rather than TTS-ready data.

The earlier working estimate of "~8.5k h across 30 of 80 directories" is consistent
with this table once the duplicates above are collapsed — but the honest unique
Nepali total needs `language_seconds` summed for the podcast sources, which has not
been done.

### Why only 2,215.7 h of it was used

Quality gating, not availability. Clips were kept only where three independent ASRs
(saaras / canary / conformer) agreed; `spk_overlap_percent` removed multi-talker
clips, which a TTS must never learn; DNSMOS/SNR/C50 thresholds removed residual
noise; and duration was capped at 30 s. `indicvoices-r-long` shows the effect most
plainly — 629.1 h on disk, 74.0 h after gating.

See [`ADR.md`](../pocket_TTS/docs/ADR.md#adr-007) for the gating decision and
[ADR-013](../pocket_TTS/docs/ADR.md#adr-013) for the open question about the AI4Bharat sources not
being Nepal-native speech.
