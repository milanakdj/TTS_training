# dataset

Everything about the Nepali speech **data** — how much exists, where it lives, what
may be redistributed, and how it was packaged for backup. Model documentation lives
with the model, in [`../pocket_TTS/`](../pocket_TTS/).

| file | what it answers |
|---|---|
| [`DATA_INVENTORY.md`](DATA_INVENTORY.md) | Nepali hours per corpus directory, hours that reached training, GB figures, provenance and redistribution status |
| [`scripts/nepali_inventory.sh`](scripts/nepali_inventory.sh) | counts Nepali hours per directory — one `awk` pass per manifest |
| [`scripts/nepali_inventory2.sh`](scripts/nepali_inventory2.sh) | the same for the four directories that keep sharded or differently-named manifests |
| [`scripts/backup_corpus.py`](scripts/backup_corpus.py) | packages the corpus to a gated HF dataset repo as FLAC, with provenance stripped |

## The three numbers, kept apart

They get conflated constantly:

| | hours | |
|---|---|---|
| on disk | see inventory | everything labelled Nepali, **before** de-duplication |
| **used for training** | **2,215.7** | survived quality gating; the number to quote about the models |
| **archived to HF** | **1,993.1** | training set minus what is already public on the Hub |

## Two things to read before quoting any figure

**Do not sum the per-directory column.** The same corpus appears in several
processing states — `indic_voices_long` → `IndicVoices` → `indicvoices-r` are one
corpus, not three — and `podcast-index-feeds-5min`'s hours are chunk durations for
code-mixed audio, not hours of Nepali speech. `DATA_INVENTORY.md` sets out all three
traps.

**86.6% of the training corpus is YouTube-derived and is not redistributable.** The
remaining 13.4% is AI4Bharat CC-BY-4.0 and is already public at source. The HF
archive is a backup, not a release.
