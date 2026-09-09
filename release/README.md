# release

Tooling that staged and pushed the Nepali models to the Hub. Moved here from
`/workspace/hf_release` so it lives with the code rather than on a scratch disk.

| file | what it does |
|---|---|
| [`stage_release.py`](stage_release.py) | stages the pocket-TTS student: copies weights, rewrites `config.yaml` and the card so every path is an `hf://` self-reference, writes `eval_results.json`. `--brief` summarises the training-data section instead of listing per-source hours |
| [`stage_teacher.py`](stage_teacher.py) | same for the 24-layer teacher, with a card that opens "do not use this model for inference" |
| [`push_to_hf.py`](push_to_hf.py) | `check` \| `model` \| `parler`. Reads the token from `/root/.hf_milanakdj`; never prints it |
| [`templates/`](templates/) | card and inference-example templates with a `__REPO_ID__` placeholder |

**Repo id is a parameter, not a constant.** `config.yaml`, the card and the example
all reference the repo by name through `hf://` paths, so changing the destination
means regenerating all three rather than editing them by hand:

```bash
python3 stage_release.py milanakdj/pocket-tts-nepali-6l            # detailed card
python3 stage_release.py himalaya-ai/pocket-tts-nepali-6l --brief  # summarised card
python3 push_to_hf.py check
```

## Two things that are not obvious

**Gating is not card metadata.** `extra_gated_*` in the card YAML does *not* turn
gating on, and neither does a `gated:` key. Only
`HfApi.update_repo_settings(..., gated="manual")` does, which is why
`push_to_hf.py` calls it explicitly after upload. Reading it back needs
`?expand[]=gated`, or the API reports `gated: false` regardless.

**Never use the ambient `HF_TOKEN`.** It belongs to a different account that other
processes on this box depend on. The milanakdj token is at `/root/.hf_milanakdj`.

## What is here but not tracked

Gitignored, and deliberately so:

- `staged/`, `staged_brief/`, `staged_teacher/`, `staged_parler/` — byte-identical
  copies of what is already on the Hub
- `id_map.jsonl` — 106 MB mapping the speech archive's hashed ids back to original
  audio paths. The most provenance-sensitive file in this repo, and regenerable
  (`sha256` of each manifest path), so it is kept out of git on purpose
- `corpus_archive_upload.log` — 17 MB of progress bars from the 156 GB upload

`nepali_inventory*.tsv` are the raw per-directory Nepali hour counts behind
[`../dataset/DATA_INVENTORY.md`](../dataset/DATA_INVENTORY.md); they are small and
tracked.
