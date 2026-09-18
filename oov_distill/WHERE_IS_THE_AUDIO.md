# oov_distill: what moved here, what stayed on /workspace

Moved to the repo on 2026-09-18: the code, the logs, and the small report json --
everything the 854 h / 544.3 h distinct funnel numbers actually rest on.

Four large directories deliberately stayed at `/workspace/oov_distill/` (246 GB;
`/` only had 83 GB free). None of them is unique -- each is recoverable:

| dir | size | how to get it back |
| --- | --- | --- |
| `stage/` | 118 G | `fetch.sh` re-pulls all 1000 parquet shards of `Premal-12/c9nepali-audio-dataset2` (resumable; skips what is already there) |
| `out/hf/` | 51 G | all 213 shards are live at `milanakdj/nepali-oov-distilled` -- verified 2026-09-18, shard_0000..shard_0212 |
| `out/vc_all/` | 44 G | the converted audio ships inside hf shards 0109-0212; raw wavs need a Seed-VC rerun (`seedvc_convert.py`) |
| `out/vc_src/` | 33 G | `select.py` re-extracts it from `stage/` |

The scripts here still carry their original `/workspace/oov_distill/...` paths, so
they keep working against those directories unchanged. Point `VC_DIR` or edit the
module-level constants if you ever relocate them.
