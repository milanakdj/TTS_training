#!/bin/bash
# Fetch all 1000 parquet shards of Premal-12/c9nepali-audio-dataset2 into stage/.
# Token is read from /tmp/hf_tok (never inlined) so this script stays shareable.
T=$(cat /tmp/hf_tok)
BASE="https://huggingface.co/datasets/Premal-12/c9nepali-audio-dataset2/resolve/main/data"
STAGE=/workspace/oov_distill/stage
get() {
  s=$(printf "%04d" $1)
  d="$STAGE/shard_$s"
  # already staged and non-trivial in size -> skip (makes the script resumable)
  if [ -s "$d/t.parquet" ] && [ $(stat -c%s "$d/t.parquet") -gt 1000000 ]; then return; fi
  mkdir -p "$d"
  curl -sf -H "Authorization: Bearer $T" -L \
    "$BASE/shard_$s/train-00000-of-00001.parquet" -o "$d/t.parquet" \
    || echo "FAIL $s" >> /workspace/oov_distill/logs/fetch_fail.log
}
export -f get; export STAGE BASE T
seq 0 999 | xargs -P 12 -I{} bash -c 'get {}'
echo "done: $(ls -d $STAGE/shard_* 2>/dev/null | wc -l) shards, $(du -sh $STAGE 2>/dev/null | cut -f1)"
