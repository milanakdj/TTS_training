#!/bin/bash
# Nepali hours per corpus directory. Single awk pass per manifest: 114.8 GB of JSON
# is far too slow to json.loads, and only two fields are needed.
# Counts a row as Nepali if language is ne/nep/nepali OR the audio path says Nepali
# (the AI4Bharat trees are language-partitioned by directory, not by a field).
cd /workspace/proc_data_new
for d in */; do
  for f in "${d}train/manifest.jsonl" "${d}manifest_train.jsonl" "${d}manifest.jsonl"; do
    [ -f "$f" ] || continue
    LC_ALL=C awk -v dir="${d%/}" -v file="$f" '
      /"language": *"(ne|nep|nepali|Nepali)"/ || /\/Nepali\// {
        if (match($0, /"duration": *[0-9.]+/)) {
          s = substr($0, RSTART, RLENGTH); sub(/.*: */, "", s); t += s; n++
        }
      }
      END { if (n > 0) printf "%s\t%d\t%.1f\n", dir, n, t/3600 }
    ' "$f"
    break   # one canonical manifest per directory
  done
done
