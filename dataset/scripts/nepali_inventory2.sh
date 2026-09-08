#!/bin/bash
# Complement to nepali_inventory.sh: the four directories that keep SHARDED or
# differently-named manifests, so the one-canonical-file loop skips them entirely.
cd /workspace/proc_data_new
for d in youtube_data indic_voices_long indicvoices-r-long ncert_actor_critic_education_tts; do
  ls "$d"/train/*.jsonl "$d"/*/manifest*.jsonl 2>/dev/null | sort -u | \
  LC_ALL=C xargs -r awk -v dir="$d" '
    /"language": *"(ne|nep|nepali|Nepali)"/ || /\/Nepali\// {
      if (match($0, /"duration": *[0-9.]+/)) {
        s = substr($0, RSTART, RLENGTH); sub(/.*: */, "", s); t += s; n++
      }
    }
    END { if (n > 0) printf "%s\t%d\t%.1f\n", dir, n, t/3600 }
  '
done
