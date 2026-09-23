"""Split train_v3 by language and reuse v2's alignments where they are still valid.

Two things make this necessary rather than just re-running align_v2.sh:

  1. align_data keys --resume on (path, start), NOT on the transcript. 1.8% of
     Nepali transcripts changed under ne_frontend.normalize() (glosses dropped,
     numbers verbalized, unencodable punctuation rewritten), and their v2 word
     timings no longer describe the text they are paired with. Seeding those rows
     would make the aligner skip exactly the rows that need redoing.
  2. The Nepali aligner (wav2vec2-xlsr-nepali) cannot align English -- align_data
     raises on a transcript whose characters are outside the model's alphabet. The
     two languages need separate passes with separate models.

Writes:
    train_v3_ne.jsonl / train_v3_en.jsonl     inputs for the two aligner passes
    train_v3_aligned.jsonl                    pre-seeded with the reusable v2 rows
"""
import json, os, sys

M = "/root/tts/TTS_training/pocket_TTS/manifests"


def load_v2():
    """path -> (transcript, words) from every aligned v2 manifest."""
    out = {}
    for name in ("train_v2_aligned.jsonl", "valid_v2_aligned.jsonl"):
        p = os.path.join(M, name)
        if not os.path.exists(p):
            continue
        for line in open(p):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("words"):
                out[d["path"]] = (d["transcript"], d["words"])
    return out


def main(split="train_v3"):
    v2 = load_v2()
    print(f"v2 alignments available: {len(v2)}")
    seeded = ne_todo = en_todo = 0
    fh_seed = open(f"{M}/{split}_aligned.jsonl", "w")
    fh_ne = open(f"{M}/{split}_ne.jsonl", "w")
    fh_en = open(f"{M}/{split}_en.jsonl", "w")
    for line in open(f"{M}/{split}.jsonl"):
        d = json.loads(line)
        if d.get("language") == "en":
            fh_en.write(line); en_todo += 1
            continue
        hit = v2.get(d["path"])
        # Reuse only when the TEXT is identical too -- word timings describe a
        # specific transcript, and normalize() changed 1.8% of them.
        if hit and hit[0] == d["transcript"]:
            fh_seed.write(json.dumps({**d, "words": hit[1]}, ensure_ascii=False) + "\n")
            seeded += 1
        else:
            fh_ne.write(line); ne_todo += 1
    for fh in (fh_seed, fh_ne, fh_en):
        fh.close()
    print(f"{split}: seeded {seeded} | nepali to align {ne_todo} | english to align {en_todo}")


if __name__ == "__main__":
    for s in sys.argv[1:] or ["train_v3", "valid_v3"]:
        main(s)
