"""
Build the ~500k+ sentence text pool for the production synthesis run.

Primary source: himalaya-ai/nepali-proofreader "clean" column, filtered to
>4 words OR >50 chars, deduped. (~479k unique per initial check.)

Topped up with sentence-split himalaya-ai/nepali-news-corpus articles (same
filter) to give headroom above the 500k target, since QC will reject a
fraction of synthesized rows -- more candidate text than the final row target.

Output: manifests/text_pool.parquet with columns [sentence_id, text, source]
"""
import re
import sys

import pandas as pd
from huggingface_hub import hf_hub_download

TARGET_POOL_SIZE = 900_000  # headroom above 500k final target, given ~70% QC pass rate
OUT_PATH = "/root/tts/TTS_training/synthetic_pipeline/manifests/text_pool.parquet"

SENTENCE_SPLIT_RE = re.compile(r"(?<=[।?!])\s+|\n+")


def passes_filter(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    return len(s.split()) > 4 or len(s) > 50


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def main():
    print("[1/3] loading nepali-proofreader 'clean' column...", file=sys.stderr)
    p = hf_hub_download("himalaya-ai/nepali-proofreader", "data/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p, columns=["clean"])
    clean = df["clean"].dropna().astype(str).map(clean_text)
    mask = clean.map(passes_filter)
    proofreader_sents = clean[mask].drop_duplicates()
    print(f"  proofreader: {len(proofreader_sents)} unique sentences after filter+dedup", file=sys.stderr)

    pool = pd.DataFrame({"text": proofreader_sents, "source": "nepali-proofreader"})
    seen = set(pool["text"])

    needed = TARGET_POOL_SIZE - len(pool)
    if needed > 0:
        print(f"[2/3] need {needed} more, pulling himalaya-ai/nepali-news-corpus...", file=sys.stderr)
        extra_rows = []
        shard = 1
        while needed > 0 and shard <= 9:
            fname = f"data/train-{shard:04d}-of-0009.parquet"
            try:
                p2 = hf_hub_download("himalaya-ai/nepali-news-corpus", fname, repo_type="dataset")
            except Exception as e:
                print(f"  shard {fname} unavailable ({e}), stopping", file=sys.stderr)
                break
            news_df = pd.read_parquet(p2, columns=["text"])
            print(f"  shard {shard}: {len(news_df)} articles", file=sys.stderr)
            for article in news_df["text"].dropna().astype(str):
                for sent in SENTENCE_SPLIT_RE.split(article):
                    sent = clean_text(sent)
                    if sent in seen or not passes_filter(sent):
                        continue
                    seen.add(sent)
                    extra_rows.append(sent)
                    needed -= 1
                    if needed <= 0:
                        break
                if needed <= 0:
                    break
            shard += 1
        print(f"  news-corpus contributed: {len(extra_rows)} unique sentences", file=sys.stderr)
        pool = pd.concat(
            [pool, pd.DataFrame({"text": extra_rows, "source": "nepali-news-corpus"})],
            ignore_index=True,
        )

    pool = pool.reset_index(drop=True)
    pool["sentence_id"] = pool.index
    pool = pool[["sentence_id", "text", "source"]]
    pool.to_parquet(OUT_PATH)
    print(f"[3/3] wrote {len(pool)} rows to {OUT_PATH}", file=sys.stderr)
    print(pool["source"].value_counts())


if __name__ == "__main__":
    main()
