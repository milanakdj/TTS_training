import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl) if args.output_jsonl else input_path.with_name(
        input_path.stem + "_filtered.jsonl"
    )

    rows = []
    codebook_counts = Counter()
    length_counts = Counter()
    invalid = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            codes = row.get("audio_codes")
            if (
                not isinstance(codes, list)
                or not codes
                or not isinstance(codes[0], list)
                or not codes[0]
            ):
                invalid += 1
                continue

            codebook_count = len(codes)
            token_length = len(codes[0])
            codebook_counts[codebook_count] += 1
            length_counts[token_length] += 1
            rows.append((row, codebook_count, token_length))

    if not rows:
        raise RuntimeError("No valid codec rows found.")

    expected_codebooks = codebook_counts.most_common(1)[0][0]
    kept = [row for row, codebooks, _ in rows if codebooks == expected_codebooks]

    with output_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = [length for _, codebooks, length in rows if codebooks == expected_codebooks]

    print(f"Input rows: {len(rows) + invalid}")
    print(f"Valid rows: {len(rows)}")
    print(f"Invalid rows: {invalid}")
    print(f"Codebook distribution: {dict(codebook_counts)}")
    print(f"Expected codebooks: {expected_codebooks}")
    print(f"Kept rows: {len(kept)}")
    print(f"Token length min/max: {min(lengths)} / {max(lengths)}")
    print(f"Filtered JSONL: {output_path}")


if __name__ == "__main__":
    main()

