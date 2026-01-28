#!/usr/bin/env python3
"""
Clean training data by removing PDF artifacts and other contamination.

This script:
1. Removes copyright notices, watermarks, download metadata
2. Filters out samples that are too long
3. Optionally limits sample length for better style learning

Usage:
    python scripts/clean_training_data.py
    python scripts/clean_training_data.py --max-length 3000 --output data/training/clean_train.jsonl
"""
import argparse
import json
import re
from pathlib import Path


# PDF artifacts to remove
PDF_ARTIFACTS = [
    # Journal copyright notices
    r"This journal is © The Royal Society of Chemistry \d{4}",
    r"© The Royal Society of Chemistry \d{4}",
    r"Nat\.? Prod\.? Rep\.?",
    r"Natural Product Reports",
    r"This article is licensed under a Creative Commons",

    # Download watermarks
    r"Downloaded by [A-Za-z\s/]+ on \d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M\.?",
    r"Downloaded by Universiteit Leiden[^.]*\.",
    r"Downloaded by University of[^.]*\.",
    r"Published on \d{1,2} [A-Za-z]+ \d{4}\.",
    r"View Article Online",

    # Page artifacts
    r"This journal is \u00a9",  # © symbol
    r"DOI: 10\.\d{4,5}/[^\s]+",
    r"www\.rsc\.org/[^\s]+",
]

# Compile patterns
ARTIFACT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PDF_ARTIFACTS]


def clean_text(text: str) -> str:
    """Remove PDF artifacts from text."""
    cleaned = text
    for pattern in ARTIFACT_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # Clean up multiple spaces and newlines
    cleaned = re.sub(r'\s{3,}', '  ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


def is_heavily_contaminated(text: str) -> bool:
    """Check if text has too many artifacts (should be excluded)."""
    artifact_count = 0
    for pattern in ARTIFACT_PATTERNS:
        artifact_count += len(pattern.findall(text))

    # More than 3 artifacts = too contaminated
    return artifact_count > 3


def process_sample(sample: dict, max_length: int = 0) -> dict | None:
    """Process a single training sample.

    Returns None if sample should be excluded.
    """
    inp = sample.get("input", "")
    out = sample.get("output", "")

    # Skip heavily contaminated samples
    if is_heavily_contaminated(inp) or is_heavily_contaminated(out):
        return None

    # Clean both input and output
    cleaned_input = clean_text(inp)
    cleaned_output = clean_text(out)

    # Skip if cleaning removed too much
    if len(cleaned_input) < 50 or len(cleaned_output) < 50:
        return None

    # Skip if too long (optional)
    if max_length > 0:
        total_len = len(cleaned_input) + len(cleaned_output)
        if total_len > max_length:
            return None

    return {
        "instruction": sample.get("instruction", ""),
        "input": cleaned_input,
        "output": cleaned_output,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean training data")
    parser.add_argument("--input", default="data/training/alpaca_train.jsonl",
                        help="Input training file")
    parser.add_argument("--output", default="data/training/clean_train.jsonl",
                        help="Output cleaned file")
    parser.add_argument("--max-length", type=int, default=4000,
                        help="Max total chars (input+output), 0 for no limit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show stats, don't write")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Load data
    with open(input_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Input: {input_path}")
    print(f"Total samples: {len(samples)}")
    print(f"Max length: {args.max_length or 'no limit'}")
    print()

    # Process samples
    cleaned = []
    excluded_contaminated = 0
    excluded_too_long = 0
    excluded_too_short = 0

    for sample in samples:
        inp = sample.get("input", "")
        out = sample.get("output", "")

        if is_heavily_contaminated(inp) or is_heavily_contaminated(out):
            excluded_contaminated += 1
            continue

        result = process_sample(sample, args.max_length)
        if result is None:
            if args.max_length and len(inp) + len(out) > args.max_length:
                excluded_too_long += 1
            else:
                excluded_too_short += 1
            continue

        cleaned.append(result)

    # Stats
    print("=== Cleaning Results ===")
    print(f"Kept: {len(cleaned)} samples ({100*len(cleaned)/len(samples):.1f}%)")
    print(f"Excluded (contaminated): {excluded_contaminated}")
    print(f"Excluded (too long): {excluded_too_long}")
    print(f"Excluded (too short): {excluded_too_short}")

    if cleaned:
        avg_len = sum(len(s["input"]) + len(s["output"]) for s in cleaned) / len(cleaned)
        print(f"Average sample length: {avg_len:.0f} chars")

    # Save
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in cleaned:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"\nSaved to: {output_path}")
    else:
        print("\n(Dry run - no file written)")

    return 0


if __name__ == "__main__":
    exit(main())
