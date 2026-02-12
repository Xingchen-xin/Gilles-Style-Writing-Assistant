#!/usr/bin/env python3
"""
Clean training data v2 - Aggressive cleaning for style learning.

This script addresses the root causes of LoRA overfitting:
1. Removes ALL citation patterns (not just PDF artifacts)
2. Removes figure/table references
3. Filters high-similarity samples
4. Adds more diverse instructions

Usage:
    python scripts/clean_training_data_v2.py --input data/training/alpaca_train.jsonl
"""
import argparse
import json
import re
import random
from pathlib import Path
from difflib import SequenceMatcher


# ============================================================
# Patterns to REMOVE from text (aggressive cleaning)
# ============================================================

REMOVE_PATTERNS = [
    # === Citations ===
    # Parenthetical citations: (1), (1,2), (1-5), (1, 2, 3)
    r'\s*\(\d+(?:[,–-]\s*\d+)*\)',
    # Square bracket citations: [1], [1,2], [1-5]
    r'\s*\[\d+(?:[,–-]\s*\d+)*\]',
    # Author et al. citations
    r'\([A-Z][a-z]+\s+et\s+al\.\s*,?\s*\d{4}\)',
    r'[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)',

    # === Figure/Table references ===
    r'(?:see\s+)?(?:Fig(?:ure)?|Table|Supplementary\s+(?:Fig(?:ure)?|Table|Material))\s*\.?\s*S?\d+[A-Za-z]?(?:\s*(?:and|,)\s*S?\d+[A-Za-z]?)*',
    r'\((?:Fig(?:ure)?|Table)\s*\.?\s*S?\d+[A-Za-z]?\)',

    # === Journal/Reference formats ===
    # Volume, page format: 2021, 49, 5349-5361
    r'\d{4},\s*\d+(?:\(\d+\))?,\s*\d+[-–]\d+',
    # DOI
    r'(?:doi:\s*)?10\.\d{4,}/[^\s]+',
    # URLs
    r'https?://[^\s]+',
    r'www\.[^\s]+',

    # === PDF artifacts ===
    r'This journal is © [^.]+\.',
    r'Downloaded by [^.]+\.',
    r'Published on [^.]+\.',
    r'View Article Online',

    # === Page numbers and sequences ===
    r'\b\d{3}\s+\d{3}\s+\d{3}\b',  # 123 124 125
    r'(?:^|\s)p+\.\s*\d+[-–]?\d*',  # p. 123 or pp. 123-456

    # === Supplemental material references ===
    r'(?:see\s+)?(?:the\s+)?[Ss]upplemental\s+[Mm]aterial[s]?',
    r'(?:in\s+)?[Ss]upporting\s+[Ii]nformation',
    r'\(data\s+not\s+shown\)',
]

# Compile patterns
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in REMOVE_PATTERNS]


# ============================================================
# Diverse instruction templates
# ============================================================

INSTRUCTION_TEMPLATES = [
    # Style improvement
    "Rewrite this scientific text with improved academic style:",
    "Enhance the clarity and precision of this scientific paragraph:",
    "Polish this academic text for publication quality:",
    "Refine the writing style of this scientific content:",
    "Improve the academic tone of this research paragraph:",

    # Paraphrasing
    "Paraphrase this scientific text in formal academic English:",
    "Express these scientific ideas more elegantly:",
    "Restate this research finding with better academic prose:",

    # Editing
    "Edit this scientific paragraph for improved readability:",
    "Revise this academic text with clearer expression:",
    "Restructure this scientific writing for better flow:",

    # Specific improvements
    "Make this scientific text more concise and precise:",
    "Strengthen the academic argumentation in this text:",
    "Clarify the scientific reasoning in this paragraph:",
    "Improve the transitions and coherence in this text:",
]


def clean_text(text: str) -> str:
    """Remove all citation and reference patterns from text."""
    cleaned = text

    for pattern in COMPILED_PATTERNS:
        cleaned = pattern.sub('', cleaned)

    # Clean up artifacts from removal
    # Multiple spaces
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    # Space before punctuation
    cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned)
    # Multiple punctuation
    cleaned = re.sub(r'([.,])\s*\1+', r'\1', cleaned)
    # Empty parentheses
    cleaned = re.sub(r'\(\s*\)', '', cleaned)
    cleaned = re.sub(r'\[\s*\]', '', cleaned)
    # Trailing/leading whitespace
    cleaned = cleaned.strip()

    return cleaned


def has_remaining_artifacts(text: str) -> bool:
    """Check if text still has problematic patterns after cleaning."""
    # Check for remaining citation patterns
    if re.search(r'\(\d+\)', text):  # (1)
        return True
    if re.search(r'\[\d+\]', text):  # [1]
        return True
    if re.search(r'et\s+al\.', text, re.IGNORECASE):  # et al.
        return True
    if re.search(r'Fig\.\s*S?\d+', text, re.IGNORECASE):  # Fig. 1
        return True
    if re.search(r'\d{4},\s*\d+,\s*\d+', text):  # journal format
        return True
    return False


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    return SequenceMatcher(None, text1, text2).ratio()


def process_sample(sample: dict, max_length: int = 3000,
                   max_similarity: float = 0.75) -> dict | None:
    """Process a single training sample with aggressive cleaning.

    Returns None if sample should be excluded.
    """
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    out = sample.get("output", "")

    # Clean both input and output
    cleaned_input = clean_text(inp)
    cleaned_output = clean_text(out)

    # Skip if too short after cleaning
    if len(cleaned_input) < 100 or len(cleaned_output) < 100:
        return None

    # Skip if still has artifacts
    if has_remaining_artifacts(cleaned_output):
        return None

    # Skip if too similar (model might learn to just copy)
    similarity = calculate_similarity(cleaned_input, cleaned_output)
    if similarity > max_similarity:
        return None

    # Skip if too long
    total_len = len(cleaned_input) + len(cleaned_output)
    if total_len > max_length:
        return None

    # Assign diverse instruction
    new_instruction = random.choice(INSTRUCTION_TEMPLATES)

    return {
        "instruction": new_instruction,
        "input": cleaned_input,
        "output": cleaned_output,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean training data v2")
    parser.add_argument("--input", default="data/training/alpaca_train.jsonl",
                        help="Input training file")
    parser.add_argument("--output", default="data/training/clean_v2.jsonl",
                        help="Output cleaned file")
    parser.add_argument("--max-length", type=int, default=3000,
                        help="Max total chars (input+output)")
    parser.add_argument("--max-similarity", type=float, default=0.75,
                        help="Max input-output similarity (0-1)")
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
    print(f"Max length: {args.max_length}")
    print(f"Max similarity: {args.max_similarity}")
    print()

    # Process samples
    random.seed(42)
    cleaned = []
    stats = {
        'too_short': 0,
        'has_artifacts': 0,
        'too_similar': 0,
        'too_long': 0,
        'kept': 0,
    }

    for sample in samples:
        result = process_sample(
            sample,
            max_length=args.max_length,
            max_similarity=args.max_similarity
        )

        if result is None:
            # Determine why it was rejected
            inp = sample.get("input", "")
            out = sample.get("output", "")
            cleaned_inp = clean_text(inp)
            cleaned_out = clean_text(out)

            if len(cleaned_inp) < 100 or len(cleaned_out) < 100:
                stats['too_short'] += 1
            elif has_remaining_artifacts(cleaned_out):
                stats['has_artifacts'] += 1
            elif calculate_similarity(cleaned_inp, cleaned_out) > args.max_similarity:
                stats['too_similar'] += 1
            else:
                stats['too_long'] += 1
        else:
            cleaned.append(result)
            stats['kept'] += 1

    # Stats
    print("=== Cleaning Results (v2 - Aggressive) ===")
    print(f"Kept: {stats['kept']} samples ({100*stats['kept']/len(samples):.1f}%)")
    print(f"Excluded breakdown:")
    print(f"  - Too short after cleaning: {stats['too_short']}")
    print(f"  - Still has artifacts: {stats['has_artifacts']}")
    print(f"  - Too similar (in/out): {stats['too_similar']}")
    print(f"  - Too long: {stats['too_long']}")

    if cleaned:
        avg_len = sum(len(s["input"]) + len(s["output"]) for s in cleaned) / len(cleaned)
        print(f"\nKept samples stats:")
        print(f"  Average length: {avg_len:.0f} chars")

        # Check instruction diversity
        instructions = set(s["instruction"] for s in cleaned)
        print(f"  Unique instructions: {len(instructions)}")

        # Verify no artifacts in output
        artifact_check = sum(1 for s in cleaned if has_remaining_artifacts(s["output"]))
        print(f"  Samples with remaining artifacts: {artifact_check}")

    # Save
    if not args.dry_run and cleaned:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Shuffle for training
        random.shuffle(cleaned)

        # Split 90/10
        split_idx = int(len(cleaned) * 0.9)
        train_data = cleaned[:split_idx]
        val_data = cleaned[split_idx:]

        # Save train
        train_path = output_path.parent / "clean_v2_train.jsonl"
        with open(train_path, "w") as f:
            for sample in train_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Save val
        val_path = output_path.parent / "clean_v2_val.jsonl"
        with open(val_path, "w") as f:
            for sample in val_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"\nSaved to:")
        print(f"  Train: {train_path} ({len(train_data)} samples)")
        print(f"  Val: {val_path} ({len(val_data)} samples)")
    elif args.dry_run:
        print("\n(Dry run - no file written)")
    else:
        print("\nNo samples passed filtering - check parameters")

    return 0


if __name__ == "__main__":
    exit(main())
