#!/usr/bin/env python3
"""
Export feedback data for DPO (Direct Preference Optimization) training.

This script converts collected user feedback into the format required
for DPO fine-tuning:
- prompt: The original input text
- chosen: The preferred output (best/edited variant)
- rejected: The non-preferred output (bad variant)

Usage:
    python scripts/export_dpo_data.py
    python scripts/export_dpo_data.py --output ./data/training/dpo_pairs.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def load_feedback_files(feedback_dir: Path) -> list[dict]:
    """Load all feedback records from JSONL files.

    Args:
        feedback_dir: Directory containing feedback files

    Returns:
        List of feedback records
    """
    records = []

    for feedback_file in sorted(feedback_dir.glob("feedback_*.jsonl")):
        print(f"  Loading {feedback_file.name}...")
        with open(feedback_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"    Warning: Skipping invalid JSON at line {line_num}: {e}")

    return records


def extract_dpo_pairs(record: dict) -> list[dict]:
    """Extract DPO training pairs from a single feedback record.

    Args:
        record: Feedback record with variants and ratings

    Returns:
        List of DPO pairs (prompt, chosen, rejected)
    """
    pairs = []
    input_text = record.get("input_text", "")
    section = record.get("section")
    variants = record.get("variants", [])

    if not input_text or not variants:
        return pairs

    # Categorize variants by feedback type
    best = []
    good = []
    bad = []
    edited = []

    for v in variants:
        ft = v.get("feedback_type")
        original = v.get("original_text", "")
        edited_text = v.get("edited_text")

        if ft == "best" and original:
            best.append({"text": original, "strategy": v.get("strategy")})
        elif ft == "good" and original:
            good.append({"text": original, "strategy": v.get("strategy")})
        elif ft == "bad" and original:
            bad.append({"text": original, "strategy": v.get("strategy")})
        elif ft == "edited" and edited_text:
            edited.append({"text": edited_text, "strategy": v.get("strategy"), "was_edited": True})

    # Preferred outputs (in order of preference): edited > best > good
    preferred = edited + best + good
    rejected = bad

    # Create pairs: each preferred paired with each rejected
    for pref in preferred:
        for rej in rejected:
            # Build the prompt with section context if available
            prompt = input_text
            if section:
                prompt = f"[Section: {section}]\n\n{input_text}"

            pairs.append({
                "prompt": prompt,
                "chosen": pref["text"],
                "rejected": rej["text"],
                "metadata": {
                    "section": section,
                    "chosen_strategy": pref.get("strategy"),
                    "rejected_strategy": rej.get("strategy"),
                    "chosen_was_edited": pref.get("was_edited", False),
                    "session_id": record.get("session_id"),
                    "timestamp": record.get("timestamp")
                }
            })

    return pairs


def export_for_training(pairs: list[dict], output_path: Path, format: str = "jsonl"):
    """Export pairs in training format.

    Args:
        pairs: List of DPO pairs
        output_path: Output file path
        format: Output format (jsonl, huggingface)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        # Standard JSONL format
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    elif format == "huggingface":
        # Format compatible with HuggingFace TRL DPOTrainer
        hf_data = []
        for pair in pairs:
            hf_data.append({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"]
            })
        with open(output_path, "w", encoding="utf-8") as f:
            for item in hf_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    elif format == "alpaca":
        # Alpaca-style format with instruction
        alpaca_data = []
        for pair in pairs:
            alpaca_data.append({
                "instruction": "Rewrite the following paragraph in a scientific writing style while preserving all numbers and factual content.",
                "input": pair["prompt"],
                "chosen_output": pair["chosen"],
                "rejected_output": pair["rejected"]
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(alpaca_data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Export feedback data for DPO training"
    )
    parser.add_argument(
        "--feedback-dir",
        default="./logs/feedback",
        help="Directory containing feedback JSONL files"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/training/dpo_pairs.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "huggingface", "alpaca"],
        default="jsonl",
        help="Output format"
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=0,
        help="Minimum pairs required (exit with error if not met)"
    )
    args = parser.parse_args()

    feedback_dir = Path(args.feedback_dir)
    output_path = Path(args.output)

    print("=" * 60)
    print("GSWA DPO Data Export")
    print("=" * 60)
    print(f"Feedback directory: {feedback_dir}")
    print(f"Output file: {output_path}")
    print(f"Format: {args.format}")
    print()

    # Check feedback directory exists
    if not feedback_dir.exists():
        print(f"Error: Feedback directory does not exist: {feedback_dir}")
        print("\nNo feedback has been collected yet.")
        print("Use the web UI to generate variants and provide feedback first.")
        return 1

    # Load feedback
    print("Loading feedback records...")
    records = load_feedback_files(feedback_dir)
    print(f"  Loaded {len(records)} feedback sessions")
    print()

    if not records:
        print("No feedback records found.")
        return 1

    # Extract DPO pairs
    print("Extracting DPO training pairs...")
    all_pairs = []
    for record in records:
        pairs = extract_dpo_pairs(record)
        all_pairs.extend(pairs)

    print(f"  Extracted {len(all_pairs)} training pairs")
    print()

    if len(all_pairs) < args.min_pairs:
        print(f"Error: Only {len(all_pairs)} pairs extracted, minimum {args.min_pairs} required.")
        print("\nCollect more feedback to generate sufficient training data.")
        return 1

    if not all_pairs:
        print("No valid DPO pairs could be extracted.")
        print("\nMake sure you have:")
        print("  - At least one 'best' or 'edited' rating")
        print("  - At least one 'bad' rating")
        print("  in the same feedback session to create comparison pairs.")
        return 1

    # Export
    print(f"Exporting to {output_path}...")
    export_for_training(all_pairs, output_path, args.format)
    print("  Done!")
    print()

    # Summary statistics
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total feedback sessions: {len(records)}")
    print(f"Total DPO pairs: {len(all_pairs)}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    print()
    print("Next steps for fine-tuning:")
    print("1. Review the exported data for quality")
    print("2. Split into train/validation sets if needed")
    print("3. Use with your preferred DPO training framework:")
    print("   - HuggingFace TRL: DPOTrainer")
    print("   - OpenRLHF")
    print("   - trl-dpo")

    return 0


if __name__ == "__main__":
    sys.exit(main())
