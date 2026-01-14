#!/usr/bin/env python3
"""
Prepare training data for fine-tuning from Gilles's corpus.

This script creates training datasets in various formats:
1. Alpaca format (for most fine-tuning frameworks)
2. ShareGPT format (for Axolotl, etc.)
3. DPO pairs (from feedback data)

Usage:
    python scripts/prepare_training_data.py --format alpaca --output ./data/training/
    python scripts/prepare_training_data.py --format sharegpt --weighted
    python scripts/prepare_training_data.py --format dpo --from-feedback
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_corpus(corpus_path: str) -> list[dict]:
    """Load corpus from JSONL file."""
    paragraphs = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                paragraphs.append(json.loads(line))
    return paragraphs


def load_weights(weights_path: str) -> dict:
    """Load priority weights configuration."""
    if not Path(weights_path).exists():
        return {"default_weight": 1.0, "priority_docs": {}, "exclude_docs": {}}

    with open(weights_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def apply_weights(paragraphs: list[dict], weights: dict) -> list[dict]:
    """Apply weights to paragraphs, duplicating high-priority ones.

    Weight priority:
    1. Documents in important_examples/ folder get priority_folder_weight (default 2.5)
    2. Documents explicitly listed in priority_docs get their specified weight
    3. Other documents get default_weight (default 1.0)
    """
    default_weight = weights.get("default_weight", 1.0)
    priority_folder_weight = weights.get("priority_folder_weight", 2.5)
    priority_docs = weights.get("priority_docs", {})
    exclude_docs = weights.get("exclude_docs", {})

    weighted_paragraphs = []
    stats = defaultdict(lambda: {"count": 0, "weight": 0, "is_priority": False})

    for para in paragraphs:
        doc_id = para.get("doc_id", "unknown")
        is_priority = para.get("is_priority", False)

        # Skip excluded documents
        if doc_id in exclude_docs:
            continue

        # Determine weight (priority folder > explicit config > default)
        if is_priority:
            # Document is from important_examples/ folder
            weight = priority_folder_weight
        elif doc_id in priority_docs:
            # Document explicitly configured
            weight = priority_docs[doc_id].get("weight", default_weight)
        else:
            weight = default_weight

        # Add paragraph (possibly multiple times based on weight)
        repeat_count = max(1, int(weight))
        for _ in range(repeat_count):
            weighted_paragraphs.append(para)

        stats[doc_id]["count"] += repeat_count
        stats[doc_id]["weight"] = weight
        stats[doc_id]["is_priority"] = is_priority

    return weighted_paragraphs, dict(stats)


def create_style_prompts() -> list[str]:
    """Create varied prompts for style transfer training."""
    return [
        "Rewrite the following scientific paragraph in a clear, precise academic style:",
        "Transform this text into polished scientific prose:",
        "Improve this paragraph for an academic research paper:",
        "Rewrite this passage with better scientific clarity and flow:",
        "Edit this scientific text for publication quality:",
        "Paraphrase this research paragraph in formal academic English:",
        "Refine this paragraph for a peer-reviewed journal:",
        "Rewrite this scientific content with improved structure and clarity:",
    ]


def create_alpaca_format(
    paragraphs: list[dict],
    include_source: bool = False
) -> list[dict]:
    """Create Alpaca-format training data.

    Alpaca format:
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }
    """
    prompts = create_style_prompts()
    training_data = []

    for i, para in enumerate(paragraphs):
        text = para.get("text", "").strip()
        if not text or len(text) < 100:
            continue

        # Use the actual text as both input and output for style learning
        # The model learns to reproduce Gilles's style
        entry = {
            "instruction": random.choice(prompts),
            "input": text,  # For self-supervised: same text
            "output": text,  # Model learns to reproduce the style
        }

        if include_source:
            entry["source"] = {
                "doc_id": para.get("doc_id"),
                "para_id": para.get("para_id"),
            }

        training_data.append(entry)

    return training_data


def create_sharegpt_format(
    paragraphs: list[dict],
    include_source: bool = False
) -> list[dict]:
    """Create ShareGPT-format training data.

    ShareGPT format:
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """
    prompts = create_style_prompts()
    training_data = []

    for para in paragraphs:
        text = para.get("text", "").strip()
        if not text or len(text) < 100:
            continue

        entry = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"{random.choice(prompts)}\n\n{text}"
                },
                {
                    "from": "gpt",
                    "value": text
                }
            ]
        }

        if include_source:
            entry["source"] = {
                "doc_id": para.get("doc_id"),
                "para_id": para.get("para_id"),
            }

        training_data.append(entry)

    return training_data


def create_completion_format(paragraphs: list[dict]) -> list[dict]:
    """Create completion-only format for continued pretraining.

    This format is for teaching the model Gilles's writing style
    through exposure to his actual text.
    """
    training_data = []

    for para in paragraphs:
        text = para.get("text", "").strip()
        if not text or len(text) < 100:
            continue

        training_data.append({
            "text": text,
            "doc_id": para.get("doc_id"),
        })

    return training_data


def create_contrastive_pairs(paragraphs: list[dict]) -> list[dict]:
    """Create contrastive pairs for style learning.

    Format for teaching model to distinguish Gilles's style.
    """
    training_data = []

    # Group by document
    by_doc = defaultdict(list)
    for para in paragraphs:
        by_doc[para.get("doc_id", "unknown")].append(para)

    prompts = create_style_prompts()

    for doc_id, doc_paras in by_doc.items():
        for para in doc_paras:
            text = para.get("text", "").strip()
            if not text or len(text) < 100:
                continue

            # Create "humanized" version instructions
            entry = {
                "instruction": random.choice(prompts),
                "input": text,
                "chosen": text,  # Gilles's actual text is preferred
                "rejected": "",  # Will be filled by model generations
                "doc_id": doc_id,
            }
            training_data.append(entry)

    return training_data


def load_feedback_data(feedback_dir: str) -> list[dict]:
    """Load feedback data for DPO training."""
    feedback_path = Path(feedback_dir)
    if not feedback_path.exists():
        return []

    all_feedback = []
    for f in feedback_path.glob("*.jsonl"):
        with open(f, 'r') as file:
            for line in file:
                if line.strip():
                    all_feedback.append(json.loads(line))

    return all_feedback


def create_dpo_format(feedback_data: list[dict]) -> list[dict]:
    """Create DPO training data from user feedback.

    DPO format:
    {
        "prompt": "...",
        "chosen": "...",
        "rejected": "..."
    }
    """
    dpo_data = []

    for entry in feedback_data:
        input_text = entry.get("input_text", "")
        variants = entry.get("variants", [])

        # Find best and worst rated variants
        best = None
        worst = None

        for v in variants:
            fb_type = v.get("feedback_type", "")
            if fb_type == "best":
                best = v
            elif fb_type == "bad":
                worst = v

        if best and worst:
            dpo_data.append({
                "prompt": f"Rewrite this scientific paragraph:\n\n{input_text}",
                "chosen": best.get("text", ""),
                "rejected": worst.get("text", ""),
            })

    return dpo_data


def split_data(
    data: list[dict],
    train_ratio: float = 0.9,
    seed: int = 42
) -> tuple[list, list]:
    """Split data into train and validation sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for fine-tuning")
    parser.add_argument(
        "--corpus", "-c",
        default="./data/corpus/parsed/corpus.jsonl",
        help="Path to corpus JSONL file"
    )
    parser.add_argument(
        "--weights", "-w",
        default="./data/corpus/priority_weights.json",
        help="Path to priority weights JSON"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/training",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["alpaca", "sharegpt", "completion", "contrastive", "dpo", "all"],
        default="alpaca",
        help="Output format"
    )
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Apply priority weights to corpus"
    )
    parser.add_argument(
        "--from-feedback",
        action="store_true",
        help="Create DPO data from feedback (for dpo format)"
    )
    parser.add_argument(
        "--feedback-dir",
        default="./logs/feedback",
        help="Directory containing feedback JSONL files"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/validation sets"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Training set ratio (default: 0.9)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GSWA Training Data Preparation")
    print("=" * 60)

    # Load corpus
    print(f"\nLoading corpus from: {args.corpus}")
    paragraphs = load_corpus(args.corpus)
    print(f"  Loaded {len(paragraphs)} paragraphs")

    # Apply weights if requested
    if args.weighted:
        print(f"\nApplying weights from: {args.weights}")
        weights = load_weights(args.weights)
        paragraphs, stats = apply_weights(paragraphs, weights)
        print(f"  After weighting: {len(paragraphs)} paragraphs")

        # Show weight stats
        folder_priority = sum(1 for d, s in stats.items() if s.get("is_priority"))
        config_priority = sum(1 for d, s in stats.items()
                            if s["weight"] > 1.0 and not s.get("is_priority"))
        print(f"  Priority documents (from important_examples/): {folder_priority}")
        print(f"  Priority documents (from config): {config_priority}")

    # Generate training data
    formats_to_generate = (
        ["alpaca", "sharegpt", "completion", "contrastive"]
        if args.format == "all"
        else [args.format]
    )

    for fmt in formats_to_generate:
        print(f"\nGenerating {fmt} format...")

        if fmt == "alpaca":
            data = create_alpaca_format(paragraphs)
        elif fmt == "sharegpt":
            data = create_sharegpt_format(paragraphs)
        elif fmt == "completion":
            data = create_completion_format(paragraphs)
        elif fmt == "contrastive":
            data = create_contrastive_pairs(paragraphs)
        elif fmt == "dpo":
            if args.from_feedback:
                feedback = load_feedback_data(args.feedback_dir)
                print(f"  Loaded {len(feedback)} feedback entries")
                data = create_dpo_format(feedback)
            else:
                print("  Warning: --from-feedback not set, creating empty DPO file")
                data = []
        else:
            continue

        # Split if requested
        if args.split and data:
            train_data, val_data = split_data(data, args.train_ratio)

            train_path = output_dir / f"{fmt}_train.jsonl"
            val_path = output_dir / f"{fmt}_val.jsonl"

            with open(train_path, 'w', encoding='utf-8') as f:
                for entry in train_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            with open(val_path, 'w', encoding='utf-8') as f:
                for entry in val_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            print(f"  Saved: {train_path} ({len(train_data)} entries)")
            print(f"  Saved: {val_path} ({len(val_data)} entries)")
        else:
            output_path = output_dir / f"{fmt}.jsonl"
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            print(f"  Saved: {output_path} ({len(data)} entries)")

    print("\n" + "=" * 60)
    print("Training data preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review the generated training data")
    print("  2. Run fine-tuning:")
    print("     - LoRA: python scripts/finetune_lora.py")
    print("     - QLoRA (Mac): python scripts/finetune_qlora_mac.py")
    print("  3. Test the fine-tuned model")


if __name__ == "__main__":
    main()
