#!/usr/bin/env python3
"""
GSWA Corpus Manager - 傻瓜式语料管理工具
========================================

Easy corpus management for fine-tuning:
- List all corpus files
- Check corpus health and statistics
- Auto-organize files into priority folders
- Preview training weights
- Validate file formats

Usage:
    python scripts/corpus_manager.py                 # Show status
    python scripts/corpus_manager.py --add file.pdf  # Add file to corpus
    python scripts/corpus_manager.py --priority      # Mark files as priority
    python scripts/corpus_manager.py --validate      # Validate all files
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# ==============================================================================
# Constants
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
RAW_DIR = CORPUS_DIR / "raw"
PRIORITY_DIR = RAW_DIR / "important_examples"
WEIGHTS_FILE = CORPUS_DIR / "priority_weights.json"

SUPPORTED_FORMATS = {
    ".pdf": "PDF Document",
    ".docx": "Word Document",
    ".txt": "Plain Text",
    ".md": "Markdown",
    ".tex": "LaTeX",
}

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def colored(text: str, color: str) -> str:
    """Add color to text if terminal supports it."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.ENDC}"
    return text


# ==============================================================================
# Corpus Analysis
# ==============================================================================

def get_corpus_stats() -> dict:
    """Get comprehensive corpus statistics."""
    stats = {
        "regular": [],
        "priority": [],
        "by_format": {},
        "total_size_mb": 0,
        "warnings": [],
        "suggestions": [],
    }

    # Ensure directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PRIORITY_DIR.mkdir(parents=True, exist_ok=True)

    # Scan regular files
    for f in RAW_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
            size_kb = f.stat().st_size / 1024
            stats["regular"].append({
                "name": f.name,
                "path": str(f),
                "format": f.suffix.lower(),
                "size_kb": size_kb,
                "weight": 1.0,
            })
            stats["total_size_mb"] += size_kb / 1024

            ext = f.suffix.lower()
            stats["by_format"][ext] = stats["by_format"].get(ext, 0) + 1

    # Scan priority files
    for f in PRIORITY_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
            size_kb = f.stat().st_size / 1024
            stats["priority"].append({
                "name": f.name,
                "path": str(f),
                "format": f.suffix.lower(),
                "size_kb": size_kb,
                "weight": 2.5,  # Default priority weight
            })
            stats["total_size_mb"] += size_kb / 1024

            ext = f.suffix.lower()
            stats["by_format"][ext] = stats["by_format"].get(ext, 0) + 1

    # Load custom weights
    if WEIGHTS_FILE.exists():
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                custom_weights = json.load(f)

            priority_weight = custom_weights.get("priority_folder_weight", 2.5)
            for doc in stats["priority"]:
                doc["weight"] = priority_weight

            # Apply per-doc weights
            for doc_id, doc_config in custom_weights.get("priority_docs", {}).items():
                for doc in stats["regular"] + stats["priority"]:
                    if doc_id in doc["name"]:
                        doc["weight"] = doc_config.get("weight", doc["weight"])
                        doc["note"] = doc_config.get("reason", "")

        except Exception as e:
            stats["warnings"].append(f"Could not load weights file: {e}")

    # Generate warnings and suggestions
    total = len(stats["regular"]) + len(stats["priority"])

    if total == 0:
        stats["warnings"].append("No corpus files found!")
    elif total < 5:
        stats["warnings"].append(f"Only {total} files - more files improve quality")
    elif total < 10:
        stats["suggestions"].append("Consider adding more files for better results")

    if len(stats["priority"]) == 0 and total > 0:
        stats["suggestions"].append("Add important examples to raw/important_examples/ for higher weight")

    if stats["total_size_mb"] > 500:
        stats["warnings"].append(f"Large corpus ({stats['total_size_mb']:.1f}MB) may slow training")

    return stats


def print_corpus_status():
    """Print formatted corpus status."""
    stats = get_corpus_stats()

    print("\n" + "=" * 70)
    print(colored("GSWA Corpus Manager - 语料库状态", Colors.HEADER + Colors.BOLD))
    print("=" * 70)

    # Directory info
    print(f"\n{colored('目录 (Directories):', Colors.BLUE)}")
    print(f"  常规文章:  {RAW_DIR}")
    print(f"  重要文章:  {PRIORITY_DIR}")

    # Regular files
    print(f"\n{colored('常规文章 (Regular Documents):', Colors.BLUE)} [{len(stats['regular'])}]")
    if stats["regular"]:
        for doc in stats["regular"][:10]:
            weight_str = f"x{doc['weight']}" if doc["weight"] != 1.0 else ""
            size_str = f"{doc['size_kb']:.0f}KB"
            print(f"  📄 {doc['name']:<50} {size_str:>8} {weight_str}")
        if len(stats["regular"]) > 10:
            print(f"  ... 还有 {len(stats['regular']) - 10} 个文件")
    else:
        print(f"  {colored('(空)', Colors.YELLOW)}")

    # Priority files
    print(f"\n{colored('重要文章 (Priority Documents):', Colors.BLUE)} [{len(stats['priority'])}] - 权重 x2.5")
    if stats["priority"]:
        for doc in stats["priority"]:
            size_str = f"{doc['size_kb']:.0f}KB"
            print(f"  ⭐ {doc['name']:<50} {size_str:>8} x{doc['weight']}")
    else:
        print(f"  {colored('(空 - 把重要的例子放这里!)', Colors.YELLOW)}")

    # Summary
    total = len(stats["regular"]) + len(stats["priority"])
    print(f"\n{colored('统计 (Statistics):', Colors.BLUE)}")
    print(f"  总文件数:    {total}")
    print(f"  总大小:      {stats['total_size_mb']:.2f} MB")

    if stats["by_format"]:
        format_str = ", ".join([f"{k}: {v}" for k, v in stats["by_format"].items()])
        print(f"  文件格式:    {format_str}")

    # Warnings
    if stats["warnings"]:
        print(f"\n{colored('⚠️  警告 (Warnings):', Colors.RED)}")
        for w in stats["warnings"]:
            print(f"  • {w}")

    # Suggestions
    if stats["suggestions"]:
        print(f"\n{colored('💡 建议 (Suggestions):', Colors.YELLOW)}")
        for s in stats["suggestions"]:
            print(f"  • {s}")

    # Training weight preview
    print(f"\n{colored('训练权重预览 (Training Weight Preview):', Colors.BLUE)}")
    regular_weight = len(stats["regular"]) * 1.0
    priority_weight = len(stats["priority"]) * 2.5
    total_weight = regular_weight + priority_weight

    if total_weight > 0:
        print(f"  常规文章权重:   {regular_weight:.1f} ({regular_weight/total_weight*100:.0f}%)")
        print(f"  重要文章权重:   {priority_weight:.1f} ({priority_weight/total_weight*100:.0f}%)")
        print(f"  总训练权重:     {total_weight:.1f}")


def print_quick_guide():
    """Print quick start guide."""
    print(f"\n{colored('快速开始 (Quick Start):', Colors.GREEN)}")
    print("-" * 70)
    print("""
  1. 放入文章 (Add Documents):
     # 常规文章 - 直接拖拽或复制到:
     data/corpus/raw/

     # 重要/代表性文章 - 放到:
     data/corpus/raw/important_examples/

  2. 检查状态 (Check Status):
     make list-docs  或  python scripts/corpus_manager.py

  3. 开始训练 (Start Training):
     make finetune-all

  支持的格式: .pdf, .docx, .txt, .md, .tex
""")


# ==============================================================================
# File Operations
# ==============================================================================

def add_file(filepath: str, priority: bool = False):
    """Add a file to the corpus."""
    src = Path(filepath)

    if not src.exists():
        print(f"{colored('ERROR:', Colors.RED)} File not found: {filepath}")
        return False

    if src.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"{colored('ERROR:', Colors.RED)} Unsupported format: {src.suffix}")
        print(f"Supported: {', '.join(SUPPORTED_FORMATS.keys())}")
        return False

    # Determine destination
    if priority:
        dest_dir = PRIORITY_DIR
    else:
        dest_dir = RAW_DIR

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    # Check if exists
    if dest.exists():
        print(f"{colored('WARNING:', Colors.YELLOW)} File already exists: {dest.name}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            return False

    # Copy file
    shutil.copy2(src, dest)
    priority_str = " (⭐ priority)" if priority else ""
    print(f"{colored('Added:', Colors.GREEN)} {src.name}{priority_str}")
    print(f"  → {dest}")
    return True


def move_to_priority(filename: str):
    """Move a file from regular to priority folder."""
    src = RAW_DIR / filename

    if not src.exists():
        # Try fuzzy match
        matches = list(RAW_DIR.glob(f"*{filename}*"))
        if matches:
            src = matches[0]
            print(f"Matched: {src.name}")
        else:
            print(f"{colored('ERROR:', Colors.RED)} File not found: {filename}")
            return False

    PRIORITY_DIR.mkdir(parents=True, exist_ok=True)
    dest = PRIORITY_DIR / src.name

    shutil.move(src, dest)
    print(f"{colored('Moved to priority:', Colors.GREEN)} {src.name}")
    print(f"  → {dest}")
    return True


def validate_corpus():
    """Validate all corpus files."""
    print("\n" + "=" * 70)
    print(colored("Validating Corpus Files", Colors.HEADER))
    print("=" * 70)

    stats = get_corpus_stats()
    all_files = stats["regular"] + stats["priority"]

    valid = 0
    invalid = 0

    for doc in all_files:
        path = Path(doc["path"])
        print(f"\nChecking: {doc['name']}")

        # Check file exists and is readable
        if not path.exists():
            print(f"  {colored('✗ File not found', Colors.RED)}")
            invalid += 1
            continue

        # Check file size
        if doc["size_kb"] < 1:
            print(f"  {colored('✗ File too small (< 1KB)', Colors.RED)}")
            invalid += 1
            continue

        # Check format-specific validation
        ext = path.suffix.lower()
        if ext == ".pdf":
            # Check if PDF is readable
            try:
                with open(path, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        print(f"  {colored('✗ Invalid PDF header', Colors.RED)}")
                        invalid += 1
                        continue
                print(f"  {colored('✓ Valid PDF', Colors.GREEN)} ({doc['size_kb']:.0f}KB)")
                valid += 1
            except Exception as e:
                print(f"  {colored(f'✗ Cannot read: {e}', Colors.RED)}")
                invalid += 1

        elif ext == ".docx":
            # Check if DOCX is valid ZIP
            try:
                import zipfile
                if zipfile.is_zipfile(path):
                    print(f"  {colored('✓ Valid DOCX', Colors.GREEN)} ({doc['size_kb']:.0f}KB)")
                    valid += 1
                else:
                    print(f"  {colored('✗ Invalid DOCX format', Colors.RED)}")
                    invalid += 1
            except Exception as e:
                print(f"  {colored(f'✗ Cannot validate: {e}', Colors.RED)}")
                invalid += 1

        elif ext in [".txt", ".md", ".tex"]:
            # Check if text file is readable
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read(100)
                print(f"  {colored('✓ Valid text file', Colors.GREEN)} ({doc['size_kb']:.0f}KB)")
                valid += 1
            except Exception as e:
                print(f"  {colored(f'✗ Cannot read: {e}', Colors.RED)}")
                invalid += 1

    print("\n" + "-" * 70)
    print(f"Validation Results: {colored(f'{valid} valid', Colors.GREEN)}, {colored(f'{invalid} invalid', Colors.RED)}")

    return invalid == 0


def set_weight(filename: str, weight: float, reason: str = ""):
    """Set custom weight for a document."""
    # Load existing weights
    weights = {"default_weight": 1.0, "priority_folder_weight": 2.5, "priority_docs": {}, "exclude_docs": {}}

    if WEIGHTS_FILE.exists():
        with open(WEIGHTS_FILE, 'r') as f:
            weights = json.load(f)

    # Find the document
    stats = get_corpus_stats()
    found = None
    for doc in stats["regular"] + stats["priority"]:
        if filename.lower() in doc["name"].lower():
            found = doc
            break

    if not found:
        print(f"{colored('ERROR:', Colors.RED)} Document not found: {filename}")
        return False

    # Extract doc ID (filename without extension)
    doc_id = Path(found["name"]).stem

    # Set weight
    weights["priority_docs"][doc_id] = {
        "weight": weight,
        "reason": reason or f"Custom weight set {datetime.now().strftime('%Y-%m-%d')}",
    }

    # Save
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)

    print(f"{colored('Weight set:', Colors.GREEN)} {found['name']} → x{weight}")
    return True


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GSWA Corpus Manager - 傻瓜式语料管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show corpus status
  python scripts/corpus_manager.py

  # Add a file to corpus
  python scripts/corpus_manager.py --add paper.pdf

  # Add a file as priority (high weight)
  python scripts/corpus_manager.py --add paper.pdf --priority

  # Move existing file to priority
  python scripts/corpus_manager.py --move-priority paper.pdf

  # Set custom weight
  python scripts/corpus_manager.py --set-weight paper.pdf 3.0

  # Validate all files
  python scripts/corpus_manager.py --validate

  # Show quick guide
  python scripts/corpus_manager.py --guide
        """
    )

    parser.add_argument("--add", type=str, metavar="FILE",
                        help="Add a file to the corpus")
    parser.add_argument("--priority", action="store_true",
                        help="Add file as priority (use with --add)")
    parser.add_argument("--move-priority", type=str, metavar="FILE",
                        help="Move existing file to priority folder")
    parser.add_argument("--set-weight", nargs=2, metavar=("FILE", "WEIGHT"),
                        help="Set custom weight for a document")
    parser.add_argument("--validate", action="store_true",
                        help="Validate all corpus files")
    parser.add_argument("--guide", action="store_true",
                        help="Show quick start guide")
    parser.add_argument("--json", action="store_true",
                        help="Output stats as JSON")

    args = parser.parse_args()

    # Handle operations
    if args.add:
        add_file(args.add, args.priority)
        print_corpus_status()

    elif args.move_priority:
        move_to_priority(args.move_priority)
        print_corpus_status()

    elif args.set_weight:
        filename, weight = args.set_weight
        set_weight(filename, float(weight))
        print_corpus_status()

    elif args.validate:
        validate_corpus()

    elif args.guide:
        print_quick_guide()

    elif args.json:
        stats = get_corpus_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    else:
        print_corpus_status()
        print_quick_guide()


if __name__ == "__main__":
    main()
