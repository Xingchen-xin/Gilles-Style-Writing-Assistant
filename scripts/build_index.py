#!/usr/bin/env python3
"""
Build similarity index from corpus.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --corpus-path ./data/corpus/parsed --index-path ./data/index
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build GSWA similarity index")
    parser.add_argument(
        "--corpus-path",
        default="./data/corpus/parsed",
        help="Path to corpus JSONL files"
    )
    parser.add_argument(
        "--index-path",
        default="./data/index",
        help="Path to save index files"
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus_path)
    index_path = Path(args.index_path)

    print("=" * 50)
    print("Building GSWA Similarity Index")
    print("=" * 50)
    print(f"Corpus path: {corpus_path}")
    print(f"Index path: {index_path}")
    print()

    # Check corpus exists
    if not corpus_path.exists():
        print(f"Error: Corpus path does not exist: {corpus_path}")
        print("Please add JSONL files to the corpus directory first.")
        print()
        print("Expected format (one JSON per line):")
        print('  {"text": "Paragraph text...", "doc_id": "paper1", "para_id": "p1"}')
        return 1

    # Load corpus
    print("Loading corpus...")
    texts = []
    doc_ids = []
    para_ids = []

    for jsonl_file in corpus_path.glob("*.jsonl"):
        print(f"  Reading {jsonl_file.name}...")
        with open(jsonl_file) as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    texts.append(item["text"])
                    doc_ids.append(item.get("doc_id", f"{jsonl_file.stem}"))
                    para_ids.append(item.get("para_id", f"p{line_num}"))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"    Warning: Skipping line {line_num}: {e}")

    if not texts:
        print("Error: No valid paragraphs found in corpus.")
        return 1

    print(f"Loaded {len(texts)} paragraphs from corpus")
    print()

    # Build n-gram index
    print("Building n-gram index...")
    from gswa.utils.ngram import build_ngram_index

    ngram_set = build_ngram_index(texts, n=8)
    print(f"  Created {len(ngram_set)} unique 8-grams")

    # Save n-gram index
    index_path.mkdir(parents=True, exist_ok=True)
    ngram_file = index_path / "ngrams.json"
    with open(ngram_file, "w") as f:
        json.dump([list(ng) for ng in ngram_set], f)
    print(f"  Saved to {ngram_file}")
    print()

    # Build embedding index
    print("Building embedding index...")
    try:
        from gswa.utils.embedding import EmbeddingService

        embedding_service = EmbeddingService()
        embedding_service.build_index(texts, doc_ids, para_ids)
        embedding_service.save_index(str(index_path))
        print(f"  Saved embedding index to {index_path}")
    except ImportError:
        print("  Warning: sentence-transformers not available, skipping embedding index")
        print("  Install with: pip install sentence-transformers faiss-cpu")

    print()
    print("=" * 50)
    print("Index build complete!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
