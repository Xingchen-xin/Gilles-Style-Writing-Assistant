#!/usr/bin/env python3
"""
Parse corpus documents (PDF/DOCX) to JSONL format.

This script extracts paragraphs from Gilles's papers for:
1. Building similarity index (anti-verbatim checking)
2. Style reference for the model

Folder structure:
    data/corpus/raw/                  <- Regular articles (weight = 1.0)
    data/corpus/raw/important_examples/   <- Priority articles (weight = 2.5)

Usage:
    python scripts/parse_corpus.py --input ./data/corpus/raw --output ./data/corpus/parsed

Input: PDF or DOCX files in the input directory
Output: JSONL files with one paragraph per line
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator


def extract_paragraphs_from_text(text: str, min_words: int = 20) -> list[str]:
    """Extract meaningful paragraphs from text.

    Args:
        text: Raw text content
        min_words: Minimum words for a valid paragraph

    Returns:
        List of paragraph strings
    """
    # Split by double newlines or multiple spaces
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)

    result = []
    for para in paragraphs:
        # Clean up whitespace
        para = ' '.join(para.split())

        # Skip if too short
        if len(para.split()) < min_words:
            continue

        # Skip if looks like header/footer/reference
        if re.match(r'^(Figure|Table|References|Bibliography|\d+\.?\s*$)', para):
            continue

        # Skip if mostly numbers (likely a table)
        words = para.split()
        num_count = sum(1 for w in words if re.match(r'^[\d.,]+$', w))
        if num_count > len(words) * 0.5:
            continue

        result.append(para)

    return result


def parse_pdf(file_path: Path, is_priority: bool = False) -> Iterator[dict]:
    """Parse PDF file to paragraphs.

    Args:
        file_path: Path to PDF file
        is_priority: Whether this is a priority document

    Yields:
        Paragraph dictionaries
    """
    try:
        import pymupdf  # PyMuPDF
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            print("Error: PyMuPDF not installed. Run: pip install pymupdf")
            return

    doc_id = file_path.stem

    try:
        doc = pymupdf.open(str(file_path))
        full_text = ""

        for page in doc:
            full_text += page.get_text() + "\n\n"

        doc.close()

        paragraphs = extract_paragraphs_from_text(full_text)

        for i, para in enumerate(paragraphs):
            yield {
                "text": para,
                "doc_id": doc_id,
                "para_id": f"p{i+1}",
                "source_type": "pdf",
                "source_file": file_path.name,
                "is_priority": is_priority
            }

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")


def parse_docx(file_path: Path, is_priority: bool = False) -> Iterator[dict]:
    """Parse DOCX file to paragraphs.

    Args:
        file_path: Path to DOCX file
        is_priority: Whether this is a priority document

    Yields:
        Paragraph dictionaries
    """
    try:
        from docx import Document
    except ImportError:
        print("Error: python-docx not installed. Run: pip install python-docx")
        return

    doc_id = file_path.stem

    try:
        doc = Document(str(file_path))
        full_text = "\n\n".join([para.text for para in doc.paragraphs])

        paragraphs = extract_paragraphs_from_text(full_text)

        for i, para in enumerate(paragraphs):
            yield {
                "text": para,
                "doc_id": doc_id,
                "para_id": f"p{i+1}",
                "source_type": "docx",
                "source_file": file_path.name,
                "is_priority": is_priority
            }

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")


def parse_txt(file_path: Path, is_priority: bool = False) -> Iterator[dict]:
    """Parse plain text file to paragraphs.

    Args:
        file_path: Path to TXT file
        is_priority: Whether this is a priority document

    Yields:
        Paragraph dictionaries
    """
    doc_id = file_path.stem

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        paragraphs = extract_paragraphs_from_text(full_text)

        for i, para in enumerate(paragraphs):
            yield {
                "text": para,
                "doc_id": doc_id,
                "para_id": f"p{i+1}",
                "source_type": "txt",
                "source_file": file_path.name,
                "is_priority": is_priority
            }

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")


def collect_files(directory: Path, is_priority: bool = False) -> list[tuple[Path, bool]]:
    """Collect all document files from a directory.

    Args:
        directory: Directory to scan
        is_priority: Whether files in this directory are priority

    Returns:
        List of (file_path, is_priority) tuples
    """
    files = []
    if directory.exists():
        for ext in ['*.pdf', '*.docx', '*.txt', '*.PDF', '*.DOCX', '*.TXT']:
            for f in directory.glob(ext):
                files.append((f, is_priority))
    return files


def main():
    parser = argparse.ArgumentParser(description="Parse corpus documents to JSONL")
    parser.add_argument(
        "--input", "-i",
        default="./data/corpus/raw",
        help="Input directory with PDF/DOCX files"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/corpus/parsed",
        help="Output directory for JSONL files"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=20,
        help="Minimum words per paragraph"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    priority_dir = input_dir / "important_examples"

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        print("\nPlease add your source documents (PDF/DOCX/TXT) to this directory.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GSWA Corpus Parser")
    print("=" * 60)
    print(f"Input (regular):  {input_dir}")
    print(f"Input (priority): {priority_dir}")
    print(f"Output: {output_dir}")
    print()

    # Collect files from both directories
    regular_files = collect_files(input_dir, is_priority=False)
    priority_files = collect_files(priority_dir, is_priority=True)
    all_files = regular_files + priority_files

    if not all_files:
        print("No PDF, DOCX, or TXT files found.")
        print()
        print("=" * 60)
        print("📁 Folder Structure Guide")
        print("=" * 60)
        print()
        print("Place your documents in these folders:")
        print()
        print("  data/corpus/raw/")
        print("  └── [Regular articles go here]")
        print("  │   └── paper1.pdf")
        print("  │   └── paper2.docx")
        print("  │")
        print("  └── important_examples/")
        print("      └── [Priority articles go here - will have 2.5x weight]")
        print("      └── best_paper.pdf")
        print()
        print("Priority articles (in important_examples/) will be weighted")
        print("2.5x higher during fine-tuning, helping the model better")
        print("capture Gilles's preferred writing style.")
        return 1

    # Count files by type and priority
    regular_count = len(regular_files)
    priority_count = len(priority_files)
    pdf_count = sum(1 for f, _ in all_files if f.suffix.lower() == '.pdf')
    docx_count = sum(1 for f, _ in all_files if f.suffix.lower() == '.docx')
    txt_count = sum(1 for f, _ in all_files if f.suffix.lower() == '.txt')

    print(f"Found {len(all_files)} documents:")
    print(f"  - Regular articles:  {regular_count}")
    print(f"  - Priority articles: {priority_count} (in important_examples/)")
    print()
    print(f"  - PDF: {pdf_count}")
    print(f"  - DOCX: {docx_count}")
    print(f"  - TXT: {txt_count}")
    print()

    # Process each file
    total_paragraphs = 0
    priority_paragraphs = 0
    output_file = output_dir / "corpus.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path, is_priority in all_files:
            priority_marker = " ⭐" if is_priority else ""
            print(f"Processing: {file_path.name}{priority_marker}...")

            if file_path.suffix.lower() == '.pdf':
                paragraphs = parse_pdf(file_path, is_priority)
            elif file_path.suffix.lower() == '.docx':
                paragraphs = parse_docx(file_path, is_priority)
            else:
                paragraphs = parse_txt(file_path, is_priority)

            count = 0
            for para in paragraphs:
                f.write(json.dumps(para, ensure_ascii=False) + '\n')
                count += 1
                if is_priority:
                    priority_paragraphs += 1

            print(f"  Extracted {count} paragraphs")
            total_paragraphs += count

    print()
    print("=" * 60)
    print(f"Total paragraphs extracted: {total_paragraphs}")
    print(f"  - From regular articles:  {total_paragraphs - priority_paragraphs}")
    print(f"  - From priority articles: {priority_paragraphs} ⭐")
    print(f"Output file: {output_file}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Prepare training data: make prepare-training")
    print("2. Fine-tune model: make finetune-mlx  (Mac)")
    print("                    make finetune-lora (Linux)")
    print("3. Or just build index and run: make build-index && make run")

    return 0


if __name__ == "__main__":
    sys.exit(main())
