#!/usr/bin/env python3
"""
Parse corpus documents (PDF/DOCX) to JSONL format.

This script extracts paragraphs from Gilles's papers for:
1. Building similarity index (anti-verbatim checking)
2. Style reference for the model

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


def parse_pdf(file_path: Path) -> Iterator[dict]:
    """Parse PDF file to paragraphs.

    Args:
        file_path: Path to PDF file

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
                "source_file": file_path.name
            }

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")


def parse_docx(file_path: Path) -> Iterator[dict]:
    """Parse DOCX file to paragraphs.

    Args:
        file_path: Path to DOCX file

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
                "source_file": file_path.name
            }

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")


def parse_txt(file_path: Path) -> Iterator[dict]:
    """Parse plain text file to paragraphs.

    Args:
        file_path: Path to TXT file

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
                "source_file": file_path.name
            }

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")


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

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        print("\nPlease add your source documents (PDF/DOCX/TXT) to this directory.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GSWA Corpus Parser")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Find all documents
    pdf_files = list(input_dir.glob("*.pdf"))
    docx_files = list(input_dir.glob("*.docx"))
    txt_files = list(input_dir.glob("*.txt"))

    all_files = pdf_files + docx_files + txt_files

    if not all_files:
        print("No PDF, DOCX, or TXT files found in input directory.")
        print("\nTo prepare your corpus:")
        print("1. Collect Gilles's published papers (PDF or DOCX)")
        print("2. Place them in: ./data/corpus/raw/")
        print("3. Run this script again")
        return 1

    print(f"Found {len(all_files)} documents:")
    print(f"  - PDF: {len(pdf_files)}")
    print(f"  - DOCX: {len(docx_files)}")
    print(f"  - TXT: {len(txt_files)}")
    print()

    # Process each file
    total_paragraphs = 0
    output_file = output_dir / "corpus.jsonl"

    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in all_files:
            print(f"Processing: {file_path.name}...")

            if file_path.suffix.lower() == '.pdf':
                paragraphs = parse_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                paragraphs = parse_docx(file_path)
            else:
                paragraphs = parse_txt(file_path)

            count = 0
            for para in paragraphs:
                f.write(json.dumps(para, ensure_ascii=False) + '\n')
                count += 1

            print(f"  Extracted {count} paragraphs")
            total_paragraphs += count

    print()
    print("=" * 60)
    print(f"Total paragraphs extracted: {total_paragraphs}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Build similarity index: python scripts/build_index.py")
    print("2. Start the server: make run")

    return 0


if __name__ == "__main__":
    sys.exit(main())
