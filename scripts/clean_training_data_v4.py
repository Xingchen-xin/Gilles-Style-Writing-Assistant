#!/usr/bin/env python3
"""
Clean training data v4 - MAXIMUM AGGRESSIVE cleaning.

Addresses ALL remaining contamination including:
1. ALL year references (1900-2099)
2. ALL author-year patterns
3. ALL partial citations
4. ALL journal abbreviations
5. Numeric artifacts from figures/tables

The goal is ABSOLUTELY ZERO academic reference artifacts.

Usage:
    python scripts/clean_training_data_v4.py --input data/training/alpaca_train.jsonl
"""
import argparse
import json
import re
import random
from pathlib import Path
from difflib import SequenceMatcher


# ============================================================
# MAXIMUM AGGRESSIVE patterns to REMOVE
# ============================================================

REMOVE_PATTERNS = [
    # === Year references (CRITICAL - remove ALL years) ===
    r'\((?:19|20)\d{2}[a-z]?\)',                  # (2001) or (2001a)
    r',\s*(?:19|20)\d{2}[a-z]?\)',                # , 2001)
    r'\((?:19|20)\d{2}[a-z]?,\s*(?:19|20)\d{2}[a-z]?\)',  # (2001, 2002)
    r'(?:19|20)\d{2}[a-z]?(?:\s*[-–]\s*(?:19|20)\d{2}[a-z]?)?',  # 2001 or 2001-2005

    # === Author patterns ===
    r'[A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\(?(?:19|20)\d{2}[a-z]?\)?',  # Smith et al., 2001
    r'\([A-Z][a-z]+\s+and\s+[A-Z][a-z]+,?\s*(?:19|20)?\d{2,4}?\)',   # (Smith and Jones, 2001)
    r'[A-Z][a-z]+-[A-Z][a-z]+',                   # Hyphenated names

    # === Citation formats ===
    r'\s*\(\d+(?:[,–-]\s*\d+)*\)',                # (1), (1,2), (1-5)
    r'\s*\[\d+(?:[,–-]\s*\d+)*\]',                # [1], [1,2], [1-5]
    r'\bet\s+al\.\s*',                            # et al.

    # === Journal patterns ===
    r'(?:Nat|Nature|Science|Cell|J|Mol|Biol|Chem|Biochem|Microbiol|Struct|Genet|Rev|Proc|Acad|Ann|Appl|BMC|PLoS|PNAS|EMBO|Sci|Rep|Res)\.(?:\s+[A-Z])?',
    r'[A-Z][a-z]+\s+\d+:\s*\d+[-–]\d+',          # Cell 87: 1295-1306
    r'\d{4},\s*\d+(?:\(\d+\))?,\s*\d+[-–]?\d*',  # 2021, 49, 5349-5361
    r'\d+:\s*\d+[-–]\d+',                         # 87: 1295-1306
    r'\d+\(\d+\):\s*\d+',                         # 49(3): 123

    # === DOI/URLs ===
    r'(?:doi:\s*)?10\.\d{4,}/[^\s\)]+',
    r'https?://[^\s]+',
    r'www\.[^\s]+',

    # === Figure/Table/Section ===
    r'(?:see\s+)?(?:Fig(?:ure)?|Table|Supplementary)\s*\.?\s*S?\d+[A-Za-z]?(?:\s*(?:and|,)\s*S?\d+[A-Za-z]?)*',
    r'\((?:Fig(?:ure)?|Table)\s*\.?\s*S?\d+[A-Za-z]?\)',
    r'(?:Section|Sect\.|§)\s*\d+(?:\.\d+)*',
    r'^\s*\d+\.\d+(?:\.\d+)*\s+',

    # === Author lists ===
    r'[A-Z][a-z]+,\s*[A-Z]\.\s*(?:[A-Z]\.\s*)?(?:,\s*(?:and\s+)?[A-Z][a-z]+,\s*[A-Z]\.\s*(?:[A-Z]\.\s*)?)+',
    r'[A-Z]\.\s*[A-Z]\.\s*[A-Z][a-z]+',          # J. A. Smith

    # === PDF artifacts ===
    r'This journal is © [^.]+\.',
    r'Downloaded by [^.]+\.',
    r'Published on [^.]+\.',
    r'View Article Online',
    r'©\s*\d{4}[^.]*\.',

    # === Page numbers ===
    r'\b[Pp]p?\.\s*\d+[-–]?\d*',
    r'\b\d{4}\s*[-–]\s*\d{4}\b',                  # 1234-5678 page ranges

    # === Supplemental ===
    r'(?:see\s+)?(?:the\s+)?[Ss]upplemental\s+[Mm]aterial[s]?',
    r'(?:in\s+)?[Ss]upporting\s+[Ii]nformation',
    r'\(data\s+not\s+shown\)',

    # === Methods artifacts ===
    r'Materials\s+and\s+[Mm]ethods',
    r'Experimental\s+[Pp]rocedures',

    # === Numeric table artifacts ===
    r'\b\d+\s+\d+\s+\d+\s+\d+\b',                 # Sequences of numbers
    r'\d+\.\d+\s+\d+\.\d+\s+\d+\.\d+',           # Decimal sequences
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in REMOVE_PATTERNS]


# ============================================================
# STRICT contamination detection - ANY of these = REJECT
# ============================================================

CONTAMINATION_PATTERNS = [
    # Year references
    r'\b(?:19|20)\d{2}\b',                        # ANY year 1900-2099

    # Citations
    r'\(\d+\)',                                   # (1)
    r'\[\d+\]',                                   # [1]
    r'et\s+al\.',                                 # et al.

    # Journal abbreviations
    r'(?:Nat|J|Mol|Biol|Chem|Struct|Genet|Rev|Proc|Acad|Sci)\.\s',

    # Reference formats
    r'\d+:\s*\d+[-–]\d+',                         # 87: 1295-1306
    r'10\.\d{4,}/',                               # DOI

    # Figure/Table
    r'Fig\.?\s*S?\d',
    r'Table\s*S?\d',

    # Section numbers
    r'\b\d+\.\d+\.\d+\b',                         # 3.2.1

    # Author patterns
    r'[A-Z][a-z]+,\s*[A-Z]\.\s*[A-Z]\.',         # Smith, J. A.
]

COMPILED_CONTAMINATION = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in CONTAMINATION_PATTERNS]


# ============================================================
# Instruction templates
# ============================================================

INSTRUCTION_TEMPLATES = [
    "Rewrite this scientific text with improved academic style:",
    "Enhance the clarity and precision of this scientific paragraph:",
    "Polish this academic text for publication quality:",
    "Refine the writing style of this scientific content:",
    "Improve the academic tone of this research paragraph:",
    "Paraphrase this scientific text in formal academic English:",
    "Express these scientific ideas more elegantly:",
    "Restate this research finding with better academic prose:",
    "Rephrase this text using formal scientific language:",
    "Edit this scientific paragraph for improved readability:",
    "Revise this academic text with clearer expression:",
    "Restructure this scientific writing for better flow:",
    "Improve this paragraph for academic publication:",
    "Make this scientific text more concise and precise:",
    "Strengthen the academic argumentation in this text:",
    "Clarify the scientific reasoning in this paragraph:",
    "Improve the transitions and coherence in this text:",
    "Enhance the formal tone of this scientific writing:",
    "Refine this research statement for academic clarity:",
]


def clean_text(text: str) -> str:
    """Remove all contamination patterns from text."""
    cleaned = text

    # Apply all removal patterns
    for pattern in COMPILED_PATTERNS:
        cleaned = pattern.sub(' ', cleaned)

    # Remove orphaned punctuation and parentheses
    cleaned = re.sub(r'\(\s*\)', '', cleaned)          # Empty ()
    cleaned = re.sub(r'\[\s*\]', '', cleaned)          # Empty []
    cleaned = re.sub(r'\(\s*,\s*\)', '', cleaned)      # (, )
    cleaned = re.sub(r',\s*\)', ')', cleaned)          # , ) -> )
    cleaned = re.sub(r'\(\s*,', '(', cleaned)          # (, -> (
    cleaned = re.sub(r'\(\s*\.\s*\)', '', cleaned)     # (.)
    cleaned = re.sub(r'\s*\(\s*\)', '', cleaned)       # Remaining empty ()

    # Clean up spacing and punctuation
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)          # Multiple spaces
    cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned) # Space before punctuation
    cleaned = re.sub(r'([.,])\s*\1+', r'\1', cleaned)  # Multiple punctuation
    cleaned = re.sub(r'\s*\n\s*\n\s*', '\n', cleaned)  # Multiple newlines
    cleaned = re.sub(r'^\s+', '', cleaned, flags=re.MULTILINE)  # Leading whitespace per line

    return cleaned.strip()


def has_contamination(text: str) -> bool:
    """STRICT check for ANY remaining contamination."""
    for pattern in COMPILED_CONTAMINATION:
        if pattern.search(text):
            return True
    return False


def get_contamination_details(text: str) -> list:
    """Return all contamination patterns found."""
    found = []
    for i, pattern in enumerate(COMPILED_CONTAMINATION):
        matches = pattern.findall(text)
        if matches:
            found.append((i, matches[:3]))
    return found


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def process_sample(sample: dict, max_length: int = 3000,
                   max_similarity: float = 0.75,
                   min_length: int = 100) -> dict | None:
    """Process sample with maximum aggressive cleaning."""
    inp = sample.get("input", "")
    out = sample.get("output", "")

    # Clean both
    cleaned_input = clean_text(inp)
    cleaned_output = clean_text(out)

    # Length checks
    if len(cleaned_input) < min_length or len(cleaned_output) < min_length:
        return None

    # STRICT: Reject if ANY contamination in output
    if has_contamination(cleaned_output):
        return None

    # Similarity check
    if calculate_similarity(cleaned_input, cleaned_output) > max_similarity:
        return None

    # Total length check
    if len(cleaned_input) + len(cleaned_output) > max_length:
        return None

    return {
        "instruction": random.choice(INSTRUCTION_TEMPLATES),
        "input": cleaned_input,
        "output": cleaned_output,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean training data v4 (maximum aggressive)")
    parser.add_argument("--input", default="data/training/alpaca_train.jsonl")
    parser.add_argument("--output-prefix", default="data/training/clean_v4")
    parser.add_argument("--max-length", type=int, default=3000)
    parser.add_argument("--max-similarity", type=float, default=0.75)
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading samples from: {input_path}")
    with open(input_path) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} samples")

    print("\nProcessing with MAXIMUM AGGRESSIVE cleaning...")
    cleaned = []
    rejected = {"short": 0, "contaminated": 0, "similar": 0, "long": 0}

    for sample in samples:
        result = process_sample(sample, args.max_length, args.max_similarity, args.min_length)
        if result:
            cleaned.append(result)
        else:
            # Track reason
            inp = clean_text(sample.get("input", ""))
            out = clean_text(sample.get("output", ""))
            if len(inp) < args.min_length or len(out) < args.min_length:
                rejected["short"] += 1
            elif has_contamination(out):
                rejected["contaminated"] += 1
                if args.debug and rejected["contaminated"] <= 3:
                    details = get_contamination_details(out)
                    print(f"  Contaminated: {details}")
            elif calculate_similarity(inp, out) > args.max_similarity:
                rejected["similar"] += 1
            else:
                rejected["long"] += 1

    print(f"\nResults:")
    print(f"  Original: {len(samples)}")
    print(f"  Cleaned: {len(cleaned)} ({len(cleaned)/len(samples)*100:.1f}%)")
    print(f"\nRejected:")
    for reason, count in rejected.items():
        print(f"  {reason}: {count}")

    # Verify ZERO contamination
    print("\nVERIFYING zero contamination...")
    contaminated_count = 0
    for s in cleaned:
        if has_contamination(s["output"]):
            contaminated_count += 1
            if contaminated_count <= 3:
                print(f"  Found: {get_contamination_details(s['output'])}")

    if contaminated_count > 0:
        print(f"  WARNING: {contaminated_count} still contaminated!")
    else:
        print(f"  ✓ VERIFIED: ZERO contamination!")

    # Split and save
    random.seed(42)
    random.shuffle(cleaned)
    val_size = int(len(cleaned) * args.val_split)
    train = cleaned[val_size:]
    val = cleaned[:val_size]

    train_path = Path(f"{args.output_prefix}_train.jsonl")
    val_path = Path(f"{args.output_prefix}_val.jsonl")

    with open(train_path, 'w') as f:
        for s in train:
            f.write(json.dumps(s) + '\n')
    with open(val_path, 'w') as f:
        for s in val:
            f.write(json.dumps(s) + '\n')

    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train)} samples)")
    print(f"  Val: {val_path} ({len(val)} samples)")

    return 0


if __name__ == "__main__":
    exit(main())
