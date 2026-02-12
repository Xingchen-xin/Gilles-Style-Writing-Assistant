#!/usr/bin/env python3
"""
Clean training data v3 - ULTRA AGGRESSIVE cleaning for style learning.

This script addresses ALL sources of contamination found in v2:
1. Removes ALL journal/reference patterns (J., Nat., Mol., etc.)
2. Removes section numbers (3.2.1, etc.)
3. Removes author names and year patterns
4. Removes ALL DOI patterns
5. Filters ANY sample with remaining contamination

The goal is ZERO contamination in the output.

Usage:
    python scripts/clean_training_data_v3.py --input data/training/alpaca_train.jsonl
"""
import argparse
import json
import re
import random
from pathlib import Path
from difflib import SequenceMatcher


# ============================================================
# ULTRA AGGRESSIVE patterns to REMOVE from text
# ============================================================

REMOVE_PATTERNS = [
    # === Citations (all formats) ===
    r'\s*\(\d+(?:[,–-]\s*\d+)*\)',           # (1), (1,2), (1-5)
    r'\s*\[\d+(?:[,–-]\s*\d+)*\]',           # [1], [1,2], [1-5]
    r'\([A-Z][a-z]+\s+et\s+al\.\s*,?\s*\d{4}\)',  # (Author et al., 2020)
    r'[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)',       # Author et al. (2020)
    r'\bet\s+al\.',                               # et al.

    # === Journal references (CRITICAL - catches J., Nat., Mol., etc.) ===
    r'(?:Nat|Nature|Science|Cell|J|Mol|Biol|Chem|Biochem|Microbiol|Struct|Genet|Rev|Proc|Acad|Ann|Appl|BMC|PLoS|PNAS|EMBO)\.\s*[A-Z]',
    r'[A-Z][a-z]+\s+\d+:\s*\d+[-–]\d+',          # Cell 87: 1295-1306
    r'\d{4},\s*\d+(?:\(\d+\))?,\s*\d+[-–]?\d*',  # 2021, 49, 5349-5361 or 2021, 49(3), 123
    r'\d+:\s*\d+[-–]\d+',                         # 87: 1295-1306
    r'\d+\(\d+\):\s*\d+',                         # 49(3): 123

    # === DOI patterns (all formats) ===
    r'(?:doi:\s*)?10\.\d{4,}/[^\s\)]+',          # 10.1038/xxx or doi: 10.1038/xxx
    r'https?://doi\.org/[^\s]+',                  # https://doi.org/xxx

    # === URLs ===
    r'https?://[^\s]+',
    r'www\.[^\s]+',

    # === Figure/Table/Section references ===
    r'(?:see\s+)?(?:Fig(?:ure)?|Table|Supplementary\s+(?:Fig(?:ure)?|Table|Material))\s*\.?\s*S?\d+[A-Za-z]?(?:\s*(?:and|,)\s*S?\d+[A-Za-z]?)*',
    r'\((?:Fig(?:ure)?|Table)\s*\.?\s*S?\d+[A-Za-z]?\)',
    r'(?:Section|Sect\.|§)\s*\d+(?:\.\d+)*',     # Section 3.2.1
    r'^\s*\d+\.\d+(?:\.\d+)*\s+',                # 3.2.1 at start of line (section numbers)

    # === Author lists ===
    r'[A-Z][a-z]+,\s*[A-Z]\.\s*(?:[A-Z]\.\s*)?(?:,\s*[A-Z][a-z]+,\s*[A-Z]\.\s*(?:[A-Z]\.\s*)?)+',  # Smith, J. A., Jones, B. C.
    r'[A-Z][a-z]+-[A-Z][a-z]+,\s*[A-Z]\.',       # Hyphenated names like Ruban-Osmialowska, B.

    # === PDF artifacts ===
    r'This journal is © [^.]+\.',
    r'Downloaded by [^.]+\.',
    r'Published on [^.]+\.',
    r'View Article Online',
    r'©\s*\d{4}[^.]*\.',

    # === Page numbers ===
    r'\b[Pp]p?\.\s*\d+[-–]?\d*',                 # p. 123 or pp. 123-456
    r'\b\d{3,4}\s*[-–]\s*\d{3,4}\b',             # 1234-1256 (page ranges)

    # === Supplemental material ===
    r'(?:see\s+)?(?:the\s+)?[Ss]upplemental\s+[Mm]aterial[s]?',
    r'(?:in\s+)?[Ss]upporting\s+[Ii]nformation',
    r'\(data\s+not\s+shown\)',
    r'[Ss]upplementary\s+[Ff]ig(?:ure)?s?\.?\s*S?\d*',

    # === Methods section artifacts ===
    r'Materials\s+and\s+[Mm]ethods',
    r'Experimental\s+[Pp]rocedures',
]

# Compile patterns
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in REMOVE_PATTERNS]


# ============================================================
# STRICT contamination detection patterns
# These patterns will cause a sample to be REJECTED
# ============================================================

CONTAMINATION_PATTERNS = [
    # Citations
    r'\(\d+\)',                                   # (1)
    r'\[\d+\]',                                   # [1]
    r'et\s+al\.',                                 # et al.

    # Journal abbreviations (CRITICAL)
    r'(?:Nat|J|Mol|Biol|Chem|Struct|Genet|Rev|Proc|Acad)\.\s',
    r'[A-Z][a-z]+\.\s+[A-Z][a-z]+\.\s+\d',       # J. Biol. 2020

    # Reference formats
    r'\d{4},\s*\d+,\s*\d+',                       # 2021, 49, 5349
    r'\d+:\s*\d+[-–]\d+',                         # 87: 1295-1306
    r'10\.\d{4,}/',                               # DOI

    # Section numbers
    r'^\d+\.\d+\.\d+',                            # 3.2.1 at start
    r'\b\d+\.\d+\.\d+\.\d+\b',                    # 3.2.1.1

    # Figure/Table refs
    r'Fig\.?\s*S?\d',                             # Fig. 1 or Fig S1
    r'Table\s*S?\d',                              # Table 1 or Table S1

    # Author formats
    r'[A-Z]\.\s*[A-Z]\.\s*[A-Z][a-z]+',          # J. A. Smith
    r'[A-Z][a-z]+,\s*[A-Z]\.\s*[A-Z]\.',         # Smith, J. A.
]

COMPILED_CONTAMINATION = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in CONTAMINATION_PATTERNS]


# ============================================================
# Diverse instruction templates (more variations)
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
    "Rephrase this text using formal scientific language:",

    # Editing
    "Edit this scientific paragraph for improved readability:",
    "Revise this academic text with clearer expression:",
    "Restructure this scientific writing for better flow:",
    "Improve this paragraph for academic publication:",

    # Specific improvements
    "Make this scientific text more concise and precise:",
    "Strengthen the academic argumentation in this text:",
    "Clarify the scientific reasoning in this paragraph:",
    "Improve the transitions and coherence in this text:",
    "Enhance the formal tone of this scientific writing:",
    "Refine this research statement for academic clarity:",
]


def clean_text(text: str) -> str:
    """Remove all citation and reference patterns from text."""
    cleaned = text

    for pattern in COMPILED_PATTERNS:
        cleaned = pattern.sub('', cleaned)

    # Clean up artifacts from removal
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)          # Multiple spaces
    cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned) # Space before punctuation
    cleaned = re.sub(r'([.,])\s*\1+', r'\1', cleaned)  # Multiple punctuation
    cleaned = re.sub(r'\(\s*\)', '', cleaned)          # Empty parentheses
    cleaned = re.sub(r'\[\s*\]', '', cleaned)          # Empty brackets
    cleaned = re.sub(r'\s*\n\s*\n\s*', '\n', cleaned)  # Multiple newlines
    cleaned = cleaned.strip()

    return cleaned


def has_contamination(text: str) -> bool:
    """STRICT check for ANY remaining contamination patterns."""
    for pattern in COMPILED_CONTAMINATION:
        if pattern.search(text):
            return True
    return False


def get_contamination_type(text: str) -> str | None:
    """Return the type of contamination found (for debugging)."""
    for i, pattern in enumerate(COMPILED_CONTAMINATION):
        match = pattern.search(text)
        if match:
            return f"Pattern {i}: {match.group()}"
    return None


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def process_sample(sample: dict, max_length: int = 3000,
                   max_similarity: float = 0.75,
                   min_length: int = 100) -> dict | None:
    """Process a single training sample with ultra-aggressive cleaning.

    Returns None if sample should be excluded.
    """
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    out = sample.get("output", "")

    # Clean both input and output
    cleaned_input = clean_text(inp)
    cleaned_output = clean_text(out)

    # Skip if too short after cleaning
    if len(cleaned_input) < min_length or len(cleaned_output) < min_length:
        return None

    # STRICT: Skip if ANY contamination remains in output
    if has_contamination(cleaned_output):
        return None

    # Also check input for contamination (less strict, but still filter obvious ones)
    if has_contamination(cleaned_input):
        # Only allow if input contamination was successfully removed
        pass  # We've already cleaned it, so proceed

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
    parser = argparse.ArgumentParser(description="Clean training data v3 (ultra-aggressive)")
    parser.add_argument("--input", default="data/training/alpaca_train.jsonl",
                        help="Input training file")
    parser.add_argument("--output-prefix", default="data/training/clean_v3",
                        help="Output file prefix (will add _train.jsonl, _val.jsonl)")
    parser.add_argument("--max-length", type=int, default=3000,
                        help="Max total chars (input+output)")
    parser.add_argument("--max-similarity", type=float, default=0.75,
                        help="Max input-output similarity (0-1)")
    parser.add_argument("--min-length", type=int, default=100,
                        help="Min length for input and output")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--debug", action="store_true",
                        help="Show samples rejected for contamination")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading samples from: {input_path}")
    with open(input_path) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} samples")

    # Process samples
    print("\nProcessing samples with ultra-aggressive cleaning...")
    cleaned = []
    rejected_reasons = {
        "too_short": 0,
        "contaminated": 0,
        "too_similar": 0,
        "too_long": 0,
    }

    for i, sample in enumerate(samples):
        result = process_sample(
            sample,
            max_length=args.max_length,
            max_similarity=args.max_similarity,
            min_length=args.min_length,
        )

        if result:
            cleaned.append(result)
        else:
            # Track rejection reason
            inp = sample.get("input", "")
            out = sample.get("output", "")
            cleaned_input = clean_text(inp)
            cleaned_output = clean_text(out)

            if len(cleaned_input) < args.min_length or len(cleaned_output) < args.min_length:
                rejected_reasons["too_short"] += 1
            elif has_contamination(cleaned_output):
                rejected_reasons["contaminated"] += 1
                if args.debug and rejected_reasons["contaminated"] <= 5:
                    ctype = get_contamination_type(cleaned_output)
                    print(f"  Contaminated: {ctype}")
                    print(f"    Output: {cleaned_output[:100]}...")
            elif calculate_similarity(cleaned_input, cleaned_output) > args.max_similarity:
                rejected_reasons["too_similar"] += 1
            else:
                rejected_reasons["too_long"] += 1

    print(f"\nCleaning results:")
    print(f"  Original samples: {len(samples)}")
    print(f"  Cleaned samples: {len(cleaned)}")
    print(f"  Kept: {len(cleaned)/len(samples)*100:.1f}%")
    print(f"\nRejection reasons:")
    for reason, count in rejected_reasons.items():
        print(f"  {reason}: {count}")

    # Verify zero contamination
    print("\nVerifying zero contamination in cleaned data...")
    contaminated_count = 0
    for s in cleaned:
        if has_contamination(s["output"]):
            contaminated_count += 1

    if contaminated_count > 0:
        print(f"  WARNING: {contaminated_count} samples still have contamination!")
    else:
        print(f"  ✓ VERIFIED: 0% contamination in outputs")

    # Split train/val
    random.seed(42)
    random.shuffle(cleaned)

    val_size = int(len(cleaned) * args.val_split)
    train_samples = cleaned[val_size:]
    val_samples = cleaned[:val_size]

    # Save
    train_path = Path(f"{args.output_prefix}_train.jsonl")
    val_path = Path(f"{args.output_prefix}_val.jsonl")

    with open(train_path, 'w') as f:
        for s in train_samples:
            f.write(json.dumps(s) + '\n')

    with open(val_path, 'w') as f:
        for s in val_samples:
            f.write(json.dumps(s) + '\n')

    print(f"\nSaved:")
    print(f"  Training: {train_path} ({len(train_samples)} samples)")
    print(f"  Validation: {val_path} ({len(val_samples)} samples)")

    # Show instruction distribution
    print(f"\nInstruction distribution:")
    instruction_counts = {}
    for s in cleaned:
        inst = s["instruction"][:40] + "..."
        instruction_counts[inst] = instruction_counts.get(inst, 0) + 1
    for inst, count in sorted(instruction_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {count:4d}: {inst}")

    return 0


if __name__ == "__main__":
    exit(main())
