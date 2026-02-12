#!/usr/bin/env python3
"""
Clean training data v5 - Fix PDF artifacts, citations, and fragment outputs.

This script addresses issues found in v4 and post-training evaluation:
1. PDF ligatures (fi, ff, fl, ffi, ffl) → proper ASCII
2. Fragment outputs (starting with lowercase) → filtered
3. Journal headers/footers → removed
4. Special unicode (private use area) → replaced/removed
5. Citation page numbers in output → cleaned
6. Inline citations stripped: [1,2], (Author 2024), superscript21,22
7. Section headers detected & text truncated before them
8. Reference list entries → sample filtered
9. Cross-references (see Fig/Table/S1) → stripped

Usage:
    python scripts/clean_training_data_v5.py --input data/training/clean_v4_train.jsonl
"""
import argparse
import json
import re
import random
from pathlib import Path


# ============================================================
# PDF ligature replacements
# ============================================================

LIGATURE_MAP = {
    '\ufb00': 'ff',    # ff ligature
    '\ufb01': 'fi',    # fi ligature
    '\ufb02': 'fl',    # fl ligature
    '\ufb03': 'ffi',   # ffi ligature
    '\ufb04': 'ffl',   # ffl ligature
    '\ufb05': 'st',    # st ligature (rare)
    '\ufb06': 'st',    # st ligature variant
    # Common special characters
    '\u2010': '-',     # hyphen
    '\u2011': '-',     # non-breaking hyphen
    '\u2012': '-',     # figure dash
    '\u2013': '-',     # en dash
    '\u2014': '-',     # em dash
    '\u2018': "'",     # left single quote
    '\u2019': "'",     # right single quote
    '\u201c': '"',     # left double quote
    '\u201d': '"',     # right double quote
    '\u2026': '...',   # ellipsis
    '\u00a0': ' ',     # non-breaking space
    '\u00ad': '',      # soft hyphen
    '\u200b': '',      # zero-width space
    '\u200c': '',      # zero-width non-joiner
    '\u200d': '',      # zero-width joiner
    '\ufeff': '',      # byte order mark
}

# Private use area characters (PDF artifacts) - just remove
PRIVATE_USE_PATTERN = re.compile(r'[\ue000-\uf8ff]')


# ============================================================
# Patterns to remove from text
# ============================================================

REMOVE_PATTERNS = [
    # Journal headers (case-insensitive)
    r'NAture RevIeWS.*?(?:Reviews|Microbiology)',
    r'Chemistry & Biology.*?(?:reserved|\d{3,4})',
    r'rights?\s+reserved',

    # Copyright notices (anywhere in text)
    r'©\s*\d{4}[^.]*\.?',
    r'©\s*[A-Z][^.]*\.?',
    r'©\s*Macmillan[^.]*\.?',
    r'\(c\)\s*\d{4}[^.]*',

    # Journal headers/footers with author names
    r'[A-Z]\s+[A-Z][a-z]+\s+et\s+al\s+\d+\s+The\s+\w+\s+Journal',
    r'The\s+ISME\s+Journal',
    r'The\s+EMBO\s+Journal',

    # Page numbers and journal identifiers
    r'Antibiotics,\s*\d+,\s*\d+\s+\d+\s+of\s+\d+',
    r'\d+\s+of\s+\d+\s*$',
    r'(?:mbio|asm|nature|science|cell)\.(?:asm\.)?org\s*\d*',

    # Date/volume patterns
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)/?(?:January|February|March|April|May|June|July|August|September|October|November|December)?\s+Volume\s+\d+',
    r'Volume\s+\d+\s+Issue\s+\d+',
    r'e\d{5,}-\d+',

    # Stock photo credits
    r'(?:ALAMY|GETTY|STOCK)\s+PHOTO[^.]*',
    r'FANATIC STUDIO[^.]*',

    # Article section headers that shouldn't be in prose
    r'\bSIGNIFICANCE\b',
    r'\bIMPORTANCE\b',
    r'\bCHAPTER\s+(?:ONE|TWO|THREE|FOUR|FIVE|\d+)\b',
    r'\bContents\s*\n',

    # Author affiliation patterns
    r'\d+Corresponding\s+authors?:?\s+e-mail[^.]*',
    r'e-mail\s+address:\s*\S+@\S+',
    r'\d+Shared\s+last\s+author',

    # Citation artifacts
    r'Chemistry & Biology \d+,\s*\d+[-–]\d+,.*',
    r'Nature Reviews.*Microbiology.*',
    r'\d{1,2},\s*\d{3,4}[-–]\d{3,4},.*?(?:Elsevier|Ltd|All).*',

    # Page numbers at end
    r'\b\d{3,4}\s*[-–]\s*\d{3,4}\b\s*$',

    # DOI
    r'(?:doi:?\s*)?10\.\d{4,}/\S+',

    # Publisher names
    r'Macmillan\s+Publishers[^.]*',
    r'Elsevier\s+Ltd[^.]*',

    # Table of contents patterns
    r'\d+\.\d+\s+[A-Z][^.]{5,50}\s+\d+\s*\n',

    # Evolving patterns
    r'Evolving\s+resistance\s+at\s+antibiotics[^.]*',
]

COMPILED_REMOVE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in REMOVE_PATTERNS]


# ============================================================
# Inline citation patterns to STRIP from text (not filter)
# ============================================================

CITATION_STRIP_PATTERNS = [
    # Bracketed reference numbers: [1], [1,2], [1-3], [25,26-30]
    (re.compile(r'\s*\[\d+(?:\s*[-,]\s*\d+)*\]'), ''),

    # Superscript citation numbers after sentences: text.1,2 or text21,22
    # Must NOT be followed by hyphen (which indicates chemical nomenclature like 1,2,3-trisubstituted)
    (re.compile(r'(?<=[.;,)])(\d{1,3}(?:[,]\d{1,3})*(?:[-]\d{1,3})*)(?=[\s.,;:)\]])'), ''),
    # Single superscript after period: "treat.1 The" -> "treat. The"
    (re.compile(r'(?<=[.;,)])(\d{1,2})(?=\s+[A-Z])'), ''),

    # Parenthetical author-year citations: (Author 2024), (Author et al., 2024)
    (re.compile(r'\s*\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?,?\s*\d{4}[a-z]?\)'), ''),
    # Multiple author citations: (Author 2024; Author 2023)
    (re.compile(r'\s*\([A-Z][a-z]+.*?;\s*[A-Z][a-z]+.*?\d{4}[a-z]?\)'), ''),
    # Year-only in parens right after author name: Author (2024)
    (re.compile(r'(?<=[A-Z][a-z]{2})\s*\(\d{4}[a-z]?\)'), ''),

    # Cross-references: "see Fig. 1", "see S3 Table", "see Material and methods"
    (re.compile(r'\s*\(?\bsee\s+(?:S\d+\s+)?(?:Fig(?:ure)?\.?\s*\d+|Table\s*\d+|Material[s]?\s+and\s+[Mm]ethods|Supplement\w*|Appendix|Section\s*\d*)[^)]*\)?', re.IGNORECASE), ''),

    # "reviewed in/by Author" phrases
    (re.compile(r'\s*\(?\breviewed\s+(?:in|by)\s+[A-Z][^)]*\)?'), ''),

    # Section header markers
    (re.compile(r'■'), ''),
]

# ============================================================
# Section header patterns - truncate text before these
# ============================================================

# Case-sensitive: only match ALL-CAPS headers or numbered Title Case headers
SECTION_HEADERS = re.compile(
    r'(?:'
    r'\b(?:INTRODUCTION|METHODS|RESULTS|DISCUSSION|CONCLUSIONS?|ABSTRACT|'
    r'REFERENCES|ACKNOWLEDGMENTS?|MATERIALS?\s+AND\s+METHODS|'
    r'SUPPLEMENTARY|SUPPORTING\s+INFORMATION|AUTHOR\s+CONTRIBUTIONS?|'
    r'COMPETING\s+INTERESTS?|DATA\s+AVAILABILITY|FUNDING)\b'
    r'|'
    r'^\d+\.\s+(?:Introduction|Method|Result|Discussion|Conclusion|'
    r'Abstract|Reference)'
    r')',
    re.MULTILINE
)

# ============================================================
# Reference entry detection - filter samples with these
# ============================================================

REFERENCE_ENTRY_PATTERNS = [
    # Journal abbreviation + volume:page: "Nat Chem Biol, 13:470"
    re.compile(r'(?:Nat|Sci|Chem|Biol|Rev|Rep|PNAS|Proc|Ann|PLoS|BMC|Mol|Cell|Appl|Environ|Microbiol|Biochem|Biotechnol|Genet|Genomics|J\s+Am|J\s+Biol)\s+\w+.*?\d+\s*[:(]\s*\d{3,}'),
    # "J Am Chem Soc," style
    re.compile(r'J\s+(?:Am|Biol|Nat|Org)\s+Chem\s+Soc'),
    # Numbered reference list: "134. Yang JY, Sanchez LM..."
    re.compile(r'\d{1,3}\.\s+[A-Z][a-z]+\s+[A-Z]{1,2}[,:].*?\d{4}'),
    # "Author: Title. Journal, vol:page" pattern
    re.compile(r'[A-Z][a-z]+\s+[A-Z]{1,2}[,:]\s+\w+.*?(?:Nature|Science|Cell|PNAS|J\s+\w+|Proc\s+Natl|PLoS)\s+\w+'),
]


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


def fix_ligatures(text: str) -> str:
    """Replace PDF ligatures with proper ASCII characters."""
    for ligature, replacement in LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)

    # Remove private use area characters
    text = PRIVATE_USE_PATTERN.sub('', text)

    # Remove control characters (except newline and tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text


def strip_citations(text: str) -> str:
    """Remove inline citations from text while preserving prose."""
    cleaned = text
    for pattern, replacement in CITATION_STRIP_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def truncate_at_section_header(text: str) -> str:
    """Truncate text at the first section header (INTRODUCTION, METHODS, etc.).

    Many outputs contain a good first paragraph followed by section headers
    from the original paper. We keep only the text before the first header.
    """
    match = SECTION_HEADERS.search(text)
    if match:
        # Keep only text before the section header
        truncated = text[:match.start()].strip()
        if len(truncated) >= 100:  # Only truncate if enough text remains
            return truncated
    return text


def has_reference_entries(text: str) -> bool:
    """Check if text contains bibliographic reference entries."""
    for pattern in REFERENCE_ENTRY_PATTERNS:
        if pattern.search(text):
            return True
    return False


def clean_text(text: str) -> str:
    """Clean PDF artifacts and citations from text."""
    # Fix ligatures first
    cleaned = fix_ligatures(text)

    # Remove journal headers/footers
    for pattern in COMPILED_REMOVE:
        cleaned = pattern.sub('', cleaned)

    # Strip inline citations
    cleaned = strip_citations(cleaned)

    # Truncate at section headers
    cleaned = truncate_at_section_header(cleaned)

    # Clean up artifacts
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)          # Multiple spaces
    cleaned = re.sub(r'\s+([.,;:!?])', r'\1', cleaned) # Space before punctuation
    cleaned = re.sub(r'([.,])\s*\1+', r'\1', cleaned)  # Multiple punctuation
    cleaned = re.sub(r'\(\s*\)', '', cleaned)          # Empty parentheses
    cleaned = re.sub(r'\[\s*\]', '', cleaned)          # Empty brackets
    cleaned = re.sub(r'\s*\n\s*\n\s*', '\n', cleaned)  # Multiple newlines

    # Remove trailing incomplete sentences (no ending punctuation)
    # Only if the text ends mid-sentence without proper punctuation
    cleaned = cleaned.strip()
    if cleaned and not cleaned[-1] in '.?!':
        # Find last complete sentence
        last_period = max(cleaned.rfind('. '), cleaned.rfind('.\n'), cleaned.rfind('.'))
        last_question = cleaned.rfind('?')
        last_exclaim = cleaned.rfind('!')
        last_punct = max(last_period, last_question, last_exclaim)

        if last_punct > len(cleaned) * 0.5:  # Only trim if we keep at least half
            cleaned = cleaned[:last_punct + 1].strip()

    return cleaned


def is_complete_sentence(text: str) -> bool:
    """Check if text appears to be a complete sentence/paragraph.

    Rejects:
    - Text starting with lowercase (mid-sentence fragments)
    - Text starting with conjunctions that indicate continuation
    - Text starting with artifacts (Email:, Funding, etc.)
    """
    if not text:
        return False

    first_char = text[0]

    # Must start with uppercase letter or number
    if not (first_char.isupper() or first_char.isdigit()):
        return False

    # Check for continuation words at start (allow comma/semicolon after)
    continuation_patterns = [
        r'^(and|but|or|nor|yet|so|however|therefore|thus|hence|consequently)[,;]?\s',
        r'^(moreover|furthermore|additionally|also|besides)[,;]?\s',
        r'^(nevertheless|nonetheless|otherwise|meanwhile)[,;]?\s',
        r'^(this|these|that|those|such)\s+(toxicity|suggests?|indicate)',
        r'^(in\s+contrast|in\s+addition|on\s+the\s+other\s+hand)',
    ]

    text_lower = text.lower()
    for pattern in continuation_patterns:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return False

    # Check for artifact starts
    artifact_patterns = [
        r'^email:',
        r'^funding',
        r'^abstract',
        r'^acknowledgment',
        r'^supplementary',
        r'^copyright',
        r'^doi:',
        r'^\d+\.\s+[a-z]',  # numbered list items starting lowercase
        # Article header patterns
        r'^article\s',
        r'^brief\s+communication',
        r'^received\s+\d',
        r'^accepted\s+\d',
        r'^published\s+\d',
        r'^original\s+article',
        r'^research\s+article',
        r'^review\s+article',
        r'^letter\s+to',
        r'^correspondence',
        r'^nature\s+communications',
        # Author/affiliation patterns
        r'^katherine\s',
        r'^[a-z]+\s+[a-z]+\d,\*',  # Author names with superscript
        # Figure/table captions as starts
        r'^fig(?:ure)?\.?\s*\d',
        r'^table\s*\d',
        r'^[a-d]\)',  # (A), (B), etc. figure panel labels
    ]

    for pattern in artifact_patterns:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return False

    return True


def has_structural_artifacts(text: str) -> bool:
    """Check if text contains structural artifacts from papers."""
    # Case-insensitive patterns (safe generic patterns)
    ci_patterns = [
        r'Contents\s*\n.*\d+\.\d+',  # Table of contents
        r'CHAPTER\s+(?:ONE|TWO|THREE|\d+)',
        r'Corresponding\s+authors?',
        r'Shared\s+last\s+author',
        r'The\s+ISME\s+Journal',
        r'The\s+EMBO\s+Journal',
        r'\d+\s+of\s+\d+$',  # Page numbers at end
        r'Antibiotics,\s*\d+,\s*\d+',  # Journal page refs
    ]

    for pattern in ci_patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return True

    # Case-SENSITIVE patterns — only match ALL-CAPS section headers
    # These words in lowercase/mixed case are fine in regular prose
    cs_patterns = [
        r'\bSIGNIFICANCE\b',
        r'\bIMPORTANCE\b',
        r'\bINTRODUCTION\b',
        r'\bMETHODS\b',
        r'\bRESULTS\b',
        r'\bDISCUSSION\b',
        r'\bCONCLUSIONS?\b',
        r'\bREFERENCES\b',
        r'\bACKNOWLEDGMENTS?\b',
        r'\bMATERIALS?\s+(?:AND|&)\s+METHODS\b',
        r'\bSUPPLEMENTARY\b',
        r'\bSUPPORTING\s+INFORMATION\b',
        r'\bAUTHOR\s+CONTRIBUTIONS?\b',
        # Numbered section headers (Title Case)
        r'^\d+\.\s+(?:Introduction|Methods?|Results?|Discussion)',
    ]

    for pattern in cs_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True

    return False


def has_academic_metadata(text: str) -> bool:
    """Check if text contains academic paper metadata artifacts."""
    metadata_patterns = [
        # Author name lists: "T. Dahl, N. Dowidar, S. Ferree"
        re.compile(r'(?:[A-Z][a-z]+,?\s+[A-Z]\.?\s*){3,}'),
        # Email addresses
        re.compile(r'[Ee]-?mail:\s*\S+@\S+'),
        # Article received/accepted dates
        re.compile(
            r'(?:Received|Accepted|Revised|Published)\s+\d',
            re.IGNORECASE,
        ),
        # REVIEWS header with page/volume
        re.compile(r'\bREVIEWS?\b\s+\d'),
        # "Journal of X" titles
        re.compile(r'(?:The\s+)?Journal\s+of\s+\w+'),
        # Nature Reviews/Communications
        re.compile(r'Nature\s+(?:Reviews?|Communications?)'),
        # Journal layout pipes: "| FEBRUARY | VOLUME 12"
        re.compile(
            r'\|\s*(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|'
            r'JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|VOL)',
            re.IGNORECASE,
        ),
        # Journal abbreviations at start: "J Ind Microbiol Biotechnol"
        re.compile(
            r'^[A-Z]\s+[A-Z][a-z]+\s+(?:Microbiol|Biotechnol|Biochem|'
            r'Chem|Biol|Genet|Pharmacol|Toxicol)',
        ),
        # Author names with affiliation superscripts: "Name,a Name,b"
        re.compile(
            r'[A-Z][a-z]+,\s*\[?[a-z]\]?\s+[A-Z][a-z]+\s+'
            r'[A-Z][a-z]+,\s*\[?[a-z]\]?',
        ),
    ]
    for pat in metadata_patterns:
        if pat.search(text):
            return True
    return False


def process_sample(sample: dict, min_length: int = 100) -> dict | None:
    """Process a single training sample.

    Returns None if sample should be excluded.
    """
    instruction = sample.get("instruction", "")
    inp = sample.get("input", "")
    out = sample.get("output", "")

    # Clean both input and output
    cleaned_input = clean_text(inp)
    cleaned_output = clean_text(out)

    # Skip if output is too short
    if len(cleaned_output) < min_length:
        return None

    # Skip if output is not a complete sentence
    if not is_complete_sentence(cleaned_output):
        return None

    # Skip if output contains structural artifacts
    if has_structural_artifacts(cleaned_output):
        return None

    # Skip if output contains bibliographic reference entries
    if has_reference_entries(cleaned_output):
        return None

    # Skip if output has too many "et al." (indicates heavy citation text)
    et_al_count = len(re.findall(r'et\s+al\.?', cleaned_output, re.IGNORECASE))
    if et_al_count >= 3:
        return None

    # Skip if output contains remaining academic metadata
    if has_academic_metadata(cleaned_output):
        return None

    # Also check input for completeness (less strict)
    if len(cleaned_input) < min_length:
        return None

    # Assign diverse instruction
    new_instruction = random.choice(INSTRUCTION_TEMPLATES)

    return {
        "instruction": new_instruction,
        "input": cleaned_input,
        "output": cleaned_output,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean training data v5 (fix PDF artifacts)")
    parser.add_argument("--input", default="data/training/clean_v4_train.jsonl",
                        help="Input training file")
    parser.add_argument("--output-prefix", default="data/training/clean_v5",
                        help="Output file prefix")
    parser.add_argument("--min-length", type=int, default=100,
                        help="Min length for input and output")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--debug", action="store_true",
                        help="Show rejected samples")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading samples from: {input_path}")
    with open(input_path) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} samples")

    # Analyze issues before cleaning
    print("\n=== Pre-cleaning analysis ===")
    lowercase_count = sum(1 for s in samples if s.get('output', '') and s['output'][0].islower())
    ligature_count = sum(1 for s in samples if any(c in s.get('output', '') for c in LIGATURE_MAP))
    print(f"  Lowercase starts: {lowercase_count} ({lowercase_count/len(samples)*100:.1f}%)")
    print(f"  PDF ligatures: {ligature_count} ({ligature_count/len(samples)*100:.1f}%)")

    # Process samples
    print("\n=== Processing samples ===")
    cleaned = []
    rejected_reasons = {
        "too_short": 0,
        "fragment": 0,
        "artifacts": 0,
        "reference_entries": 0,
        "et_al_heavy": 0,
        "academic_metadata": 0,
        "input_short": 0,
    }

    for sample in samples:
        result = process_sample(sample, min_length=args.min_length)

        if result:
            cleaned.append(result)
        else:
            # Track rejection reason
            out = clean_text(sample.get("output", ""))
            inp = clean_text(sample.get("input", ""))
            if len(out) < args.min_length:
                rejected_reasons["too_short"] += 1
            elif not is_complete_sentence(out):
                rejected_reasons["fragment"] += 1
                if args.debug and rejected_reasons["fragment"] <= 3:
                    print(f"  Fragment: {out[:80]}...")
            elif has_structural_artifacts(out):
                rejected_reasons["artifacts"] += 1
                if args.debug and rejected_reasons["artifacts"] <= 3:
                    print(f"  Artifact: {out[:80]}...")
            elif has_reference_entries(out):
                rejected_reasons["reference_entries"] += 1
                if args.debug and rejected_reasons["reference_entries"] <= 3:
                    print(f"  Ref entry: {out[:80]}...")
            elif len(re.findall(r'et\s+al\.?', out, re.IGNORECASE)) >= 3:
                rejected_reasons["et_al_heavy"] += 1
            elif has_academic_metadata(out):
                rejected_reasons["academic_metadata"] += 1
            elif len(inp) < args.min_length:
                rejected_reasons["input_short"] += 1
            else:
                rejected_reasons["fragment"] += 1

    print(f"\nCleaning results:")
    print(f"  Original samples: {len(samples)}")
    print(f"  Cleaned samples: {len(cleaned)}")
    print(f"  Kept: {len(cleaned)/len(samples)*100:.1f}%")
    print(f"\nRejection reasons:")
    for reason, count in rejected_reasons.items():
        print(f"  {reason}: {count}")

    # Verify cleaning
    print("\n=== Post-cleaning verification ===")
    issues = {
        "lowercase_start": 0,
        "pdf_ligatures": 0,
        "bracketed_refs": 0,
        "superscript_nums": 0,
        "section_headers": 0,
        "reference_entries": 0,
        "et_al_heavy": 0,
        "journal_names": 0,
    }
    for s in cleaned:
        out = s["output"]
        if out and out[0].islower():
            issues["lowercase_start"] += 1
        if any(c in out for c in LIGATURE_MAP):
            issues["pdf_ligatures"] += 1
        if re.search(r'\[\d+(?:[-,]\d+)*\]', out):
            issues["bracketed_refs"] += 1
        if re.search(r'(?<=[.;,)])(\d{1,3}(?:,\d{1,3})+)', out):
            issues["superscript_nums"] += 1
        if SECTION_HEADERS.search(out):
            issues["section_headers"] += 1
        if has_reference_entries(out):
            issues["reference_entries"] += 1
        if len(re.findall(r'et\s+al\.?', out, re.IGNORECASE)) >= 3:
            issues["et_al_heavy"] += 1
        if re.search(r'(?:Nature|Science|Cell|PNAS|Journal|Proc)\b.*?\d+\s*[:(]\s*\d{3,}', out):
            issues["journal_names"] += 1

    for issue, count in issues.items():
        pct = count / len(cleaned) * 100 if cleaned else 0
        status = "OK" if count == 0 else "WARN"
        print(f"  [{status}] {issue}: {count} ({pct:.2f}%)")

    all_clean = all(v == 0 for v in issues.values())
    if all_clean:
        print("  All issues fixed!")

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

    # Show sample
    print("\n=== Sample cleaned output ===")
    if cleaned:
        sample = cleaned[0]
        print(f"Instruction: {sample['instruction']}")
        print(f"Input: {sample['input'][:150]}...")
        print(f"Output: {sample['output'][:150]}...")

    return 0


if __name__ == "__main__":
    exit(main())
