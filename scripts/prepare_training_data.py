#!/usr/bin/env python3
"""
Prepare training data for fine-tuning from Gilles's corpus.

This script creates training datasets in various formats:
1. Alpaca format with style-transfer pairs (for most fine-tuning frameworks)
2. ShareGPT format (for Axolotl, etc.)
3. DPO pairs (from feedback data)

The key innovation is generating "generic" academic versions of Gilles's paragraphs
using a local LLM, then training on {generic_input → gilles_output} pairs to teach
the model Gilles's distinctive writing style.

Usage:
    # Step 1: Generate style-transfer pairs (calls local LLM)
    python scripts/prepare_training_data.py --generate-pairs --ollama-model llama3:70b

    # Step 2: Create training data from generated pairs
    python scripts/prepare_training_data.py --format alpaca --split --output ./data/training/

    # Or do both in one step:
    python scripts/prepare_training_data.py --generate-pairs --format alpaca --split

    # Other formats
    python scripts/prepare_training_data.py --format sharegpt --weighted
    python scripts/prepare_training_data.py --format dpo --from-feedback
"""
import argparse
import json
import random
import sys
import time
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


def is_reference_content(text: str) -> bool:
    """Detect if text is bibliographic reference content (reference list entries).

    Reference LIST entries (not inline citations) typically have:
    - Journal abbreviation patterns: J. Bacteriol., Mol. Gen. Genet.
    - Volume:page patterns: 4:1699-1708
    - Multiple year patterns: 1994, 2012, 2018
    - Author year patterns: Author et al. (YYYY)

    Returns True if the text appears to be from a reference/bibliography section.
    """
    import re

    # Skip very short text
    if len(text) < 100:
        return False

    indicators = 0

    # Pattern 1: Journal abbreviation format (strongest indicator)
    # Matches: J. Proteome Res., Mol. Gen. Genet., Nucleic Acids Res.
    journal_abbrev = r'[A-Z][a-z]+\.\s+[A-Z][a-z]+\.\s*[A-Z]?[a-z]*\.?'
    journal_matches = len(re.findall(journal_abbrev, text))
    if journal_matches >= 2:
        indicators += 3  # Very strong indicator
    elif journal_matches >= 1:
        indicators += 1

    # Pattern 2: Volume:page format (e.g., 4:1699-1708, 244:135-143)
    vol_page_colon = len(re.findall(r'\d+:\d+[-–]\d+', text))
    if vol_page_colon >= 2:
        indicators += 3  # Very strong indicator
    elif vol_page_colon >= 1:
        indicators += 2

    # Pattern 3: Comma-separated volume,page (e.g., 45, 123-456)
    vol_page_comma = len(re.findall(r'\d+,\s*\d+[-–]\d+', text))
    if vol_page_comma >= 2:
        indicators += 2
    elif vol_page_comma >= 1:
        indicators += 1

    # Pattern 4: High density of bare years (not in parentheses)
    # References often have: 1994. Author... 2012. Author...
    bare_years = len(re.findall(r'\b(19|20)\d{2}\.', text))
    if bare_years >= 3:
        indicators += 2

    # Pattern 5: Author et al. patterns with periods (reference style)
    # Matches: "Author S, Author B. 2018." or "Author et al. 2018."
    author_year = len(re.findall(r'[A-Z][a-z]+\s+[A-Z]{1,2}[,.].*?(19|20)\d{2}', text))
    if author_year >= 3:
        indicators += 2
    elif author_year >= 1:
        indicators += 1

    # Pattern 6: Multiple DOIs
    doi_count = len(re.findall(r'10\.\d{4,}/', text))
    if doi_count >= 1:
        indicators += 2

    # Pattern 7: Reference numbering patterns (e.g., "13. Author", "14. Author")
    ref_numbers = len(re.findall(r'\b\d{1,3}\.\s+[A-Z][a-z]+\s+[A-Z]', text))
    if ref_numbers >= 2:
        indicators += 2

    # Threshold: 3+ indicators means likely reference content
    return indicators >= 3


GENERIC_PROMPT = """You are a scientific writing simplifier. Your task is to rewrite the following paragraph in plain, generic academic English while preserving ALL scientific facts, data, and technical terms.

Rules:
- Keep the SAME scientific content, facts, organisms, methods, and results
- Remove stylistic flourishes: vivid verbs, complex subordination, discourse markers (Indeed, Interestingly, Notably, etc.)
- Use simple sentence structures (subject-verb-object)
- Replace nuanced vocabulary with common alternatives (e.g., "precocious erection" → "early development")
- Keep technical terms and abbreviations unchanged (SEM, WT, etc.)
- Output ONLY the rewritten paragraph, nothing else
- Match approximately the same length as the input
- Do NOT add any commentary, notes, or explanations

Input paragraph:
{text}

Rewritten paragraph:"""


AUGMENT_V2_PROMPT = """You are a scientific writing assistant. Rewrite the following paragraph by restructuring the sentences. Change the sentence order, combine or split sentences, and use different grammatical constructions while keeping ALL scientific facts identical.

Rules:
- Keep the SAME scientific content, facts, organisms, methods, and results
- Restructure: change sentence order, merge or split sentences
- Use different grammatical patterns (passive→active, noun phrases→clauses, etc.)
- Keep it plain academic English without stylistic flourishes
- Keep technical terms and abbreviations unchanged
- Output ONLY the rewritten paragraph, nothing else
- Do NOT add any commentary, notes, or explanations

Input paragraph:
{text}

Restructured paragraph:"""

AUGMENT_V3_PROMPT = """You are a scientific writing condenser. Rewrite the following paragraph more concisely. Keep ALL key scientific facts but use fewer words, shorter sentences, and more direct phrasing.

Rules:
- Preserve ALL key scientific facts, data, and conclusions
- Reduce length by 20-40% through concise phrasing
- Use shorter sentences and direct subject-verb-object structure
- Remove redundancy and filler phrases
- Keep technical terms and abbreviations unchanged
- Output ONLY the condensed paragraph, nothing else
- Do NOT add any commentary, notes, or explanations

Input paragraph:
{text}

Condensed paragraph:"""


def generate_generic_version(text: str, client, model: str, max_retries: int = 3) -> Optional[str]:
    """Call local LLM to generate a 'generic' version of a Gilles paragraph."""
    prompt = GENERIC_PROMPT.format(text=text)
    # qwen3 models: disable chain-of-thought for speed
    if 'qwen3' in model.lower():
        prompt = "/no_think\n" + prompt

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=len(text.split()) * 3,  # generous token budget
            )
            result = response.choices[0].message.content.strip()

            # Basic quality checks
            if len(result) < len(text) * 0.3:
                # Too short - likely a refusal or error
                continue
            if len(result) > len(text) * 3:
                # Too long - trim
                result = result[:len(text) * 2]

            # Remove any meta-commentary the model might have added
            for prefix in ["Here is", "Here's", "Rewritten:", "Output:", "Sure,"]:
                if result.startswith(prefix):
                    # Find the actual content after the first newline
                    nl = result.find('\n')
                    if nl != -1:
                        result = result[nl+1:].strip()

            return result

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    WARN: Failed after {max_retries} attempts: {e}")
                return None

    return None


def split_into_paragraphs(text: str, min_length: int = 100, max_length: int = 1500) -> list[str]:
    """Split a long text block into individual paragraphs suitable for training.

    Splits on double newlines first, then on single newlines if chunks are still
    too long, then on sentence boundaries as a last resort.
    """
    # First try splitting on double newlines
    chunks = [c.strip() for c in text.split('\n\n') if c.strip()]

    # If still too long, split on single newlines
    result = []
    for chunk in chunks:
        if len(chunk) <= max_length:
            if len(chunk) >= min_length:
                result.append(chunk)
        else:
            # Try splitting on single newlines
            sub_chunks = [s.strip() for s in chunk.split('\n') if s.strip()]
            current = ""
            for sub in sub_chunks:
                if current and len(current) + len(sub) + 1 > max_length:
                    if len(current) >= min_length:
                        result.append(current)
                    current = sub
                else:
                    current = current + " " + sub if current else sub

            # Handle remaining text - split on sentences if too long
            if current:
                if len(current) <= max_length:
                    if len(current) >= min_length:
                        result.append(current)
                else:
                    # Split on sentence boundaries
                    sentences = current.replace('. ', '.\n').split('\n')
                    buf = ""
                    for sent in sentences:
                        if buf and len(buf) + len(sent) + 1 > max_length:
                            if len(buf) >= min_length:
                                result.append(buf)
                            buf = sent
                        else:
                            buf = buf + " " + sent if buf else sent
                    if buf and len(buf) >= min_length:
                        result.append(buf)

    return result


def generate_style_pairs(
    corpus_path: str,
    pairs_path: str,
    ollama_model: str = "llama3:70b",
    ollama_url: str = "http://localhost:11434/v1",
    min_length: int = 100,
    max_length: int = 1500,
    batch_size: int = 10,
) -> list[dict]:
    """Generate {generic_input → gilles_output} pairs using local LLM.

    Processes the corpus, splits long text blocks into paragraphs, calls ollama
    to generate 'generic' versions of each paragraph, and saves pairs
    incrementally to support resume.
    """
    from openai import OpenAI

    pairs_file = Path(pairs_path)
    pairs_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing pairs for resume support
    existing_ids = set()
    existing_pairs = []
    if pairs_file.exists():
        with open(pairs_file) as f:
            for line in f:
                if line.strip():
                    pair = json.loads(line)
                    existing_ids.add(pair.get("para_id", ""))
                    existing_pairs.append(pair)
        print(f"  Resuming: {len(existing_pairs)} pairs already generated")

    # Load corpus
    paragraphs = load_corpus(corpus_path)
    print(f"  Corpus: {len(paragraphs)} text blocks")

    # Split long text blocks into individual paragraphs
    eligible = []
    split_count = 0
    for para in paragraphs:
        text = para.get("text", "").strip()
        doc_id = para.get("doc_id", "")
        orig_para_id = para.get("para_id", "")

        if len(text) <= max_length:
            # Short enough - use as-is
            chunks = [text] if len(text) >= min_length else []
        else:
            # Split into individual paragraphs
            chunks = split_into_paragraphs(text, min_length, max_length)
            if len(chunks) > 1:
                split_count += 1

        for chunk_idx, chunk in enumerate(chunks):
            para_id = f"{doc_id}_{orig_para_id}_{chunk_idx}"
            if (para_id not in existing_ids
                and chunk[0:1].isalpha()  # starts with a letter
                and chunk.count('. ') >= 1  # at least 2 sentences
                and 'http' not in chunk
                and '10.1' not in chunk[:20]  # not a DOI/reference line
                and not is_reference_content(chunk)):  # not bibliographic references
                eligible.append((para_id, {"text": chunk, "doc_id": doc_id, "para_id": f"{orig_para_id}_{chunk_idx}"}))

    print(f"  Split {split_count} long blocks into smaller paragraphs")
    print(f"  Eligible (new): {len(eligible)} paragraphs")

    if not eligible:
        print("  Nothing to generate.")
        return existing_pairs

    # Connect to ollama
    client = OpenAI(base_url=ollama_url, api_key="ollama")
    print(f"  Using model: {ollama_model} at {ollama_url}")

    # Test connection
    try:
        test = client.chat.completions.create(
            model=ollama_model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        print(f"  Connection OK: {test.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to ollama: {e}")
        print(f"  Make sure ollama is running: ollama serve")
        print(f"  And model is available: ollama pull {ollama_model}")
        sys.exit(1)

    # Process in batches
    generated = 0
    failed = 0
    total = len(eligible)

    with open(pairs_file, 'a', encoding='utf-8') as out_f:
        for i, (para_id, para) in enumerate(eligible):
            text = para.get("text", "").strip()
            doc_id = para.get("doc_id", "")

            print(f"  [{i+1}/{total}] {doc_id} ({len(text)} chars)...", end="", flush=True)

            generic = generate_generic_version(text, client, ollama_model)

            if generic:
                pair = {
                    "para_id": para_id,
                    "doc_id": doc_id,
                    "original": text,
                    "generic": generic,
                    "original_len": len(text),
                    "generic_len": len(generic),
                }
                out_f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                out_f.flush()
                existing_pairs.append(pair)
                generated += 1
                print(f" OK ({len(generic)} chars)")
            else:
                failed += 1
                print(f" FAILED")

            # Small delay to avoid overwhelming the GPU
            if (i + 1) % batch_size == 0:
                time.sleep(0.5)

    print(f"\n  Generation complete: {generated} new, {failed} failed, {len(existing_pairs)} total")
    return existing_pairs


def generate_augmented_pairs(
    pairs_path: str,
    ollama_model: str = "llama3:70b",
    ollama_url: str = "http://localhost:11434/v1",
    variants: list[str] = None,
) -> int:
    """Generate augmented variants (V2/V3) of existing style pairs.

    For each V1 pair in style_pairs.jsonl, generates additional generic inputs
    with different prompts and temperatures to improve generalization:
    - V2 (restructured, temp=0.6): Different sentence structure/order
    - V3 (condensed, temp=0.8): Shorter, more concise version

    Supports resume by tracking which para_ids already have variants.

    Returns:
        Number of new pairs generated
    """
    from openai import OpenAI

    if variants is None:
        variants = ["v2", "v3"]

    pairs_file = Path(pairs_path)
    if not pairs_file.exists():
        print("  ERROR: No V1 pairs file found. Run --generate-pairs first.")
        return 0

    # Load existing pairs and track which variants already exist
    v1_pairs = []
    existing_variant_ids = set()
    with open(pairs_file) as f:
        for line in f:
            if line.strip():
                pair = json.loads(line)
                para_id = pair.get("para_id", "")
                variant = pair.get("variant", "v1")
                if variant == "v1" or "variant" not in pair:
                    v1_pairs.append(pair)
                else:
                    existing_variant_ids.add(f"{para_id}_{variant}")

    print(f"  V1 pairs: {len(v1_pairs)}")
    print(f"  Existing variants: {len(existing_variant_ids)}")

    # Determine which variants need to be generated
    to_generate = []
    for pair in v1_pairs:
        para_id = pair["para_id"]
        for variant in variants:
            variant_id = f"{para_id}_{variant}"
            if variant_id not in existing_variant_ids:
                to_generate.append((pair, variant))

    print(f"  To generate: {len(to_generate)} variant pairs")

    if not to_generate:
        print("  All variants already generated.")
        return 0

    # Connect to ollama
    client = OpenAI(base_url=ollama_url, api_key="ollama")
    print(f"  Using model: {ollama_model} at {ollama_url}")

    # Test connection
    try:
        test = client.chat.completions.create(
            model=ollama_model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        print(f"  Connection OK: {test.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"  ERROR: Cannot connect to ollama: {e}")
        sys.exit(1)

    # Variant configuration
    variant_config = {
        "v2": {"prompt": AUGMENT_V2_PROMPT, "temperature": 0.6, "desc": "restructured"},
        "v3": {"prompt": AUGMENT_V3_PROMPT, "temperature": 0.8, "desc": "condensed"},
    }

    generated = 0
    failed = 0
    total = len(to_generate)

    with open(pairs_file, 'a', encoding='utf-8') as out_f:
        for i, (pair, variant) in enumerate(to_generate):
            original_text = pair["original"]
            para_id = pair["para_id"]
            doc_id = pair.get("doc_id", "")
            config = variant_config[variant]

            print(f"  [{i+1}/{total}] {variant}({config['desc']}) {doc_id} ({len(original_text)} chars)...",
                  end="", flush=True)

            prompt = config["prompt"].format(text=original_text)
            # qwen3 models: disable chain-of-thought
            model_lower = ollama_model.lower()
            if 'qwen3' in model_lower:
                prompt = "/no_think\n" + prompt

            # Generate variant
            result = None
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
                        model=ollama_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=config["temperature"],
                        max_tokens=len(original_text.split()) * 3,
                    )
                    result = response.choices[0].message.content.strip()

                    # Quality checks
                    if variant == "v3":
                        # Condensed: should be shorter but not too short
                        if len(result) < len(original_text) * 0.2:
                            result = None
                            continue
                    else:
                        if len(result) < len(original_text) * 0.3:
                            result = None
                            continue

                    if len(result) > len(original_text) * 3:
                        result = result[:len(original_text) * 2]

                    # Remove meta-commentary
                    for prefix in ["Here is", "Here's", "Rewritten:", "Output:", "Sure,",
                                   "Restructured:", "Condensed:"]:
                        if result.startswith(prefix):
                            nl = result.find('\n')
                            if nl != -1:
                                result = result[nl+1:].strip()
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        print(f" WARN: {e}")
                        result = None

            if result:
                augmented_pair = {
                    "para_id": para_id,
                    "doc_id": doc_id,
                    "original": original_text,
                    "generic": result,
                    "original_len": len(original_text),
                    "generic_len": len(result),
                    "variant": variant,
                }
                out_f.write(json.dumps(augmented_pair, ensure_ascii=False) + '\n')
                out_f.flush()
                generated += 1
                print(f" OK ({len(result)} chars)")
            else:
                failed += 1
                print(f" FAILED")

            # Small delay
            if (i + 1) % 10 == 0:
                time.sleep(0.5)

    print(f"\n  Augmentation complete: {generated} new variants, {failed} failed")
    return generated


def load_style_pairs(pairs_path: str) -> dict:
    """Load pre-generated style pairs, keyed by para_id (with variant suffix).

    V1 pairs (no variant field): keyed by para_id
    V2/V3 pairs (variant field): keyed by para_id_v2, para_id_v3
    This allows create_alpaca_format to find all variants for each paragraph.
    """
    pairs = {}
    pairs_file = Path(pairs_path)
    if not pairs_file.exists():
        return pairs

    with open(pairs_file) as f:
        for line in f:
            if line.strip():
                pair = json.loads(line)
                para_id = pair["para_id"]
                variant = pair.get("variant", "v1")
                if variant == "v1":
                    pairs[para_id] = pair
                else:
                    pairs[f"{para_id}_{variant}"] = pair

    return pairs


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


def create_style_prompts(section: str = None) -> list[str]:
    """Create varied prompts for multi-task style transfer training.

    Diverse prompts help the model learn writing style from multiple angles,
    improving generalization and reducing overfitting.

    Args:
        section: Optional section label (e.g., "Introduction", "Discussion")
                 to include in prompts for section-aware training.
    """
    section_hint = f" (from the {section} section)" if section else ""

    return [
        # Basic style transfer
        f"Rewrite the following scientific paragraph{section_hint} in a clear, precise academic style:",
        f"Transform this text{section_hint} into polished scientific prose:",
        f"Improve this paragraph{section_hint} for an academic research paper:",
        f"Rewrite this passage{section_hint} with better scientific clarity and flow:",

        # Publication quality
        f"Edit this scientific text{section_hint} for publication quality:",
        f"Refine this paragraph{section_hint} for a peer-reviewed journal:",
        f"Polish this research paragraph{section_hint} for journal submission:",

        # Paraphrasing
        f"Paraphrase this research paragraph{section_hint} in formal academic English:",
        f"Rewrite this scientific content{section_hint} with improved structure and clarity:",
        f"Express the same scientific ideas{section_hint} with better academic writing:",

        # Specific improvements
        f"Improve the clarity and logical flow of this scientific paragraph{section_hint}:",
        f"Enhance the academic tone and precision of this text{section_hint}:",
        f"Revise this paragraph{section_hint} to be more concise and impactful:",

        # Style matching
        f"Write this paragraph{section_hint} in a clear, authoritative scientific voice:",
        f"Rewrite this text{section_hint} to match the style of a well-written research paper:",
        f"Transform this{section_hint} into professional scientific writing:",
    ]


def create_alpaca_format(
    paragraphs: list[dict],
    style_pairs: dict = None,
    include_source: bool = False
) -> list[dict]:
    """Create Alpaca-format training data with style-transfer pairs.

    Uses pre-generated {generic → gilles} pairs for proper style transfer.
    Falls back to completion-only format (no input) if pairs are unavailable.

    Alpaca format:
    {
        "instruction": "...",
        "input": "<generic academic version>",
        "output": "<gilles original>"
    }
    """
    prompts = create_style_prompts()
    training_data = []
    pair_used = 0
    completion_used = 0
    ref_filtered = 0

    if style_pairs is None:
        style_pairs = {}

    for i, para in enumerate(paragraphs):
        text = para.get("text", "").strip()
        if not text or len(text) < 100:
            continue

        doc_id = para.get("doc_id", "")
        orig_para_id = para.get("para_id", "")

        # Split long text blocks into individual paragraphs (same as pair generation)
        if len(text) <= 1500:
            chunks = [(text, 0)]
        else:
            split_chunks = split_into_paragraphs(text, min_length=100, max_length=1500)
            chunks = [(c, idx) for idx, c in enumerate(split_chunks)]

        for chunk_text, chunk_idx in chunks:
            para_id = f"{doc_id}_{orig_para_id}_{chunk_idx}"

            # Skip reference content (second line of defense after corpus parsing)
            if is_reference_content(chunk_text):
                ref_filtered += 1
                continue

            if para_id in style_pairs:
                # V1 style-transfer pair: generic input → gilles output
                pair = style_pairs[para_id]

                # Also filter pairs that contain reference content in original
                if is_reference_content(pair["original"]):
                    ref_filtered += 1
                    continue

                entry = {
                    "instruction": random.choice(prompts),
                    "input": pair["generic"],
                    "output": pair["original"],
                }
                if include_source:
                    entry["source"] = {"doc_id": doc_id, "para_id": f"{orig_para_id}_{chunk_idx}"}
                training_data.append(entry)
                pair_used += 1

                # Also add augmented variants (v2, v3) if available
                for variant in ["v2", "v3"]:
                    variant_key = f"{para_id}_{variant}"
                    if variant_key in style_pairs:
                        v_pair = style_pairs[variant_key]
                        v_entry = {
                            "instruction": random.choice(prompts),
                            "input": v_pair["generic"],
                            "output": v_pair["original"],
                        }
                        if include_source:
                            v_entry["source"] = {"doc_id": doc_id, "para_id": f"{orig_para_id}_{chunk_idx}", "variant": variant}
                        training_data.append(v_entry)
                        pair_used += 1
            else:
                # Fallback: completion-only (no input, just learn to write like Gilles)
                entry = {
                    "instruction": "Continue writing in this scientific style:",
                    "input": "",
                    "output": chunk_text,
                }
                if include_source:
                    entry["source"] = {"doc_id": doc_id, "para_id": f"{orig_para_id}_{chunk_idx}"}
                training_data.append(entry)
                completion_used += 1

    print(f"    Style-transfer pairs: {pair_used}")
    print(f"    Completion-only (no pair): {completion_used}")
    print(f"    Reference content filtered: {ref_filtered}")
    return training_data


# Section detection heuristics based on common paragraph patterns
SECTION_KEYWORDS = {
    "Abstract": ["abstract", "summary", "in this study", "we report", "here we show", "we demonstrate"],
    "Introduction": ["introduction", "has been shown", "previous studies", "it is known", "in recent years",
                     "however, little is known", "remains unclear", "has attracted attention"],
    "Methods": ["methods", "materials and methods", "was performed", "were cultured", "was extracted",
                "protocol", "strains were", "pcr", "primers", "sequencing"],
    "Results": ["results", "we found that", "as shown in", "figure", "fig.", "table",
                "we observed", "the results show", "analysis revealed"],
    "Discussion": ["discussion", "our findings", "these results suggest", "consistent with",
                   "in contrast to", "taken together", "limitations", "future studies"],
    "Conclusion": ["conclusion", "in summary", "in conclusion", "collectively", "overall,"],
}


def detect_section(text: str) -> Optional[str]:
    """Heuristically detect the section type of a paragraph.

    Uses keyword matching on the first 200 characters to guess the section.
    Returns None if no confident match.
    """
    text_lower = text[:300].lower()

    scores = {}
    for section, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[section] = score

    if not scores:
        return None

    best = max(scores, key=scores.get)
    # Require at least 2 keyword hits for confidence (except Methods which has unique terms)
    if scores[best] >= 2 or (best == "Methods" and scores[best] >= 1):
        return best
    return None


def create_context_window_format(
    paragraphs: list[dict],
    style_pairs: dict,
    window_size: int = 3,
    section_aware: bool = True,
) -> list[dict]:
    """Create multi-paragraph context-window training data.

    Bundles consecutive paragraphs from the same document into single training
    examples. This allows the model to learn:
    - Cross-paragraph transition patterns (转折)
    - Argument progression within sections (思路)
    - Setup-payoff patterns across paragraphs (铺垫)

    Args:
        paragraphs: Corpus paragraphs
        style_pairs: Pre-generated {generic → gilles} pairs
        window_size: Number of consecutive paragraphs per example (2-5)
        section_aware: Include section labels in prompts

    Returns:
        List of Alpaca-format training entries with multi-paragraph content
    """
    training_data = []
    window_used = 0
    single_used = 0
    ref_filtered = 0

    if style_pairs is None:
        style_pairs = {}

    # Group paragraphs by document, preserving order
    # Collect all variants (V1, V2, V3) for each paragraph
    doc_paragraphs = defaultdict(list)
    for para in paragraphs:
        text = para.get("text", "").strip()
        if not text or len(text) < 100:
            continue
        doc_id = para.get("doc_id", "")
        orig_para_id = para.get("para_id", "")

        # Split long text blocks into individual paragraphs
        if len(text) <= 1500:
            chunks = [(text, 0)]
        else:
            split_chunks = split_into_paragraphs(text, min_length=100, max_length=1500)
            chunks = [(c, idx) for idx, c in enumerate(split_chunks)]

        for chunk_text, chunk_idx in chunks:
            para_id = f"{doc_id}_{orig_para_id}_{chunk_idx}"

            # Skip reference content
            if is_reference_content(chunk_text):
                ref_filtered += 1
                continue

            if para_id in style_pairs:
                pair = style_pairs[para_id]

                # Also filter pairs that contain reference content in original
                if is_reference_content(pair["original"]):
                    ref_filtered += 1
                    continue

                # Collect all available variants for this paragraph
                variants = {"v1": pair}
                for variant in ["v2", "v3"]:
                    variant_key = f"{para_id}_{variant}"
                    if variant_key in style_pairs:
                        variants[variant] = style_pairs[variant_key]
                doc_paragraphs[doc_id].append({
                    "para_id": para_id,
                    "chunk_text": chunk_text,
                    "pair": pair,
                    "variants": variants,
                })

    # Determine which variants are available across all paragraphs
    available_variants = {"v1"}
    for doc_paras in doc_paragraphs.values():
        for p in doc_paras:
            available_variants.update(p["variants"].keys())
    variant_list = sorted(available_variants)  # consistent order: v1, v2, v3

    # Create sliding-window examples from each document
    for doc_id, doc_paras in doc_paragraphs.items():
        if len(doc_paras) < 2:
            # Single paragraph: use standard format with all variants
            for p in doc_paras:
                section = detect_section(p["chunk_text"]) if section_aware else None
                prompts = create_style_prompts(section=section)
                for variant_key in variant_list:
                    if variant_key in p["variants"]:
                        v_pair = p["variants"][variant_key]
                        entry = {
                            "instruction": random.choice(prompts),
                            "input": v_pair["generic"],
                            "output": v_pair["original"],
                        }
                        training_data.append(entry)
                        single_used += 1
            continue

        # Slide window across document paragraphs
        step = max(1, window_size - 1)  # Overlap by 1 paragraph
        for start in range(0, len(doc_paras) - window_size + 1, step):
            window = doc_paras[start:start + window_size]

            # Create a window entry for each variant (V1, V2, V3)
            for variant_key in variant_list:
                # Use this variant's generic input for each paragraph in the window
                # Fall back to V1 if this variant is not available for a paragraph
                generic_parts = []
                for p in window:
                    if variant_key in p["variants"]:
                        generic_parts.append(p["variants"][variant_key]["generic"])
                    else:
                        generic_parts.append(p["variants"]["v1"]["generic"])

                # Original outputs are always the same regardless of variant
                original_parts = [p["pair"]["original"] for p in window]

                combined_generic = "\n\n".join(generic_parts)
                combined_original = "\n\n".join(original_parts)

                # Detect section from first paragraph of window
                section = detect_section(window[0]["chunk_text"]) if section_aware else None
                prompts = create_style_prompts(section=section)

                # Multi-paragraph prompt variant
                n = len(window)
                if section:
                    instruction = random.choice([
                        f"Rewrite the following {n} consecutive paragraphs (from the {section} section) in polished academic style, maintaining logical flow between them:",
                        f"Transform these {n} paragraphs ({section} section) into publication-quality scientific prose with smooth transitions:",
                        f"Improve the writing style and inter-paragraph coherence of these {n} paragraphs (from {section}):",
                        random.choice(prompts),
                    ])
                else:
                    instruction = random.choice([
                        f"Rewrite the following {n} consecutive paragraphs in polished academic style, maintaining logical flow between them:",
                        f"Transform these {n} paragraphs into publication-quality scientific prose with smooth transitions:",
                        f"Improve the writing style and inter-paragraph coherence of these {n} paragraphs:",
                        random.choice(prompts),
                    ])

                entry = {
                    "instruction": instruction,
                    "input": combined_generic,
                    "output": combined_original,
                }
                training_data.append(entry)
                window_used += 1

        # Also add remaining paragraphs that don't fill a full window (all variants)
        remaining_start = ((len(doc_paras) - window_size) // step) * step + window_size
        for p in doc_paras[remaining_start:]:
            section = detect_section(p["chunk_text"]) if section_aware else None
            prompts = create_style_prompts(section=section)
            for variant_key in variant_list:
                if variant_key in p["variants"]:
                    v_pair = p["variants"][variant_key]
                    entry = {
                        "instruction": random.choice(prompts),
                        "input": v_pair["generic"],
                        "output": v_pair["original"],
                    }
                    training_data.append(entry)
                    single_used += 1

    print(f"    Multi-paragraph windows (size={window_size}): {window_used}")
    print(f"    Single-paragraph entries: {single_used}")
    print(f"    Reference content filtered: {ref_filtered}")
    print(f"    Variants used: {', '.join(variant_list)}")
    print(f"    Section-aware prompts: {'enabled' if section_aware else 'disabled'}")
    return training_data


def create_sharegpt_format(
    paragraphs: list[dict],
    style_pairs: dict = None,
    include_source: bool = False
) -> list[dict]:
    """Create ShareGPT-format training data with style-transfer pairs.

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

    if style_pairs is None:
        style_pairs = {}

    for para in paragraphs:
        text = para.get("text", "").strip()
        if not text or len(text) < 100:
            continue

        para_id = f"{para.get('doc_id', '')}_{para.get('para_id', '')}"

        if para_id in style_pairs:
            pair = style_pairs[para_id]
            human_text = f"{random.choice(prompts)}\n\n{pair['generic']}"
            gpt_text = pair["original"]
        else:
            human_text = f"Continue writing in this scientific style:\n\n"
            gpt_text = text

        entry = {
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": gpt_text}
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
        choices=["alpaca", "context-window", "sharegpt", "completion", "contrastive", "dpo", "all"],
        default="alpaca",
        help="Output format (context-window = multi-paragraph style-enhanced mode)"
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
    # Pair generation arguments
    parser.add_argument(
        "--generate-pairs",
        action="store_true",
        help="Generate {generic -> gilles} style-transfer pairs using local LLM"
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3:70b",
        help="Ollama model for pair generation (default: llama3:70b)"
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434/v1",
        help="Ollama API URL (default: http://localhost:11434/v1)"
    )
    parser.add_argument(
        "--pairs-file",
        default="./data/training/style_pairs.jsonl",
        help="Path to style pairs file (generated/loaded)"
    )
    parser.add_argument(
        "--max-para-length",
        type=int,
        default=1500,
        help="Max paragraph length in chars for pair generation (default: 1500). "
             "Each training entry contains both generic + original, so keep this "
             "under half of max_length * 4 to avoid truncation."
    )
    parser.add_argument(
        "--augment",
        nargs="*",
        default=None,
        help="Generate augmented variants of existing V1 pairs for generalization. "
             "Specify variants: v2 (restructured, temp=0.6), v3 (condensed, temp=0.8). "
             "Default (no args): both v2 and v3. Example: --augment v2 v3"
    )
    # Style-enhanced mode arguments
    parser.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Number of consecutive paragraphs per training example in context-window "
             "format (default: 3). Teaches cross-paragraph transitions and argument flow."
    )
    parser.add_argument(
        "--section-aware",
        action="store_true",
        help="Include section labels (Introduction/Methods/Results/Discussion) in prompts. "
             "Helps the model learn section-specific style patterns."
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GSWA Training Data Preparation")
    print("=" * 60)

    # Step 1: Generate style-transfer pairs if requested
    if args.generate_pairs:
        print(f"\n--- Generating Style-Transfer Pairs ---")
        print(f"  Corpus: {args.corpus}")
        print(f"  Model: {args.ollama_model}")
        print(f"  Pairs file: {args.pairs_file}")
        generate_style_pairs(
            corpus_path=args.corpus,
            pairs_path=args.pairs_file,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            max_length=args.max_para_length,
        )

    # Step 1b: Generate augmented variants if requested
    if args.augment is not None:
        variants = args.augment if args.augment else ["v2", "v3"]
        print(f"\n--- Generating Augmented Variants ({', '.join(variants)}) ---")
        print(f"  Pairs file: {args.pairs_file}")
        print(f"  Model: {args.ollama_model}")
        generate_augmented_pairs(
            pairs_path=args.pairs_file,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            variants=variants,
        )

    # Load corpus
    print(f"\nLoading corpus from: {args.corpus}")
    paragraphs = load_corpus(args.corpus)
    print(f"  Loaded {len(paragraphs)} paragraphs")

    # Load style pairs (if available)
    style_pairs = load_style_pairs(args.pairs_file)
    if style_pairs:
        print(f"  Style pairs loaded: {len(style_pairs)} pairs from {args.pairs_file}")
    else:
        print(f"  No style pairs found at {args.pairs_file}")
        if not args.generate_pairs:
            print(f"  HINT: Run with --generate-pairs to create {'{generic -> gilles}'} pairs")

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
            data = create_alpaca_format(paragraphs, style_pairs=style_pairs)
        elif fmt == "context-window":
            data = create_context_window_format(
                paragraphs, style_pairs=style_pairs,
                window_size=args.context_window,
                section_aware=args.section_aware,
            )
        elif fmt == "sharegpt":
            data = create_sharegpt_format(paragraphs, style_pairs=style_pairs)
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

    # Show sample entry
    if data:
        print(f"\n--- Sample Entry ---")
        sample = data[0]
        if "instruction" in sample:
            print(f"  instruction: {sample['instruction'][:80]}...")
            print(f"  input: {sample['input'][:120]}..." if sample.get('input') else "  input: (empty)")
            print(f"  output: {sample['output'][:120]}...")

    print("\n" + "=" * 60)
    print("Training data preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    if not style_pairs:
        print("  0. Generate style pairs: python scripts/prepare_training_data.py --generate-pairs")
    print("  1. Review the generated training data")
    print("  2. Run fine-tuning:")
    print("     make train MODEL=Mistral")
    print("  3. Evaluate: make evaluate")


if __name__ == "__main__":
    main()
