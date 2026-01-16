#!/usr/bin/env python3
"""
Training Data Preprocessor - Splits long sequences to prevent OOM errors.

This script analyzes and preprocesses training data to ensure sequences
fit within the model's context window without truncation.

Key Features:
1. Analyzes sequence length distribution
2. Splits long sequences intelligently at sentence boundaries
3. Reports statistics before/after preprocessing
4. Auto-detects optimal max_seq_length based on hardware

Usage:
    # Analyze data without modifying
    python scripts/preprocess_training_data.py --analyze

    # Split sequences to fit within 2048 tokens
    python scripts/preprocess_training_data.py --max-tokens 2048

    # Auto-detect based on hardware
    python scripts/preprocess_training_data.py --auto
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Generator


# ==============================================================================
# Token Estimation (without loading tokenizer for speed)
# ==============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count without loading a tokenizer.
    Uses a conservative estimate of ~4 characters per token for English.
    For mixed content with special chars, this is more accurate.
    """
    # More accurate estimation based on word boundaries and punctuation
    words = len(text.split())
    chars = len(text)

    # Average between word-based and char-based estimates
    # Most tokenizers: ~1.3 tokens per word, ~4 chars per token
    word_estimate = words * 1.3
    char_estimate = chars / 4

    return int((word_estimate + char_estimate) / 2)


def get_accurate_token_count(text: str, tokenizer=None) -> int:
    """Get accurate token count using a tokenizer if available."""
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass
    return estimate_tokens(text)


# ==============================================================================
# Text Splitting
# ==============================================================================

def split_at_sentence_boundary(text: str, max_tokens: int, overlap_tokens: int = 50) -> list[str]:
    """
    Split text at sentence boundaries to fit within max_tokens.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Tokens to overlap between chunks for context

    Returns:
        List of text chunks
    """
    if estimate_tokens(text) <= max_tokens:
        return [text]

    # Split into sentences
    # This regex handles most academic writing patterns
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n'
    sentences = re.split(sentence_pattern, text)

    # Filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        # If single sentence exceeds max, we need to split it
        if sentence_tokens > max_tokens:
            # Save current chunk first
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split the long sentence by phrases
            phrase_chunks = split_long_sentence(sentence, max_tokens)
            chunks.extend(phrase_chunks)
            continue

        # Check if adding this sentence exceeds limit
        if current_tokens + sentence_tokens > max_tokens:
            # Save current chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))

                # Keep last sentence for overlap/context
                if overlap_tokens > 0 and len(current_chunk) > 1:
                    # Start new chunk with last sentence for context
                    last_sentence = current_chunk[-1]
                    if estimate_tokens(last_sentence) < overlap_tokens:
                        current_chunk = [last_sentence]
                        current_tokens = estimate_tokens(last_sentence)
                    else:
                        current_chunk = []
                        current_tokens = 0
                else:
                    current_chunk = []
                    current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    """Split a very long sentence by phrase boundaries."""
    # Try splitting by semicolons, commas, or conjunctions
    phrase_patterns = [
        r';\s*',           # Semicolon
        r',\s+(?=and|or|but|however|therefore|moreover)\s*',  # Comma before conjunction
        r'\s+(?=and|or|but)\s+',  # Conjunctions
        r',\s*',           # Any comma (last resort)
    ]

    for pattern in phrase_patterns:
        parts = re.split(pattern, sentence)
        if len(parts) > 1:
            chunks = []
            current = []
            current_tokens = 0

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                part_tokens = estimate_tokens(part)
                if current_tokens + part_tokens > max_tokens:
                    if current:
                        chunks.append(' '.join(current))
                    current = [part]
                    current_tokens = part_tokens
                else:
                    current.append(part)
                    current_tokens += part_tokens

            if current:
                chunks.append(' '.join(current))

            if chunks and all(estimate_tokens(c) <= max_tokens for c in chunks):
                return chunks

    # Last resort: split by word count
    words = sentence.split()
    words_per_chunk = max_tokens * 3 // 4  # Conservative estimate

    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk = ' '.join(words[i:i + words_per_chunk])
        chunks.append(chunk)

    return chunks


# ==============================================================================
# Data Processing
# ==============================================================================

def analyze_training_data(input_file: str, tokenizer=None) -> dict:
    """
    Analyze training data and return statistics.

    Returns:
        Dictionary with statistics including:
        - total_entries
        - token_distribution (min, max, mean, median, percentiles)
        - long_sequences (count over various thresholds)
    """
    stats = {
        'total_entries': 0,
        'token_counts': [],
        'format': 'unknown',
    }

    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
                stats['total_entries'] += 1

                # Detect format and extract text
                if 'text' in entry:
                    text = entry['text']
                    stats['format'] = 'completion'
                elif 'instruction' in entry:
                    text = f"{entry.get('instruction', '')} {entry.get('input', '')} {entry.get('output', '')}"
                    stats['format'] = 'alpaca'
                elif 'conversations' in entry:
                    text = ' '.join(c.get('value', '') for c in entry.get('conversations', []))
                    stats['format'] = 'sharegpt'
                else:
                    text = str(entry)

                tokens = get_accurate_token_count(text, tokenizer)
                stats['token_counts'].append(tokens)

            except json.JSONDecodeError:
                continue

    if stats['token_counts']:
        counts = sorted(stats['token_counts'])
        n = len(counts)

        stats['min_tokens'] = counts[0]
        stats['max_tokens'] = counts[-1]
        stats['mean_tokens'] = sum(counts) / n
        stats['median_tokens'] = counts[n // 2]
        stats['p90_tokens'] = counts[int(n * 0.9)]
        stats['p95_tokens'] = counts[int(n * 0.95)]
        stats['p99_tokens'] = counts[int(n * 0.99)]

        # Count sequences over various thresholds
        thresholds = [512, 1024, 2048, 4096]
        stats['over_threshold'] = {}
        for t in thresholds:
            count = sum(1 for c in counts if c > t)
            stats['over_threshold'][t] = count

    return stats


def process_entry(entry: dict, max_tokens: int, format_type: str) -> Generator[dict, None, None]:
    """
    Process a single training entry, splitting if necessary.

    Yields:
        One or more entries with text fitting within max_tokens
    """
    # Extract text based on format
    if format_type == 'completion' or 'text' in entry:
        text = entry.get('text', '')
        chunks = split_at_sentence_boundary(text, max_tokens)

        for chunk in chunks:
            yield {'text': chunk}

    elif format_type == 'alpaca' or 'instruction' in entry:
        # For instruction format, we need to be more careful
        instruction = entry.get('instruction', '')
        input_text = entry.get('input', '')
        output_text = entry.get('output', '')

        # Instruction and input are usually short, output might be long
        inst_input_tokens = estimate_tokens(f"{instruction} {input_text}")
        max_output_tokens = max_tokens - inst_input_tokens - 50  # Buffer

        if max_output_tokens < 100:
            # Instruction itself is too long, skip or truncate
            yield entry
            return

        output_chunks = split_at_sentence_boundary(output_text, max_output_tokens)

        for i, chunk in enumerate(output_chunks):
            new_entry = {
                'instruction': instruction if i == 0 else f"{instruction} (continued)",
                'input': input_text if i == 0 else '',
                'output': chunk,
            }
            yield new_entry

    elif format_type == 'sharegpt' or 'conversations' in entry:
        # For conversation format, try to keep conversations together
        conversations = entry.get('conversations', [])
        total_tokens = sum(estimate_tokens(c.get('value', '')) for c in conversations)

        if total_tokens <= max_tokens:
            yield entry
        else:
            # Split at conversation turn boundaries
            current_conv = []
            current_tokens = 0

            for conv in conversations:
                conv_tokens = estimate_tokens(conv.get('value', ''))

                if current_tokens + conv_tokens > max_tokens and current_conv:
                    yield {'conversations': current_conv}
                    current_conv = []
                    current_tokens = 0

                current_conv.append(conv)
                current_tokens += conv_tokens

            if current_conv:
                yield {'conversations': current_conv}
    else:
        yield entry


def preprocess_training_data(
    input_file: str,
    output_file: str,
    max_tokens: int,
    tokenizer=None,
) -> dict:
    """
    Preprocess training data by splitting long sequences.

    Returns:
        Statistics about the preprocessing
    """
    # First analyze the data
    print(f"\n  Analyzing: {input_file}")
    stats_before = analyze_training_data(input_file, tokenizer)

    # Detect format
    format_type = stats_before['format']
    print(f"  Format detected: {format_type}")
    print(f"  Total entries: {stats_before['total_entries']}")
    print(f"  Max tokens: {stats_before['max_tokens']}")
    print(f"  Over {max_tokens} tokens: {stats_before['over_threshold'].get(max_tokens, '?')}")

    # Process entries
    print(f"\n  Processing with max_tokens={max_tokens}...")

    entries_processed = 0
    entries_written = 0

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
                entries_processed += 1

                for processed_entry in process_entry(entry, max_tokens, format_type):
                    f_out.write(json.dumps(processed_entry, ensure_ascii=False) + '\n')
                    entries_written += 1

            except json.JSONDecodeError:
                continue

    # Analyze output
    stats_after = analyze_training_data(output_file, tokenizer)

    return {
        'entries_before': entries_processed,
        'entries_after': entries_written,
        'max_tokens_before': stats_before['max_tokens'],
        'max_tokens_after': stats_after['max_tokens'],
        'expansion_ratio': entries_written / entries_processed if entries_processed > 0 else 0,
    }


# ==============================================================================
# Hardware Detection
# ==============================================================================

def get_system_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True
            )
            return int(result.stdout.strip()) / (1024 ** 3)
        elif platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
    except Exception:
        pass
    return 16.0


def get_recommended_max_tokens(memory_gb: float = None) -> int:
    """
    Get recommended max_tokens based on available memory.

    Memory guidelines for 7B parameter model:
    - 8GB:  max_seq_length 512
    - 16GB: max_seq_length 1024
    - 32GB: max_seq_length 2048
    - 64GB: max_seq_length 4096
    """
    if memory_gb is None:
        memory_gb = get_system_memory_gb()

    if memory_gb < 12:
        return 512
    elif memory_gb < 24:
        return 1024
    elif memory_gb < 48:
        return 2048
    else:
        return 4096


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess training data to prevent OOM errors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just analyze data without modifying
  python scripts/preprocess_training_data.py --analyze

  # Split long sequences to fit 1024 tokens
  python scripts/preprocess_training_data.py --max-tokens 1024

  # Auto-detect based on hardware
  python scripts/preprocess_training_data.py --auto

Memory Guidelines:
  8GB RAM  -> max_tokens 512
  16GB RAM -> max_tokens 1024
  32GB RAM -> max_tokens 2048
  64GB RAM -> max_tokens 4096
        """
    )
    parser.add_argument(
        "--input",
        default="./data/training/alpaca_train.jsonl",
        help="Input training data file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file (default: input with .preprocessed.jsonl suffix)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per sequence"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect max-tokens based on hardware"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze data, don't process"
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify input file in place (backs up original)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Training Data Preprocessor")
    print("=" * 60)

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\nERROR: Input file not found: {args.input}")
        print("\nGenerate training data first:")
        print("  make parse-corpus")
        print("  make prepare-training")
        sys.exit(1)

    # Analyze mode
    if args.analyze:
        print("\n" + "-" * 60)
        print("Analysis Mode")
        print("-" * 60)

        stats = analyze_training_data(args.input)

        print(f"\nFile: {args.input}")
        print(f"Format: {stats['format']}")
        print(f"Total entries: {stats['total_entries']}")
        print(f"\nToken Distribution:")
        print(f"  Min:    {stats.get('min_tokens', 0):,}")
        print(f"  Max:    {stats.get('max_tokens', 0):,}")
        print(f"  Mean:   {stats.get('mean_tokens', 0):,.0f}")
        print(f"  Median: {stats.get('median_tokens', 0):,.0f}")
        print(f"  P90:    {stats.get('p90_tokens', 0):,}")
        print(f"  P95:    {stats.get('p95_tokens', 0):,}")
        print(f"  P99:    {stats.get('p99_tokens', 0):,}")

        print(f"\nSequences Over Threshold:")
        for threshold, count in stats.get('over_threshold', {}).items():
            pct = count / stats['total_entries'] * 100 if stats['total_entries'] > 0 else 0
            print(f"  > {threshold:,} tokens: {count:,} ({pct:.1f}%)")

        # Recommendation
        memory_gb = get_system_memory_gb()
        recommended = get_recommended_max_tokens(memory_gb)
        over_recommended = stats.get('over_threshold', {}).get(recommended, 0)

        print(f"\n" + "-" * 60)
        print(f"Recommendation (for {memory_gb:.0f}GB RAM)")
        print("-" * 60)
        print(f"  Recommended max_tokens: {recommended}")
        print(f"  Sequences need splitting: {over_recommended}")

        if over_recommended > 0:
            print(f"\n  Run preprocessing:")
            print(f"    python scripts/preprocess_training_data.py --max-tokens {recommended}")
        else:
            print(f"\n  Your data is ready for training!")

        sys.exit(0)

    # Determine max_tokens
    if args.auto:
        memory_gb = get_system_memory_gb()
        max_tokens = get_recommended_max_tokens(memory_gb)
        print(f"\n  Detected RAM: {memory_gb:.0f}GB")
        print(f"  Auto-selected max_tokens: {max_tokens}")
    elif args.max_tokens:
        max_tokens = args.max_tokens
    else:
        print("\nERROR: Please specify --max-tokens or use --auto")
        sys.exit(1)

    # Determine output file
    if args.in_place:
        # Backup original
        backup_path = input_path.with_suffix('.jsonl.backup')
        import shutil
        shutil.copy(input_path, backup_path)
        print(f"\n  Backed up original to: {backup_path}")
        output_file = args.input
    elif args.output:
        output_file = args.output
    else:
        output_file = str(input_path.with_suffix('.preprocessed.jsonl'))

    # Process
    print("\n" + "-" * 60)
    print("Processing")
    print("-" * 60)

    result = preprocess_training_data(args.input, output_file, max_tokens)

    print("\n" + "-" * 60)
    print("Results")
    print("-" * 60)
    print(f"  Input entries:  {result['entries_before']:,}")
    print(f"  Output entries: {result['entries_after']:,}")
    print(f"  Expansion ratio: {result['expansion_ratio']:.2f}x")
    print(f"  Max tokens before: {result['max_tokens_before']:,}")
    print(f"  Max tokens after:  {result['max_tokens_after']:,}")
    print(f"\n  Output file: {output_file}")

    if not args.in_place:
        print(f"\n  To use preprocessed data, either:")
        print(f"    1. Rename to original: mv {output_file} {args.input}")
        print(f"    2. Or run with --in-place flag")


if __name__ == "__main__":
    main()
