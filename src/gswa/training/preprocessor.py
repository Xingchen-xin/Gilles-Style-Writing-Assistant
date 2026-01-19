"""
Data Preprocessor Module - Handles long sequence splitting.

Provides:
- Task-aware text splitting (paragraph, section, sentence)
- Token estimation without loading tokenizer
- Statistics reporting (min/median/p90/p99/max)
- Markdown and JSON reports
"""

import json
import re
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any, Tuple
from datetime import datetime


@dataclass
class PreprocessStats:
    """Statistics from preprocessing."""
    # Input stats
    input_file: str = ""
    total_entries_before: int = 0
    format_detected: str = ""

    # Token distribution before
    min_tokens_before: int = 0
    max_tokens_before: int = 0
    mean_tokens_before: float = 0.0
    median_tokens_before: float = 0.0
    p90_tokens_before: int = 0
    p95_tokens_before: int = 0
    p99_tokens_before: int = 0

    # Truncation analysis before
    over_512_before: int = 0
    over_1024_before: int = 0
    over_2048_before: int = 0
    over_4096_before: int = 0
    truncation_pct_before: float = 0.0

    # Output stats
    output_file: str = ""
    total_entries_after: int = 0

    # Token distribution after
    min_tokens_after: int = 0
    max_tokens_after: int = 0
    mean_tokens_after: float = 0.0
    median_tokens_after: float = 0.0
    p90_tokens_after: int = 0
    p95_tokens_after: int = 0
    p99_tokens_after: int = 0

    # Truncation analysis after
    over_512_after: int = 0
    over_1024_after: int = 0
    over_2048_after: int = 0
    over_4096_after: int = 0
    truncation_pct_after: float = 0.0

    # Processing stats
    expansion_ratio: float = 1.0
    chunks_per_entry: Dict[int, int] = field(default_factory=dict)  # {num_chunks: count}
    max_chunks_from_single: int = 0
    processing_time_sec: float = 0.0

    # Configuration
    target_max_tokens: int = 0
    overlap_tokens: int = 0
    split_strategy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str):
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DataPreprocessor:
    """Preprocesses training data to handle long sequences."""

    # Alpaca format template
    ALPACA_TEMPLATE = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )

    def __init__(
        self,
        max_tokens: int = 2048,
        overlap_tokens: int = 50,
        split_strategy: str = "auto"
    ):
        """Initialize preprocessor.

        Args:
            max_tokens: Target maximum tokens per sample
            overlap_tokens: Token overlap between chunks
            split_strategy: "auto", "paragraph", "sentence", or "token"
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.split_strategy = split_strategy
        self.stats = PreprocessStats()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count without loading a tokenizer.

        Uses average of word-based and character-based estimates.
        Conservative for most tokenizers.
        """
        if not text:
            return 0

        words = len(text.split())
        chars = len(text)

        # Most tokenizers: ~1.3 tokens per word, ~4 chars per token
        word_estimate = words * 1.3
        char_estimate = chars / 4

        return int((word_estimate + char_estimate) / 2)

    def analyze_data(self, input_file: str) -> PreprocessStats:
        """Analyze training data and return statistics.

        Args:
            input_file: Path to input JSONL file

        Returns:
            PreprocessStats with analysis results
        """
        stats = PreprocessStats()
        stats.input_file = input_file
        stats.target_max_tokens = self.max_tokens
        stats.overlap_tokens = self.overlap_tokens
        stats.split_strategy = self.split_strategy

        token_counts = []
        format_detected = None

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                    stats.total_entries_before += 1

                    # Detect format and extract text
                    text, fmt = self._extract_text(entry)
                    if format_detected is None:
                        format_detected = fmt

                    tokens = self.estimate_tokens(text)
                    token_counts.append(tokens)

                except json.JSONDecodeError:
                    continue

        stats.format_detected = format_detected or "unknown"

        if token_counts:
            self._calculate_distribution(token_counts, stats, "before")

        return stats

    def _extract_text(self, entry: Dict) -> Tuple[str, str]:
        """Extract full text from an entry and detect format."""
        if 'text' in entry:
            return entry['text'], "completion"
        elif 'instruction' in entry:
            text = self.ALPACA_TEMPLATE.format(
                instruction=entry.get('instruction', ''),
                input=entry.get('input', ''),
                output=entry.get('output', '')
            )
            return text, "alpaca"
        elif 'conversations' in entry:
            text = ' '.join(c.get('value', '') for c in entry.get('conversations', []))
            return text, "sharegpt"
        else:
            return str(entry), "unknown"

    def _calculate_distribution(
        self,
        counts: List[int],
        stats: PreprocessStats,
        suffix: str
    ):
        """Calculate distribution statistics."""
        sorted_counts = sorted(counts)
        n = len(sorted_counts)

        setattr(stats, f"min_tokens_{suffix}", sorted_counts[0])
        setattr(stats, f"max_tokens_{suffix}", sorted_counts[-1])
        setattr(stats, f"mean_tokens_{suffix}", sum(sorted_counts) / n)
        setattr(stats, f"median_tokens_{suffix}", sorted_counts[n // 2])
        setattr(stats, f"p90_tokens_{suffix}", sorted_counts[int(n * 0.90)])
        setattr(stats, f"p95_tokens_{suffix}", sorted_counts[int(n * 0.95)])
        setattr(stats, f"p99_tokens_{suffix}", sorted_counts[int(n * 0.99)])

        # Count over thresholds
        for threshold in [512, 1024, 2048, 4096]:
            count = sum(1 for c in sorted_counts if c > threshold)
            setattr(stats, f"over_{threshold}_{suffix}", count)

        # Calculate truncation percentage at target
        over_target = sum(1 for c in sorted_counts if c > self.max_tokens)
        setattr(stats, f"truncation_pct_{suffix}", (over_target / n * 100) if n > 0 else 0)

    def preprocess(
        self,
        input_file: str,
        output_file: str,
        progress_callback=None
    ) -> PreprocessStats:
        """Preprocess training data by splitting long sequences.

        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            progress_callback: Optional callback(current, total)

        Returns:
            PreprocessStats with full statistics
        """
        import time
        start_time = time.time()

        # First analyze
        self.stats = self.analyze_data(input_file)
        self.stats.output_file = output_file

        token_counts_after = []
        chunks_per_entry = {}
        total_written = 0

        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:

            for i, line in enumerate(f_in):
                if not line.strip():
                    continue

                if progress_callback and i % 100 == 0:
                    progress_callback(i, self.stats.total_entries_before)

                try:
                    entry = json.loads(line)

                    # Process entry
                    chunks = list(self._process_entry(entry))
                    num_chunks = len(chunks)

                    # Track chunks per entry
                    chunks_per_entry[num_chunks] = chunks_per_entry.get(num_chunks, 0) + 1

                    for chunk in chunks:
                        f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                        total_written += 1

                        # Track output tokens
                        text, _ = self._extract_text(chunk)
                        token_counts_after.append(self.estimate_tokens(text))

                except json.JSONDecodeError:
                    continue

        # Calculate output statistics
        self.stats.total_entries_after = total_written
        if token_counts_after:
            self._calculate_distribution(token_counts_after, self.stats, "after")

        # Processing stats
        self.stats.chunks_per_entry = chunks_per_entry
        self.stats.max_chunks_from_single = max(chunks_per_entry.keys()) if chunks_per_entry else 0
        self.stats.expansion_ratio = (
            self.stats.total_entries_after / self.stats.total_entries_before
            if self.stats.total_entries_before > 0 else 1.0
        )
        self.stats.processing_time_sec = time.time() - start_time

        return self.stats

    def _process_entry(self, entry: Dict) -> Generator[Dict, None, None]:
        """Process a single entry, yielding one or more chunks."""
        if 'text' in entry:
            yield from self._process_completion(entry)
        elif 'instruction' in entry:
            yield from self._process_alpaca(entry)
        elif 'conversations' in entry:
            yield from self._process_sharegpt(entry)
        else:
            yield entry

    def _process_completion(self, entry: Dict) -> Generator[Dict, None, None]:
        """Process completion format entry."""
        text = entry.get('text', '')
        tokens = self.estimate_tokens(text)

        if tokens <= self.max_tokens:
            yield entry
            return

        # Split text
        chunks = self._split_text(text)
        for chunk in chunks:
            yield {'text': chunk}

    def _process_alpaca(self, entry: Dict) -> Generator[Dict, None, None]:
        """Process Alpaca format entry."""
        instruction = entry.get('instruction', '')
        input_text = entry.get('input', '')
        output_text = entry.get('output', '')

        # Calculate tokens for instruction+input (fixed overhead)
        overhead = self.estimate_tokens(f"{instruction} {input_text}")
        available_for_output = self.max_tokens - overhead - 100  # Buffer

        if available_for_output < 100:
            # Instruction itself is too long, yield as-is
            yield entry
            return

        output_tokens = self.estimate_tokens(output_text)
        if output_tokens <= available_for_output:
            yield entry
            return

        # Split the output
        output_chunks = self._split_text(output_text, available_for_output)

        for i, chunk in enumerate(output_chunks):
            new_entry = {
                'instruction': instruction if i == 0 else f"{instruction} (continued, part {i+1})",
                'input': input_text if i == 0 else '',
                'output': chunk,
            }
            yield new_entry

    def _process_sharegpt(self, entry: Dict) -> Generator[Dict, None, None]:
        """Process ShareGPT format entry."""
        conversations = entry.get('conversations', [])
        total_tokens = sum(
            self.estimate_tokens(c.get('value', ''))
            for c in conversations
        )

        if total_tokens <= self.max_tokens:
            yield entry
            return

        # Split at conversation boundaries
        current_conv = []
        current_tokens = 0

        for conv in conversations:
            conv_tokens = self.estimate_tokens(conv.get('value', ''))

            if current_tokens + conv_tokens > self.max_tokens and current_conv:
                yield {'conversations': current_conv}
                current_conv = []
                current_tokens = 0

            current_conv.append(conv)
            current_tokens += conv_tokens

        if current_conv:
            yield {'conversations': current_conv}

    def _split_text(self, text: str, max_tokens: int = None) -> List[str]:
        """Split text using the configured strategy."""
        max_tokens = max_tokens or self.max_tokens

        if self.split_strategy == "auto":
            return self._split_auto(text, max_tokens)
        elif self.split_strategy == "paragraph":
            return self._split_by_paragraph(text, max_tokens)
        elif self.split_strategy == "sentence":
            return self._split_by_sentence(text, max_tokens)
        else:  # token
            return self._split_by_tokens(text, max_tokens)

    def _split_auto(self, text: str, max_tokens: int) -> List[str]:
        """Automatically choose best splitting strategy."""
        # Check for paragraph breaks
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) > 1:
            return self._split_by_paragraph(text, max_tokens)

        # Fall back to sentence splitting
        return self._split_by_sentence(text, max_tokens)

    def _split_by_paragraph(self, text: str, max_tokens: int) -> List[str]:
        """Split by paragraphs, grouping them to fit max_tokens."""
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.estimate_tokens(para)

            # If single paragraph exceeds limit, split it further
            if para_tokens > max_tokens:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                # Split paragraph by sentences
                sub_chunks = self._split_by_sentence(para, max_tokens)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + para_tokens > max_tokens:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))

                # Overlap: keep last paragraph if small enough
                if self.overlap_tokens > 0 and current_chunk:
                    last_para = current_chunk[-1]
                    if self.estimate_tokens(last_para) < self.overlap_tokens:
                        current_chunk = [last_para]
                        current_tokens = self.estimate_tokens(last_para)
                    else:
                        current_chunk = []
                        current_tokens = 0
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(para)
            current_tokens += para_tokens

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _split_by_sentence(self, text: str, max_tokens: int) -> List[str]:
        """Split by sentences, grouping them to fit max_tokens."""
        # Sentence boundary pattern
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*\n'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return [text]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.estimate_tokens(sentence)

            # If single sentence exceeds limit, split by tokens
            if sent_tokens > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0

                sub_chunks = self._split_by_tokens(sentence, max_tokens)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + sent_tokens > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))

                # Overlap
                if self.overlap_tokens > 0 and current_chunk:
                    last_sent = current_chunk[-1]
                    if self.estimate_tokens(last_sent) < self.overlap_tokens:
                        current_chunk = [last_sent]
                        current_tokens = self.estimate_tokens(last_sent)
                    else:
                        current_chunk = []
                        current_tokens = 0
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(sentence)
            current_tokens += sent_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _split_by_tokens(self, text: str, max_tokens: int) -> List[str]:
        """Split by word count (token approximation)."""
        words = text.split()
        # Estimate words per chunk (conservative)
        words_per_chunk = int(max_tokens * 0.75)

        chunks = []
        for i in range(0, len(words), words_per_chunk):
            # Handle overlap
            start = max(0, i - int(self.overlap_tokens * 0.75)) if i > 0 else 0
            chunk = ' '.join(words[start:i + words_per_chunk])
            chunks.append(chunk)

        return chunks if chunks else [text]

    def generate_report(self, stats: PreprocessStats = None) -> str:
        """Generate a markdown report of preprocessing results."""
        stats = stats or self.stats

        report = f"""# Data Preprocessing Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Total Entries | {stats.total_entries_before:,} | {stats.total_entries_after:,} |
| Expansion Ratio | - | {stats.expansion_ratio:.2f}x |
| Max Tokens | {stats.max_tokens_before:,} | {stats.max_tokens_after:,} |
| Truncation % | {stats.truncation_pct_before:.1f}% | {stats.truncation_pct_after:.1f}% |

## Configuration

- **Target max_tokens**: {stats.target_max_tokens}
- **Overlap tokens**: {stats.overlap_tokens}
- **Split strategy**: {stats.split_strategy}
- **Format detected**: {stats.format_detected}

## Token Distribution

### Before Preprocessing

| Statistic | Value |
|-----------|-------|
| Minimum | {stats.min_tokens_before:,} |
| Maximum | {stats.max_tokens_before:,} |
| Mean | {stats.mean_tokens_before:,.0f} |
| Median | {stats.median_tokens_before:,.0f} |
| P90 | {stats.p90_tokens_before:,} |
| P95 | {stats.p95_tokens_before:,} |
| P99 | {stats.p99_tokens_before:,} |

### After Preprocessing

| Statistic | Value |
|-----------|-------|
| Minimum | {stats.min_tokens_after:,} |
| Maximum | {stats.max_tokens_after:,} |
| Mean | {stats.mean_tokens_after:,.0f} |
| Median | {stats.median_tokens_after:,.0f} |
| P90 | {stats.p90_tokens_after:,} |
| P95 | {stats.p95_tokens_after:,} |
| P99 | {stats.p99_tokens_after:,} |

## Truncation Analysis

Sequences exceeding thresholds:

| Threshold | Before | After |
|-----------|--------|-------|
| > 512 | {stats.over_512_before:,} ({stats.over_512_before/max(1,stats.total_entries_before)*100:.1f}%) | {stats.over_512_after:,} ({stats.over_512_after/max(1,stats.total_entries_after)*100:.1f}%) |
| > 1024 | {stats.over_1024_before:,} ({stats.over_1024_before/max(1,stats.total_entries_before)*100:.1f}%) | {stats.over_1024_after:,} ({stats.over_1024_after/max(1,stats.total_entries_after)*100:.1f}%) |
| > 2048 | {stats.over_2048_before:,} ({stats.over_2048_before/max(1,stats.total_entries_before)*100:.1f}%) | {stats.over_2048_after:,} ({stats.over_2048_after/max(1,stats.total_entries_after)*100:.1f}%) |
| > 4096 | {stats.over_4096_before:,} ({stats.over_4096_before/max(1,stats.total_entries_before)*100:.1f}%) | {stats.over_4096_after:,} ({stats.over_4096_after/max(1,stats.total_entries_after)*100:.1f}%) |

## Chunk Distribution

How many chunks were generated per original entry:

| Chunks | Count |
|--------|-------|
"""
        for num_chunks in sorted(stats.chunks_per_entry.keys()):
            count = stats.chunks_per_entry[num_chunks]
            pct = count / max(1, stats.total_entries_before) * 100
            report += f"| {num_chunks} | {count:,} ({pct:.1f}%) |\n"

        report += f"""
## Files

- **Input**: `{stats.input_file}`
- **Output**: `{stats.output_file}`

## Processing Time

{stats.processing_time_sec:.2f} seconds
"""
        return report

    def save_report(self, path: str, stats: PreprocessStats = None):
        """Save markdown report to file."""
        report = self.generate_report(stats)
        with open(path, 'w') as f:
            f.write(report)


def preprocess_for_training(
    input_file: str,
    output_file: str,
    max_tokens: int = 2048,
    overlap_tokens: int = 50,
    split_strategy: str = "auto"
) -> PreprocessStats:
    """Convenience function to preprocess training data.

    Args:
        input_file: Path to input JSONL
        output_file: Path to output JSONL
        max_tokens: Maximum tokens per sample
        overlap_tokens: Overlap between chunks
        split_strategy: "auto", "paragraph", "sentence", or "token"

    Returns:
        PreprocessStats with results
    """
    preprocessor = DataPreprocessor(max_tokens, overlap_tokens, split_strategy)
    return preprocessor.preprocess(input_file, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess training data")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--max-tokens", "-m", type=int, default=2048)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--strategy", choices=["auto", "paragraph", "sentence", "token"], default="auto")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--report", help="Save markdown report to file")

    args = parser.parse_args()

    preprocessor = DataPreprocessor(args.max_tokens, args.overlap, args.strategy)

    if args.analyze_only:
        stats = preprocessor.analyze_data(args.input)
        print(preprocessor.generate_report(stats))
    else:
        output = args.output or args.input.replace('.jsonl', '.preprocessed.jsonl')
        stats = preprocessor.preprocess(args.input, output)
        print(preprocessor.generate_report(stats))

        if args.report:
            preprocessor.save_report(args.report, stats)
