#!/usr/bin/env python3
"""
Author Style Fingerprint Analyzer
==================================

Analyzes corpus to extract author writing style characteristics.
Generates a style fingerprint that can be used to:
1. Enhance prompts with author-specific guidance
2. Validate generated text matches author style
3. Train models on authentic style patterns

Usage:
    python scripts/analyze_style.py                    # Analyze corpus
    python scripts/analyze_style.py --output style.json  # Custom output
    python scripts/analyze_style.py --verbose          # Detailed output
"""
import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gswa.utils.progress import (
    Colors, ProgressBar,
    print_header, print_section, print_success, print_warning, print_error, print_info
)


# ==============================================================================
# Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_PARSED_DIR = PROJECT_ROOT / "data" / "corpus" / "parsed"
STYLE_OUTPUT_DIR = PROJECT_ROOT / "data" / "style"
DEFAULT_OUTPUT = STYLE_OUTPUT_DIR / "author_fingerprint.json"


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class SentenceStats:
    """Sentence-level statistics."""
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    std_dev: float = 0.0
    short_ratio: float = 0.0  # < 10 words
    medium_ratio: float = 0.0  # 10-25 words
    long_ratio: float = 0.0  # > 25 words


@dataclass
class VocabularyStats:
    """Vocabulary usage statistics."""
    avg_word_length: float = 0.0
    unique_ratio: float = 0.0  # type-token ratio
    top_nouns: list[str] = field(default_factory=list)
    top_verbs: list[str] = field(default_factory=list)
    top_adjectives: list[str] = field(default_factory=list)
    favorite_transitions: list[str] = field(default_factory=list)
    avoided_words: list[str] = field(default_factory=list)


@dataclass
class StructureStats:
    """Writing structure statistics."""
    passive_voice_ratio: float = 0.0
    question_ratio: float = 0.0
    hedge_frequency: float = 0.0  # may/might/could per 100 words
    citation_style: str = ""  # e.g., "(Author, Year)" or "[1]"
    avg_paragraph_sentences: float = 0.0
    list_usage_ratio: float = 0.0


@dataclass
class PhrasePatterns:
    """Common phrase patterns."""
    opening_patterns: list[str] = field(default_factory=list)
    closing_patterns: list[str] = field(default_factory=list)
    transition_patterns: list[str] = field(default_factory=list)
    emphasis_patterns: list[str] = field(default_factory=list)


@dataclass
class StyleFingerprint:
    """Complete author style fingerprint."""
    author_name: str = "Gilles"
    corpus_size: int = 0
    total_words: int = 0
    total_sentences: int = 0
    analysis_version: str = "1.0"

    sentence_stats: SentenceStats = field(default_factory=SentenceStats)
    vocabulary_stats: VocabularyStats = field(default_factory=VocabularyStats)
    structure_stats: StructureStats = field(default_factory=StructureStats)
    phrase_patterns: PhrasePatterns = field(default_factory=PhrasePatterns)

    # Style guidance for prompts
    style_rules: list[str] = field(default_factory=list)
    avoid_patterns: list[str] = field(default_factory=list)
    prefer_patterns: list[str] = field(default_factory=list)


# ==============================================================================
# Analysis Functions
# ==============================================================================

def load_corpus(corpus_path: Path) -> list[dict]:
    """Load parsed corpus."""
    corpus_file = corpus_path / "corpus.jsonl"
    if not corpus_file.exists():
        return []

    paragraphs = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    paragraphs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return paragraphs


def tokenize_words(text: str) -> list[str]:
    """Simple word tokenization."""
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Handle common abbreviations
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|et al|i\.e|e\.g|vs|etc)\.\s', r'\1<DOT> ', text)
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences if len(s.strip()) > 5]
    return sentences


def analyze_sentences(paragraphs: list[dict]) -> SentenceStats:
    """Analyze sentence-level patterns."""
    all_lengths = []

    for para in paragraphs:
        text = para.get("text", "")
        sentences = split_sentences(text)
        for sent in sentences:
            word_count = len(tokenize_words(sent))
            if word_count > 0:
                all_lengths.append(word_count)

    if not all_lengths:
        return SentenceStats()

    avg = sum(all_lengths) / len(all_lengths)
    variance = sum((l - avg) ** 2 for l in all_lengths) / len(all_lengths)
    std = variance ** 0.5

    short = sum(1 for l in all_lengths if l < 10) / len(all_lengths)
    medium = sum(1 for l in all_lengths if 10 <= l <= 25) / len(all_lengths)
    long = sum(1 for l in all_lengths if l > 25) / len(all_lengths)

    return SentenceStats(
        avg_length=round(avg, 1),
        min_length=min(all_lengths),
        max_length=max(all_lengths),
        std_dev=round(std, 1),
        short_ratio=round(short, 3),
        medium_ratio=round(medium, 3),
        long_ratio=round(long, 3),
    )


def analyze_vocabulary(paragraphs: list[dict]) -> VocabularyStats:
    """Analyze vocabulary usage patterns."""
    all_words = []
    word_freq = Counter()

    for para in paragraphs:
        text = para.get("text", "")
        words = tokenize_words(text)
        all_words.extend(words)
        word_freq.update(words)

    if not all_words:
        return VocabularyStats()

    # Basic stats
    avg_word_len = sum(len(w) for w in all_words) / len(all_words)
    unique_ratio = len(set(all_words)) / len(all_words)

    # Categorize words (simplified without POS tagger)
    # Common academic verbs
    verb_patterns = [
        'show', 'demonstrate', 'indicate', 'suggest', 'reveal', 'find',
        'observe', 'report', 'describe', 'present', 'analyze', 'examine',
        'investigate', 'study', 'compare', 'evaluate', 'assess', 'measure',
        'determine', 'identify', 'confirm', 'support', 'provide', 'require',
        'perform', 'conduct', 'obtain', 'achieve', 'increase', 'decrease',
        'enhance', 'improve', 'reduce', 'affect', 'influence', 'contribute'
    ]

    # Common academic adjectives
    adj_patterns = [
        'significant', 'important', 'major', 'key', 'critical', 'essential',
        'primary', 'main', 'novel', 'new', 'recent', 'previous', 'current',
        'different', 'similar', 'various', 'specific', 'particular', 'general',
        'high', 'low', 'large', 'small', 'strong', 'weak', 'positive', 'negative'
    ]

    # Transition words
    transitions = [
        'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore',
        'additionally', 'consequently', 'nevertheless', 'although', 'whereas',
        'while', 'because', 'since', 'also', 'similarly', 'conversely',
        'specifically', 'particularly', 'notably', 'importantly', 'finally'
    ]

    # Count occurrences
    top_verbs = [(w, word_freq[w]) for w in verb_patterns if word_freq[w] > 0]
    top_verbs.sort(key=lambda x: -x[1])

    top_adjs = [(w, word_freq[w]) for w in adj_patterns if word_freq[w] > 0]
    top_adjs.sort(key=lambda x: -x[1])

    fav_transitions = [(w, word_freq[w]) for w in transitions if word_freq[w] > 0]
    fav_transitions.sort(key=lambda x: -x[1])

    # Find avoided words (common AI words that author doesn't use)
    ai_common = ['utilize', 'leverage', 'facilitate', 'elucidate', 'ascertain',
                 'commence', 'procure', 'endeavor', 'paramount', 'plethora']
    avoided = [w for w in ai_common if word_freq[w] == 0]

    return VocabularyStats(
        avg_word_length=round(avg_word_len, 2),
        unique_ratio=round(unique_ratio, 3),
        top_verbs=[w for w, _ in top_verbs[:15]],
        top_adjectives=[w for w, _ in top_adjs[:15]],
        favorite_transitions=[w for w, _ in fav_transitions[:10]],
        avoided_words=avoided,
    )


def analyze_structure(paragraphs: list[dict]) -> StructureStats:
    """Analyze writing structure patterns."""
    passive_count = 0
    total_sentences = 0
    hedge_count = 0
    total_words = 0
    question_count = 0
    citation_styles = Counter()

    for para in paragraphs:
        text = para.get("text", "")
        sentences = split_sentences(text)
        words = tokenize_words(text)

        total_sentences += len(sentences)
        total_words += len(words)

        for sent in sentences:
            # Check passive voice (simplified)
            if re.search(r'\b(was|were|been|being|is|are)\s+\w+ed\b', sent, re.IGNORECASE):
                passive_count += 1

            # Check questions
            if '?' in sent:
                question_count += 1

        # Count hedge words
        hedge_words = re.findall(r'\b(may|might|could|possibly|potentially|likely|suggest|appear|seem)\b', text, re.IGNORECASE)
        hedge_count += len(hedge_words)

        # Detect citation style
        if re.search(r'\([A-Z][a-z]+(?:\s+(?:et al\.)?)?,?\s*\d{4}\)', text):
            citation_styles['author_year'] += 1
        if re.search(r'\[\d+\]', text):
            citation_styles['numbered'] += 1

    # Calculate ratios
    passive_ratio = passive_count / total_sentences if total_sentences > 0 else 0
    question_ratio = question_count / total_sentences if total_sentences > 0 else 0
    hedge_freq = (hedge_count / total_words * 100) if total_words > 0 else 0

    # Determine citation style
    if citation_styles:
        citation_style = citation_styles.most_common(1)[0][0]
    else:
        citation_style = "unknown"

    # Average sentences per paragraph
    avg_para_sent = total_sentences / len(paragraphs) if paragraphs else 0

    return StructureStats(
        passive_voice_ratio=round(passive_ratio, 3),
        question_ratio=round(question_ratio, 3),
        hedge_frequency=round(hedge_freq, 2),
        citation_style=citation_style,
        avg_paragraph_sentences=round(avg_para_sent, 1),
    )


def analyze_phrases(paragraphs: list[dict]) -> PhrasePatterns:
    """Extract common phrase patterns."""
    opening_counter = Counter()
    closing_counter = Counter()
    transition_counter = Counter()
    emphasis_counter = Counter()

    for para in paragraphs:
        text = para.get("text", "")
        sentences = split_sentences(text)

        for i, sent in enumerate(sentences):
            words = sent.split()[:5]  # First 5 words
            if len(words) >= 3:
                opening = ' '.join(words[:3]).lower()
                opening_counter[opening] += 1

            # Check for transition patterns
            trans_match = re.match(r'^(\w+(?:\s+\w+)?),?\s', sent)
            if trans_match:
                transition_counter[trans_match.group(1).lower()] += 1

            # Check for emphasis patterns
            emphasis_patterns = [
                r'\b(importantly|notably|significantly|interestingly)\b',
                r'\b(in particular|of note|of interest)\b',
                r'\b(it is (important|notable|worth) (to note|that))\b',
            ]
            for pattern in emphasis_patterns:
                matches = re.findall(pattern, sent, re.IGNORECASE)
                for m in matches:
                    if isinstance(m, tuple):
                        m = m[0]
                    emphasis_counter[m.lower()] += 1

    return PhrasePatterns(
        opening_patterns=[p for p, _ in opening_counter.most_common(20) if _ > 2],
        transition_patterns=[p for p, _ in transition_counter.most_common(15) if _ > 2],
        emphasis_patterns=[p for p, _ in emphasis_counter.most_common(10) if _ > 1],
    )


def generate_style_rules(
    sentence_stats: SentenceStats,
    vocab_stats: VocabularyStats,
    structure_stats: StructureStats,
    phrase_patterns: PhrasePatterns,
) -> tuple[list[str], list[str], list[str]]:
    """Generate style rules from analysis.

    Returns:
        Tuple of (style_rules, avoid_patterns, prefer_patterns)
    """
    rules = []
    avoid = []
    prefer = []

    # Sentence length rules
    if sentence_stats.avg_length > 0:
        rules.append(f"Target average sentence length: {sentence_stats.avg_length:.0f} words (std: {sentence_stats.std_dev:.0f})")
        rules.append(f"Mix sentence lengths: {sentence_stats.short_ratio*100:.0f}% short (<10), {sentence_stats.medium_ratio*100:.0f}% medium (10-25), {sentence_stats.long_ratio*100:.0f}% long (>25)")

    # Passive voice
    if structure_stats.passive_voice_ratio > 0:
        pv_pct = structure_stats.passive_voice_ratio * 100
        if pv_pct > 40:
            rules.append(f"Use passive voice frequently (~{pv_pct:.0f}% of sentences)")
        elif pv_pct > 20:
            rules.append(f"Use passive voice moderately (~{pv_pct:.0f}% of sentences)")
        else:
            rules.append(f"Prefer active voice (only ~{pv_pct:.0f}% passive)")

    # Hedge words
    if structure_stats.hedge_frequency > 0:
        rules.append(f"Hedge word frequency: ~{structure_stats.hedge_frequency:.1f} per 100 words")
        if structure_stats.hedge_frequency < 1.0:
            rules.append("Be direct and confident in claims")
        else:
            rules.append("Use appropriate hedging for scientific uncertainty")

    # Vocabulary preferences
    if vocab_stats.top_verbs:
        prefer.extend([f"Use '{v}'" for v in vocab_stats.top_verbs[:5]])

    if vocab_stats.favorite_transitions:
        prefer.extend([f"Transition with '{t}'" for t in vocab_stats.favorite_transitions[:5]])

    if vocab_stats.avoided_words:
        avoid.extend([f"Avoid '{w}'" for w in vocab_stats.avoided_words])

    # General anti-AI rules
    avoid.extend([
        "Never use 'Furthermore' or 'Moreover' - use 'Also' or remove",
        "Never use 'It is worth noting that' - just state it directly",
        "Never use 'utilize' - use 'use'",
        "Don't enumerate with 'First...Second...Third' - vary structure",
        "Don't start conclusions with 'In conclusion,'",
    ])

    return rules, avoid, prefer


# ==============================================================================
# Main Analysis
# ==============================================================================

def analyze_corpus(corpus_path: Path, author_name: str = "Gilles") -> StyleFingerprint:
    """Perform full corpus analysis.

    Args:
        corpus_path: Path to parsed corpus directory
        author_name: Name of the author

    Returns:
        StyleFingerprint with analysis results
    """
    print_info(f"Loading corpus from {corpus_path}")
    paragraphs = load_corpus(corpus_path)

    if not paragraphs:
        print_error("No corpus found!")
        return StyleFingerprint(author_name=author_name)

    print_info(f"Analyzing {len(paragraphs)} paragraphs...")

    # Calculate total words and sentences
    total_words = sum(len(tokenize_words(p.get("text", ""))) for p in paragraphs)
    total_sentences = sum(len(split_sentences(p.get("text", ""))) for p in paragraphs)

    # Run analyses
    print_info("Analyzing sentences...")
    sentence_stats = analyze_sentences(paragraphs)

    print_info("Analyzing vocabulary...")
    vocab_stats = analyze_vocabulary(paragraphs)

    print_info("Analyzing structure...")
    structure_stats = analyze_structure(paragraphs)

    print_info("Extracting phrase patterns...")
    phrase_patterns = analyze_phrases(paragraphs)

    print_info("Generating style rules...")
    style_rules, avoid_patterns, prefer_patterns = generate_style_rules(
        sentence_stats, vocab_stats, structure_stats, phrase_patterns
    )

    return StyleFingerprint(
        author_name=author_name,
        corpus_size=len(paragraphs),
        total_words=total_words,
        total_sentences=total_sentences,
        sentence_stats=sentence_stats,
        vocabulary_stats=vocab_stats,
        structure_stats=structure_stats,
        phrase_patterns=phrase_patterns,
        style_rules=style_rules,
        avoid_patterns=avoid_patterns,
        prefer_patterns=prefer_patterns,
    )


def save_fingerprint(fingerprint: StyleFingerprint, output_path: Path) -> None:
    """Save fingerprint to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    data = {
        "author_name": fingerprint.author_name,
        "corpus_size": fingerprint.corpus_size,
        "total_words": fingerprint.total_words,
        "total_sentences": fingerprint.total_sentences,
        "analysis_version": fingerprint.analysis_version,
        "sentence_stats": asdict(fingerprint.sentence_stats),
        "vocabulary_stats": asdict(fingerprint.vocabulary_stats),
        "structure_stats": asdict(fingerprint.structure_stats),
        "phrase_patterns": asdict(fingerprint.phrase_patterns),
        "style_rules": fingerprint.style_rules,
        "avoid_patterns": fingerprint.avoid_patterns,
        "prefer_patterns": fingerprint.prefer_patterns,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_fingerprint(fingerprint: StyleFingerprint, verbose: bool = False) -> None:
    """Print fingerprint summary to console."""

    print_section("Corpus Overview")
    print(f"  Author:          {fingerprint.author_name}")
    print(f"  Paragraphs:      {fingerprint.corpus_size}")
    print(f"  Total words:     {fingerprint.total_words:,}")
    print(f"  Total sentences: {fingerprint.total_sentences:,}")

    print_section("Sentence Statistics")
    ss = fingerprint.sentence_stats
    print(f"  Average length:  {ss.avg_length} words (std: {ss.std_dev})")
    print(f"  Range:           {ss.min_length} - {ss.max_length} words")
    print(f"  Distribution:    {ss.short_ratio*100:.0f}% short, {ss.medium_ratio*100:.0f}% medium, {ss.long_ratio*100:.0f}% long")

    print_section("Structure")
    st = fingerprint.structure_stats
    print(f"  Passive voice:   {st.passive_voice_ratio*100:.1f}% of sentences")
    print(f"  Hedge frequency: {st.hedge_frequency:.2f} per 100 words")
    print(f"  Citation style:  {st.citation_style}")

    print_section("Vocabulary")
    vs = fingerprint.vocabulary_stats
    print(f"  Avg word length: {vs.avg_word_length:.1f} characters")
    print(f"  Vocabulary diversity: {vs.unique_ratio*100:.1f}%")

    if vs.top_verbs:
        print(f"  Top verbs:       {', '.join(vs.top_verbs[:7])}")
    if vs.favorite_transitions:
        print(f"  Transitions:     {', '.join(vs.favorite_transitions[:7])}")
    if vs.avoided_words:
        print(f"  Avoided words:   {', '.join(vs.avoided_words)}")

    if verbose:
        print_section("Phrase Patterns")
        pp = fingerprint.phrase_patterns
        if pp.opening_patterns:
            print(f"  Common openings: {', '.join(pp.opening_patterns[:5])}")
        if pp.transition_patterns:
            print(f"  Transition words: {', '.join(pp.transition_patterns[:7])}")

    print_section("Style Rules")
    for i, rule in enumerate(fingerprint.style_rules[:8], 1):
        print(f"  {i}. {rule}")

    print_section("Patterns to Avoid")
    for pattern in fingerprint.avoid_patterns[:5]:
        print(f"  - {pattern}")

    print_section("Patterns to Prefer")
    for pattern in fingerprint.prefer_patterns[:5]:
        print(f"  + {pattern}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze author writing style from corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/analyze_style.py                # Analyze and display
    python scripts/analyze_style.py --verbose      # Detailed output
    python scripts/analyze_style.py -o style.json  # Save to file
    python scripts/analyze_style.py --author "Dr. Smith"  # Custom author name
        """
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output path for fingerprint JSON"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(CORPUS_PARSED_DIR),
        help="Path to parsed corpus directory"
    )
    parser.add_argument(
        "--author",
        type=str,
        default="Gilles",
        help="Author name for fingerprint"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - only save file, minimal output"
    )

    args = parser.parse_args()

    print_header("Author Style Fingerprint Analyzer")

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print_error(f"Corpus directory not found: {corpus_path}")
        print_info("Run 'make parse-corpus' first to parse your documents")
        return 1

    # Analyze
    fingerprint = analyze_corpus(corpus_path, author_name=args.author)

    if fingerprint.corpus_size == 0:
        print_error("No paragraphs found in corpus")
        return 1

    # Save
    output_path = Path(args.output)
    save_fingerprint(fingerprint, output_path)
    print_success(f"Saved fingerprint to: {output_path}")

    # Display
    if not args.quiet:
        print()
        print_fingerprint(fingerprint, verbose=args.verbose)

    print()
    print_success("Style analysis complete!")
    print_info(f"Use this fingerprint in prompts: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
