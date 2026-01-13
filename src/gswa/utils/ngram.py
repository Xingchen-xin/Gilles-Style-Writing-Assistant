"""N-gram overlap detection.

Detects verbatim or near-verbatim copying from corpus.
"""
import re
from typing import Set, Tuple


def tokenize(text: str) -> list[str]:
    """Simple word tokenization.

    Args:
        text: Input text to tokenize

    Returns:
        List of lowercase tokens
    """
    # Lowercase and split on non-alphanumeric
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def get_ngrams(tokens: list[str], n: int) -> list[Tuple[str, ...]]:
    """Extract n-grams from token list.

    Args:
        tokens: List of tokens
        n: N-gram size

    Returns:
        List of n-gram tuples
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def build_ngram_index(corpus_texts: list[str], n: int = 8) -> Set[Tuple[str, ...]]:
    """Build n-gram set from corpus for fast lookup.

    Args:
        corpus_texts: List of corpus paragraph texts
        n: N-gram size for indexing

    Returns:
        Set of n-gram tuples for fast membership testing
    """
    ngram_set: Set[Tuple[str, ...]] = set()
    for text in corpus_texts:
        tokens = tokenize(text)
        ngrams = get_ngrams(tokens, n)
        ngram_set.update(ngrams)
    return ngram_set


def find_longest_match(
    candidate_tokens: list[str],
    corpus_ngram_sets: dict[int, Set[Tuple[str, ...]]],
    max_n: int = 20
) -> int:
    """Find longest consecutive n-gram match with corpus.

    Uses binary search for efficiency when multiple n-gram sets are available.

    Args:
        candidate_tokens: Tokenized candidate text
        corpus_ngram_sets: Dictionary mapping n-gram size to ngram sets
        max_n: Maximum n-gram size to check

    Returns:
        Length of longest matching n-gram (0 if no match)
    """
    if not candidate_tokens or not corpus_ngram_sets:
        return 0

    longest = 0
    max_possible = min(max_n, len(candidate_tokens))

    # Check n-grams of decreasing length
    for n in range(max_possible, 0, -1):
        # Get or create n-gram set for this size
        if n in corpus_ngram_sets:
            ngram_set = corpus_ngram_sets[n]
        else:
            # Skip sizes not in the index
            continue

        ngrams = get_ngrams(candidate_tokens, n)
        for ngram in ngrams:
            if ngram in ngram_set:
                return n  # Found match at this length

    return longest


def find_longest_match_simple(
    candidate_tokens: list[str],
    corpus_ngram_set: Set[Tuple[str, ...]],
    ngram_size: int = 8,
    max_n: int = 20
) -> int:
    """Find longest consecutive n-gram match with corpus (simple version).

    This version builds n-grams on the fly from a single base n-gram set.
    Less efficient but simpler when only one n-gram size is indexed.

    Args:
        candidate_tokens: Tokenized candidate text
        corpus_ngram_set: Set of corpus n-grams (at ngram_size)
        ngram_size: Size of n-grams in corpus_ngram_set
        max_n: Maximum n-gram size to check

    Returns:
        Length of longest matching n-gram (0 if no match)
    """
    if not candidate_tokens or not corpus_ngram_set:
        return 0

    # For simple version, we check if candidate n-grams exist in corpus
    # We can only reliably detect matches up to ngram_size
    max_check = min(max_n, len(candidate_tokens), ngram_size)

    for n in range(max_check, 0, -1):
        candidate_ngrams = get_ngrams(candidate_tokens, n)
        # Check against corpus n-grams of same size
        # For sizes smaller than ngram_size, we need to extract sub-ngrams
        if n == ngram_size:
            for ngram in candidate_ngrams:
                if ngram in corpus_ngram_set:
                    return n
        elif n < ngram_size:
            # Check if any candidate n-gram appears in any corpus n-gram
            for ngram in candidate_ngrams:
                for corpus_ngram in corpus_ngram_set:
                    # Check if candidate n-gram is a subsequence
                    for i in range(len(corpus_ngram) - n + 1):
                        if corpus_ngram[i:i+n] == ngram:
                            return n

    return 0


def compute_ngram_overlap(
    candidate: str,
    corpus_ngram_set: Set[Tuple[str, ...]],
    n: int = 8
) -> dict:
    """Compute n-gram overlap metrics.

    Args:
        candidate: The generated text to check
        corpus_ngram_set: Pre-built set of corpus n-grams
        n: N-gram size for overlap calculation

    Returns:
        Dictionary with:
            - max_consecutive_match: int - Longest matching n-gram length
            - overlap_ratio: float - Jaccard-like overlap ratio
            - matching_ngrams: int - Number of matching n-grams
    """
    tokens = tokenize(candidate)

    # Find longest consecutive match
    max_match = find_longest_match_simple(tokens, corpus_ngram_set, n)

    # Compute overlap ratio for n-grams
    candidate_ngrams = get_ngrams(tokens, n)
    if not candidate_ngrams:
        return {
            "max_consecutive_match": max_match,
            "overlap_ratio": 0.0,
            "matching_ngrams": 0
        }

    candidate_set = set(candidate_ngrams)
    matching = candidate_set.intersection(corpus_ngram_set)
    overlap_ratio = len(matching) / len(candidate_set) if candidate_set else 0.0

    return {
        "max_consecutive_match": max_match,
        "overlap_ratio": overlap_ratio,
        "matching_ngrams": len(matching)
    }
