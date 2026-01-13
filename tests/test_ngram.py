"""Tests for n-gram utilities."""
import pytest
from gswa.utils.ngram import (
    tokenize, get_ngrams, build_ngram_index,
    find_longest_match_simple, compute_ngram_overlap
)


class TestTokenize:
    """Tests for tokenize function."""

    def test_basic_tokenize(self):
        """Test basic tokenization."""
        text = "Hello, World! This is a test."
        tokens = tokenize(text)
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_numbers_preserved(self):
        """Test that numbers are preserved."""
        text = "We observed a 2.5-fold increase at 10 μM"
        tokens = tokenize(text)
        assert "2" in tokens
        assert "5" in tokens
        assert "10" in tokens

    def test_empty_text(self):
        """Test empty text."""
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_unicode(self):
        """Test unicode handling."""
        text = "résumé naïve café"
        tokens = tokenize(text)
        assert len(tokens) == 3


class TestGetNgrams:
    """Tests for get_ngrams function."""

    def test_basic_ngrams(self):
        """Test basic n-gram extraction."""
        tokens = ["a", "b", "c", "d", "e"]
        ngrams = get_ngrams(tokens, 3)
        assert ngrams == [("a", "b", "c"), ("b", "c", "d"), ("c", "d", "e")]

    def test_ngrams_equal_length(self):
        """Test n-grams when n equals token count."""
        tokens = ["a", "b", "c"]
        ngrams = get_ngrams(tokens, 3)
        assert ngrams == [("a", "b", "c")]

    def test_ngrams_larger_than_tokens(self):
        """Test n-grams when n is larger than token count."""
        tokens = ["a", "b"]
        ngrams = get_ngrams(tokens, 3)
        assert ngrams == []

    def test_empty_tokens(self):
        """Test empty token list."""
        assert get_ngrams([], 3) == []


class TestBuildNgramIndex:
    """Tests for build_ngram_index function."""

    def test_build_index(self):
        """Test building n-gram index."""
        corpus = [
            "The quick brown fox",
            "jumps over the lazy dog"
        ]
        ngram_set = build_ngram_index(corpus, n=3)

        # Should contain 3-grams from both texts
        assert ("the", "quick", "brown") in ngram_set
        assert ("jumps", "over", "the") in ngram_set

    def test_empty_corpus(self):
        """Test empty corpus."""
        ngram_set = build_ngram_index([], n=3)
        assert len(ngram_set) == 0

    def test_duplicate_ngrams_merged(self):
        """Test that duplicate n-grams are merged."""
        corpus = [
            "the quick fox",
            "the quick dog"
        ]
        ngram_set = build_ngram_index(corpus, n=2)
        # "the quick" appears twice but should only be in set once
        assert ("the", "quick") in ngram_set


class TestFindLongestMatch:
    """Tests for find_longest_match_simple function."""

    def test_exact_match(self):
        """Test finding exact matches."""
        corpus = ["The quick brown fox jumps over the lazy dog"]
        ngram_set = build_ngram_index(corpus, n=8)

        # Exact substring should match
        candidate_tokens = tokenize("the quick brown fox")
        match_len = find_longest_match_simple(candidate_tokens, ngram_set, ngram_size=8)
        assert match_len >= 3  # Should find at least 3-gram match

    def test_no_match(self):
        """Test no match case."""
        corpus = ["The quick brown fox jumps over the lazy dog"]
        ngram_set = build_ngram_index(corpus, n=8)

        candidate_tokens = tokenize("hello world goodbye")
        match_len = find_longest_match_simple(candidate_tokens, ngram_set, ngram_size=8)
        assert match_len == 0

    def test_partial_match(self):
        """Test partial match detection."""
        corpus = ["We observed a significant increase in enzyme activity"]
        ngram_set = build_ngram_index(corpus, n=5)

        candidate_tokens = tokenize("We observed a significant change in protein levels")
        match_len = find_longest_match_simple(candidate_tokens, ngram_set, ngram_size=5)
        # "we observed a significant" should match (4 tokens)
        assert match_len >= 3

    def test_empty_inputs(self):
        """Test empty inputs."""
        assert find_longest_match_simple([], set(), ngram_size=8) == 0
        assert find_longest_match_simple(["a", "b"], set(), ngram_size=8) == 0


class TestComputeNgramOverlap:
    """Tests for compute_ngram_overlap function."""

    def test_high_overlap(self):
        """Test high overlap detection."""
        corpus = ["We observed a significant increase in activity levels"]
        ngram_set = build_ngram_index(corpus, n=5)

        # Similar text
        candidate = "We observed a significant increase in protein activity levels"
        result = compute_ngram_overlap(candidate, ngram_set, n=5)

        assert "max_consecutive_match" in result
        assert "overlap_ratio" in result
        assert "matching_ngrams" in result

    def test_no_overlap(self):
        """Test no overlap case."""
        corpus = ["We observed a significant increase in activity levels"]
        ngram_set = build_ngram_index(corpus, n=5)

        # Completely different text (no shared words)
        candidate = "The weather seems nice today around Paris"
        result = compute_ngram_overlap(candidate, ngram_set, n=5)

        assert result["max_consecutive_match"] == 0
        assert result["matching_ngrams"] == 0

    def test_empty_candidate(self):
        """Test empty candidate."""
        ngram_set = {("a", "b", "c")}
        result = compute_ngram_overlap("", ngram_set, n=3)

        assert result["max_consecutive_match"] == 0
        assert result["overlap_ratio"] == 0.0
        assert result["matching_ngrams"] == 0

    def test_short_candidate(self):
        """Test candidate shorter than n-gram size."""
        ngram_set = {("a", "b", "c", "d", "e")}
        result = compute_ngram_overlap("one two", ngram_set, n=5)

        assert result["overlap_ratio"] == 0.0


class TestIntegration:
    """Integration tests for n-gram detection."""

    def test_scientific_text_similarity(self):
        """Test with scientific-like text."""
        corpus = [
            "We observed a 2.5-fold increase in enzyme activity when cells were treated with compound X at 10 μM for 24 hours.",
            "The results demonstrate that compound X enhances catalytic efficiency through allosteric modulation."
        ]
        ngram_set = build_ngram_index(corpus, n=8)

        # Very similar text (should trigger fallback in real scenario)
        similar = "We observed a 2.5-fold increase in enzyme activity when cells were exposed to compound X at 10 μM."
        result = compute_ngram_overlap(similar, ngram_set, n=8)

        # Should detect significant overlap
        assert result["max_consecutive_match"] >= 5 or result["matching_ngrams"] > 0

        # Different text
        different = "The protein expression levels were measured using Western blot analysis after 48 hours of incubation."
        result = compute_ngram_overlap(different, ngram_set, n=8)

        # Should have low or no overlap
        assert result["max_consecutive_match"] <= 2
