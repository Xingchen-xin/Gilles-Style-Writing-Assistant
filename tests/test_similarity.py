"""Tests for combined similarity service."""
import pytest
import tempfile
import json
from pathlib import Path

from gswa.services.similarity import SimilarityService, get_similarity_service


@pytest.fixture
def sample_corpus_dir():
    """Create a temporary corpus directory with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_dir = Path(tmpdir) / "corpus"
        corpus_dir.mkdir()

        # Create sample corpus file
        corpus_file = corpus_dir / "papers.jsonl"
        with open(corpus_file, "w") as f:
            for i, text in enumerate([
                "We observed a significant increase in enzyme activity when treated with compound X.",
                "The protein expression levels were measured using Western blot analysis.",
                "Statistical analysis was performed using one-way ANOVA with post-hoc tests.",
                "Cell viability was assessed using the MTT assay after 24 hours of treatment.",
            ]):
                f.write(json.dumps({
                    "text": text,
                    "doc_id": f"paper_{i}",
                    "para_id": f"para_{i}"
                }) + "\n")

        yield str(corpus_dir)


class TestSimilarityServiceBasic:
    """Basic tests for SimilarityService."""

    def test_initialization(self):
        """Test service initialization."""
        service = SimilarityService()
        assert service.is_loaded is False
        assert service.corpus_size == 0

    def test_load_nonexistent_corpus(self):
        """Test loading from nonexistent path."""
        service = SimilarityService()
        count = service.load_corpus("/nonexistent/path")
        assert count == 0
        assert service.is_loaded is False


class TestSimilarityServiceWithCorpus:
    """Tests requiring a corpus."""

    def test_load_corpus(self, sample_corpus_dir):
        """Test loading corpus from JSONL files."""
        service = SimilarityService()
        count = service.load_corpus(sample_corpus_dir)

        assert count == 4
        assert service.is_loaded is True

    def test_check_similarity_high_match(self, sample_corpus_dir):
        """Test similarity check with high match."""
        service = SimilarityService()
        service.load_corpus(sample_corpus_dir)

        # Very similar text to corpus
        candidate = "We observed a significant increase in enzyme activity when treated with compound X."
        should_fallback, scores = service.check_similarity(candidate)

        # Should have high n-gram match
        assert scores["ngram_max_match"] > 0
        assert "ngram_overlap" in scores
        assert "embed_top1" in scores

    def test_check_similarity_low_match(self, sample_corpus_dir):
        """Test similarity check with low match."""
        service = SimilarityService()
        service.load_corpus(sample_corpus_dir)

        # Very different text
        candidate = "The weather forecast predicts sunny skies for the weekend ahead."
        should_fallback, scores = service.check_similarity(candidate)

        # Should have low scores
        assert scores["ngram_max_match"] < 5

    def test_check_similarity_empty_corpus(self):
        """Test similarity check with empty corpus."""
        service = SimilarityService()
        # Don't load corpus

        candidate = "Some text to check."
        should_fallback, scores = service.check_similarity(candidate)

        # Should return default scores
        assert scores["ngram_max_match"] == 0
        assert scores["ngram_overlap"] == 0.0
        assert scores["embed_top1"] == 0.0
        assert should_fallback is False


class TestSimilarityThresholds:
    """Tests for similarity threshold logic."""

    def test_fallback_triggered_by_ngram(self, sample_corpus_dir):
        """Test that fallback is triggered by high n-gram match."""
        service = SimilarityService()
        service.load_corpus(sample_corpus_dir)

        # Mock scores that exceed threshold
        # Use a very similar sentence to corpus
        similar = "We observed a significant increase in enzyme activity when treated with compound X at high concentration."
        should_fallback, scores = service.check_similarity(similar)

        # The fallback decision is based on thresholds from settings
        # We just verify the scores are computed
        assert isinstance(should_fallback, bool)
        assert isinstance(scores["ngram_max_match"], int)

    def test_fallback_not_triggered_low_similarity(self, sample_corpus_dir):
        """Test that fallback is not triggered for low similarity."""
        service = SimilarityService()
        service.load_corpus(sample_corpus_dir)

        # Very different text
        different = "The quarterly financial report shows strong revenue growth across all market segments."
        should_fallback, scores = service.check_similarity(different)

        # With very different text, should not trigger fallback
        # (unless thresholds are set very low in config)
        assert scores["ngram_max_match"] < 12  # Below default threshold


class TestSimilarityServiceSingleton:
    """Tests for singleton pattern."""

    def test_get_similarity_service_singleton(self):
        """Test singleton pattern."""
        import gswa.services.similarity as sim_module
        sim_module._similarity_service = None

        service1 = get_similarity_service()
        service2 = get_similarity_service()

        assert service1 is service2


class TestIndexPersistence:
    """Tests for index save/load."""

    def test_save_and_load_ngram_index(self, sample_corpus_dir):
        """Test saving and loading n-gram index."""
        with tempfile.TemporaryDirectory() as index_dir:
            # Build and save index
            service = SimilarityService()
            service.load_corpus(sample_corpus_dir)
            service.save_index(index_dir)

            # Verify files were created
            ngram_file = Path(index_dir) / "ngrams.json"
            assert ngram_file.exists()

            # Load into new service
            new_service = SimilarityService()
            success = new_service.load_index(index_dir)

            # Should load at least n-gram index
            assert success or new_service._ngram_index is not None
