"""Tests for embedding similarity service."""
import pytest
import tempfile
import os

# Skip all tests if dependencies not available
pytest.importorskip("sentence_transformers")
pytest.importorskip("faiss")

from gswa.utils.embedding import EmbeddingService


@pytest.fixture
def embedding_service():
    """Create embedding service for tests."""
    return EmbeddingService()


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return [
        "We observed a significant increase in enzyme activity.",
        "The protein expression levels were measured using Western blot.",
        "Statistical analysis was performed using t-test.",
        "The results demonstrate a clear correlation between variables.",
        "Cell viability was assessed using MTT assay."
    ]


@pytest.fixture
def sample_ids(sample_corpus):
    """Generate sample IDs for corpus."""
    doc_ids = [f"doc_{i}" for i in range(len(sample_corpus))]
    para_ids = [f"para_{i}" for i in range(len(sample_corpus))]
    return doc_ids, para_ids


class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    def test_encode(self, embedding_service):
        """Test encoding texts to embeddings."""
        texts = ["Hello world", "Test sentence"]
        embeddings = embedding_service.encode(texts)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Embedding dimension

    def test_build_index(self, embedding_service, sample_corpus, sample_ids):
        """Test building FAISS index."""
        doc_ids, para_ids = sample_ids
        embedding_service.build_index(sample_corpus, doc_ids, para_ids)

        assert embedding_service.is_loaded
        assert embedding_service.index_size == len(sample_corpus)

    def test_search(self, embedding_service, sample_corpus, sample_ids):
        """Test similarity search."""
        doc_ids, para_ids = sample_ids
        embedding_service.build_index(sample_corpus, doc_ids, para_ids)

        # Search for similar text
        results = embedding_service.search("enzyme activity measurement", top_k=3)

        assert len(results) == 3
        assert all("similarity" in r for r in results)
        assert all("doc_id" in r for r in results)
        assert all("para_id" in r for r in results)
        # Results should be sorted by similarity
        assert results[0]["similarity"] >= results[1]["similarity"]

    def test_compute_similarity(self, embedding_service, sample_corpus, sample_ids):
        """Test compute_similarity method."""
        doc_ids, para_ids = sample_ids
        embedding_service.build_index(sample_corpus, doc_ids, para_ids)

        result = embedding_service.compute_similarity(
            "We observed a significant increase in protein levels",
            top_k=3
        )

        assert "top1_similarity" in result
        assert "top1_doc_id" in result
        assert "top1_para_id" in result
        assert "topk_avg" in result
        assert result["top1_similarity"] >= result["topk_avg"]

    def test_save_and_load_index(self, embedding_service, sample_corpus, sample_ids):
        """Test saving and loading index."""
        doc_ids, para_ids = sample_ids
        embedding_service.build_index(sample_corpus, doc_ids, para_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save index
            embedding_service.save_index(tmpdir)

            # Create new service and load
            new_service = EmbeddingService()
            success = new_service.load_index(tmpdir)

            assert success
            assert new_service.is_loaded
            assert new_service.index_size == len(sample_corpus)

            # Verify search works
            results = new_service.search("enzyme activity", top_k=2)
            assert len(results) == 2

    def test_load_nonexistent_index(self, embedding_service):
        """Test loading from nonexistent path."""
        success = embedding_service.load_index("/nonexistent/path")
        assert success is False

    def test_search_empty_index(self, embedding_service):
        """Test search on empty index."""
        results = embedding_service.search("test query")
        assert results == []

    def test_compute_similarity_empty_index(self, embedding_service):
        """Test compute_similarity on empty index."""
        result = embedding_service.compute_similarity("test query")

        assert result["top1_similarity"] == 0.0
        assert result["top1_doc_id"] is None
        assert result["topk_avg"] == 0.0


class TestSimilarityThresholds:
    """Tests for similarity thresholds used in fallback logic."""

    def test_high_similarity_detection(self, embedding_service, sample_corpus, sample_ids):
        """Test that very similar texts have high similarity scores."""
        doc_ids, para_ids = sample_ids
        embedding_service.build_index(sample_corpus, doc_ids, para_ids)

        # Almost identical text should have very high similarity
        result = embedding_service.compute_similarity(
            "We observed a significant increase in enzyme activity."
        )
        # Expect similarity > 0.9 for near-identical text
        assert result["top1_similarity"] > 0.9

    def test_low_similarity_detection(self, embedding_service, sample_corpus, sample_ids):
        """Test that different texts have lower similarity scores."""
        doc_ids, para_ids = sample_ids
        embedding_service.build_index(sample_corpus, doc_ids, para_ids)

        # Completely different topic
        result = embedding_service.compute_similarity(
            "The weather forecast predicts sunny skies tomorrow."
        )
        # Expect lower similarity for unrelated text
        assert result["top1_similarity"] < 0.5
