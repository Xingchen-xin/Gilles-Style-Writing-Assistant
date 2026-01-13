"""Embedding-based similarity detection.

Uses sentence transformers for semantic similarity.
"""
import numpy as np
from typing import Optional
from pathlib import Path


class EmbeddingService:
    """Manages embedding model and similarity search."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self._model = None
        self._index = None
        self._doc_ids: list[str] = []
        self._para_ids: list[str] = []

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            Numpy array of embeddings (normalized)
        """
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def build_index(
        self,
        texts: list[str],
        doc_ids: list[str],
        para_ids: list[str]
    ) -> None:
        """Build FAISS index from corpus texts.

        Args:
            texts: List of corpus paragraph texts
            doc_ids: List of document IDs
            para_ids: List of paragraph IDs
        """
        import faiss

        embeddings = self.encode(texts)
        dimension = embeddings.shape[1]

        # Use inner product (cosine sim with normalized vectors)
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings.astype(np.float32))

        self._doc_ids = doc_ids
        self._para_ids = para_ids

    def save_index(self, path: str) -> None:
        """Save index to disk.

        Args:
            path: Directory path to save index files
        """
        import faiss
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "corpus.faiss"))
        np.save(path / "doc_ids.npy", np.array(self._doc_ids))
        np.save(path / "para_ids.npy", np.array(self._para_ids))

    def load_index(self, path: str) -> bool:
        """Load index from disk.

        Args:
            path: Directory path containing index files

        Returns:
            True if successfully loaded, False otherwise
        """
        import faiss
        path = Path(path)

        index_file = path / "corpus.faiss"
        if not index_file.exists():
            return False

        self._index = faiss.read_index(str(index_file))
        self._doc_ids = np.load(path / "doc_ids.npy").tolist()
        self._para_ids = np.load(path / "para_ids.npy").tolist()
        return True

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """Search for most similar corpus paragraphs.

        Args:
            query: Query text to search for
            top_k: Number of results to return

        Returns:
            List of {similarity, doc_id, para_id, rank} dicts
        """
        if self._index is None:
            return []

        query_embedding = self.encode([query])
        scores, indices = self._index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for missing
                continue
            results.append({
                "similarity": float(score),
                "doc_id": self._doc_ids[idx],
                "para_id": self._para_ids[idx],
                "rank": i + 1
            })
        return results

    def compute_similarity(
        self,
        candidate: str,
        top_k: int = 5
    ) -> dict:
        """Compute embedding similarity metrics.

        Args:
            candidate: Candidate text to check
            top_k: Number of top results to consider

        Returns:
            Dictionary with:
                - top1_similarity: float - Cosine sim to most similar
                - top1_doc_id: str - Document ID of top match
                - top1_para_id: str - Paragraph ID of top match
                - topk_avg: float - Average of top-k similarities
        """
        results = self.search(candidate, top_k)

        if not results:
            return {
                "top1_similarity": 0.0,
                "top1_doc_id": None,
                "top1_para_id": None,
                "topk_avg": 0.0
            }

        top1 = results[0]
        topk_avg = sum(r["similarity"] for r in results) / len(results)

        return {
            "top1_similarity": top1["similarity"],
            "top1_doc_id": top1["doc_id"],
            "top1_para_id": top1["para_id"],
            "topk_avg": topk_avg
        }

    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self._index is not None

    @property
    def index_size(self) -> int:
        """Get number of vectors in index."""
        if self._index is None:
            return 0
        return self._index.ntotal
