"""Combined Similarity Service.

Orchestrates n-gram and embedding similarity checks.
"""
from typing import Optional, Tuple, Set
from pathlib import Path
import json
import logging

from gswa.utils.ngram import build_ngram_index, compute_ngram_overlap
from gswa.config import get_settings

logger = logging.getLogger(__name__)


class SimilarityService:
    """Combined n-gram and embedding similarity service."""

    def __init__(self):
        """Initialize similarity service."""
        self.settings = get_settings()
        self._ngram_index: Optional[Set] = None
        self._embedding_service = None
        self._corpus_texts: list[str] = []
        self._corpus_loaded = False

    def load_corpus(self, corpus_path: Optional[str] = None) -> int:
        """Load corpus from JSONL files.

        Args:
            corpus_path: Path to corpus directory (defaults to config)

        Returns:
            Number of paragraphs loaded
        """
        corpus_path = Path(corpus_path or self.settings.corpus_path)

        texts = []
        doc_ids = []
        para_ids = []

        # Load from JSONL files
        if corpus_path.exists():
            for jsonl_file in corpus_path.glob("*.jsonl"):
                with open(jsonl_file) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            texts.append(item["text"])
                            doc_ids.append(item.get("doc_id", "unknown"))
                            para_ids.append(item.get("para_id", "unknown"))
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Skipping invalid line: {e}")

        if texts:
            # Build n-gram index
            self._ngram_index = build_ngram_index(texts, n=8)
            logger.info(f"Built n-gram index with {len(self._ngram_index)} unique 8-grams")

            # Build embedding index (lazy load)
            try:
                from gswa.utils.embedding import EmbeddingService
                self._embedding_service = EmbeddingService(
                    model_name=self.settings.embedding_model
                )
                self._embedding_service.build_index(texts, doc_ids, para_ids)
                logger.info(f"Built embedding index with {len(texts)} paragraphs")
            except ImportError:
                logger.warning("sentence-transformers not available, skipping embedding index")
                self._embedding_service = None

            self._corpus_texts = texts
            self._corpus_loaded = True

        return len(texts)

    def load_index(self, index_path: Optional[str] = None) -> bool:
        """Load pre-built index from disk.

        Args:
            index_path: Path to index directory (defaults to config)

        Returns:
            True if successfully loaded, False otherwise
        """
        index_path = Path(index_path or self.settings.index_path)

        # Load n-gram index
        ngram_file = index_path / "ngrams.json"
        if ngram_file.exists():
            try:
                with open(ngram_file) as f:
                    ngram_list = json.load(f)
                    self._ngram_index = set(tuple(ng) for ng in ngram_list)
                logger.info(f"Loaded n-gram index with {len(self._ngram_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load n-gram index: {e}")

        # Load embedding index
        try:
            from gswa.utils.embedding import EmbeddingService
            self._embedding_service = EmbeddingService(
                model_name=self.settings.embedding_model
            )
            if self._embedding_service.load_index(str(index_path)):
                self._corpus_loaded = True
                logger.info("Loaded embedding index from disk")
                return True
        except ImportError:
            logger.warning("sentence-transformers not available, skipping embedding index")
            self._embedding_service = None

        # If we at least have n-gram index, consider it loaded
        if self._ngram_index:
            self._corpus_loaded = True
            return True

        return False

    def save_index(self, index_path: Optional[str] = None) -> None:
        """Save index to disk.

        Args:
            index_path: Path to save index (defaults to config)
        """
        index_path = Path(index_path or self.settings.index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        # Save n-gram index
        if self._ngram_index:
            ngram_file = index_path / "ngrams.json"
            with open(ngram_file, "w") as f:
                json.dump([list(ng) for ng in self._ngram_index], f)
            logger.info(f"Saved n-gram index to {ngram_file}")

        # Save embedding index
        if self._embedding_service and self._embedding_service.is_loaded:
            self._embedding_service.save_index(str(index_path))
            logger.info(f"Saved embedding index to {index_path}")

    def check_similarity(self, candidate: str) -> Tuple[bool, dict]:
        """Check candidate text against corpus.

        Args:
            candidate: Generated text to check

        Returns:
            Tuple of (should_fallback, scores_dict)
        """
        scores = {
            "ngram_max_match": 0,
            "ngram_overlap": 0.0,
            "embed_top1": 0.0,
            "top1_doc_id": None,
            "top1_para_id": None,
        }

        # N-gram check
        if self._ngram_index:
            ngram_result = compute_ngram_overlap(candidate, self._ngram_index)
            scores["ngram_max_match"] = ngram_result["max_consecutive_match"]
            scores["ngram_overlap"] = ngram_result["overlap_ratio"]

        # Embedding check
        if self._embedding_service and self._embedding_service.is_loaded:
            embed_result = self._embedding_service.compute_similarity(candidate)
            scores["embed_top1"] = embed_result["top1_similarity"]
            scores["top1_doc_id"] = embed_result["top1_doc_id"]
            scores["top1_para_id"] = embed_result["top1_para_id"]

        # Determine if fallback needed
        should_fallback = (
            scores["ngram_max_match"] >= self.settings.threshold_ngram_max_match or
            scores["embed_top1"] >= self.settings.threshold_embed_top1
        )

        return should_fallback, scores

    @property
    def is_loaded(self) -> bool:
        """Check if corpus/index is loaded."""
        return self._corpus_loaded

    @property
    def corpus_size(self) -> int:
        """Get number of paragraphs in corpus."""
        if self._embedding_service and self._embedding_service.is_loaded:
            return self._embedding_service.index_size
        return len(self._corpus_texts) if self._corpus_texts else 0


# Singleton instance
_similarity_service: Optional[SimilarityService] = None


def get_similarity_service() -> SimilarityService:
    """Get or create similarity service singleton."""
    global _similarity_service
    if _similarity_service is None:
        _similarity_service = SimilarityService()
    return _similarity_service
