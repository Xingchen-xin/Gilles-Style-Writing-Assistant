"""Rewriter Orchestrator.

Main orchestration logic for rewriting with similarity checks and fallback.
"""
import time
import logging
from typing import Optional

from gswa.api.schemas import (
    RewriteRequest, RewriteResponse, RewriteVariant,
    SimilarityScores, Strategy
)
from gswa.services.llm_client import get_llm_client
from gswa.services.similarity import SimilarityService
from gswa.services.prompt import PromptService
from gswa.config import get_settings


logger = logging.getLogger(__name__)


class RewriterService:
    """Orchestrates the rewriting process."""

    def __init__(self):
        """Initialize rewriter service."""
        self.settings = get_settings()
        self.llm_client = get_llm_client()
        self.similarity_service = SimilarityService()
        self.prompt_service = PromptService()

    async def initialize(self) -> None:
        """Initialize services (load corpus, etc.)."""
        try:
            # Try to load pre-built index first
            if not self.similarity_service.load_index():
                # Fall back to loading corpus directly
                count = self.similarity_service.load_corpus()
                if count > 0:
                    logger.info(f"Loaded {count} paragraphs from corpus")
                else:
                    logger.info("No corpus found, similarity checking will be disabled")
        except Exception as e:
            logger.warning(f"Could not load similarity index: {e}")

    async def rewrite(self, request: RewriteRequest) -> RewriteResponse:
        """Rewrite text with multiple variants.

        Process:
        1. Build prompts for each strategy
        2. Generate variants
        3. Check similarity for each
        4. Trigger fallback if needed
        5. Return results with scores

        Args:
            request: Rewrite request with text and options

        Returns:
            Response with variants and metadata
        """
        start_time = time.time()

        # Determine strategies to use
        strategies = self.prompt_service.get_strategies(
            request.strategies,
            request.n_variants
        )

        variants: list[RewriteVariant] = []

        for i, strategy in enumerate(strategies):
            logger.info(f"Generating variant {i+1}/{len(strategies)} with strategy {strategy.value}")

            # Build prompts
            system_prompt = self.prompt_service.build_system_prompt(
                section=request.section,
                is_fallback=False,
            )
            user_prompt = self.prompt_service.build_user_prompt(
                text=request.text,
                strategy=strategy,
            )

            # Generate initial variant
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            temp = self.settings.temperature_base + (i - len(strategies) // 2) * self.settings.temperature_variance
            temp = max(0.1, min(1.0, temp))

            generated_text = await self.llm_client.complete(
                messages=messages,
                temperature=temp,
            )

            # Check similarity
            should_fallback, scores = self.similarity_service.check_similarity(generated_text)

            fallback = False
            fallback_reason = None

            # Trigger fallback if needed (only once per variant)
            if should_fallback:
                fallback = True
                fallback_reason = self._get_fallback_reason(scores)
                logger.info(f"Triggering fallback for variant {i+1}: {fallback_reason}")

                # Regenerate with fallback prompt
                system_prompt = self.prompt_service.build_system_prompt(
                    section=request.section,
                    is_fallback=True,
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                generated_text = await self.llm_client.complete(
                    messages=messages,
                    temperature=temp + 0.1,  # Slightly higher temp for fallback
                )

                # Re-check similarity
                _, scores = self.similarity_service.check_similarity(generated_text)

            variants.append(RewriteVariant(
                text=generated_text.strip(),
                strategy=strategy,
                scores=SimilarityScores(
                    ngram_max_match=scores["ngram_max_match"],
                    ngram_overlap=scores["ngram_overlap"],
                    embed_top1=scores["embed_top1"],
                    top1_doc_id=scores.get("top1_doc_id"),
                    top1_para_id=scores.get("top1_para_id"),
                ),
                fallback=fallback,
                fallback_reason=fallback_reason,
            ))

        processing_time = int((time.time() - start_time) * 1000)

        return RewriteResponse(
            variants=variants,
            model_version=f"{self.settings.vllm_model_name}@v1",
            processing_time_ms=processing_time,
        )

    def _get_fallback_reason(self, scores: dict) -> str:
        """Generate human-readable fallback reason.

        Args:
            scores: Similarity scores dictionary

        Returns:
            Human-readable reason string
        """
        reasons = []
        if scores["ngram_max_match"] >= self.settings.threshold_ngram_max_match:
            reasons.append(f"n-gram match ({scores['ngram_max_match']} tokens)")
        if scores["embed_top1"] >= self.settings.threshold_embed_top1:
            reasons.append(f"embedding similarity ({scores['embed_top1']:.2f})")
        return "; ".join(reasons) if reasons else "threshold exceeded"


# Singleton instance
_rewriter_service: Optional[RewriterService] = None


async def get_rewriter_service() -> RewriterService:
    """Get or create rewriter service singleton."""
    global _rewriter_service
    if _rewriter_service is None:
        _rewriter_service = RewriterService()
        await _rewriter_service.initialize()
    return _rewriter_service
