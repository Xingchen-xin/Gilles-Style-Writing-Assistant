"""Rewriter Orchestrator.

Main orchestration logic for rewriting with:
- Similarity checks and fallback
- AI trace detection and correction
- Author style fingerprint integration
"""
import time
import logging
from typing import Optional

from gswa.api.schemas import (
    RewriteRequest, RewriteResponse, RewriteVariant,
    SimilarityScores
)
from gswa.services.llm_client import get_llm_client
from gswa.services.similarity import SimilarityService
from gswa.services.prompt import PromptService, get_prompt_service
from gswa.services.feedback import get_feedback_service
from gswa.utils.ai_detector import get_ai_detector, correct_ai_traces
from gswa.config import get_settings


logger = logging.getLogger(__name__)


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class RewriterService:
    """Orchestrates the rewriting process."""

    def __init__(self):
        """Initialize rewriter service."""
        self.settings = get_settings()
        self.llm_client = get_llm_client()
        self.similarity_service = SimilarityService()
        self.prompt_service = get_prompt_service()
        self.ai_detector = get_ai_detector()
        self.feedback_service = get_feedback_service()

        # AI detection thresholds
        self.ai_score_threshold = 0.5  # Above this triggers correction
        self.enable_ai_correction = True  # Auto-correct AI traces

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
        1. Build prompts for each strategy (with anti-AI rules)
        2. Generate variants
        3. Check similarity for each
        4. Trigger fallback if needed
        5. Check for AI traces and auto-correct
        6. Return results with scores

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

            # Build prompts (now includes anti-AI rules and style fingerprint)
            system_prompt = self.prompt_service.build_system_prompt(
                section=request.section,
                is_fallback=False,
                include_anti_ai=True,
                include_style=True,
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
                    include_anti_ai=True,
                    include_style=True,
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

            # Check for AI traces and auto-correct if enabled
            ai_result = self.ai_detector.detect(generated_text)
            ai_score = ai_result.ai_score

            if self.enable_ai_correction and ai_score > self.ai_score_threshold:
                logger.info(f"AI traces detected (score: {ai_score:.2f}), applying corrections")
                generated_text = correct_ai_traces(generated_text)
                # Re-check after correction
                ai_result = self.ai_detector.detect(generated_text)
                if fallback_reason:
                    fallback_reason += f"; ai_corrected (orig: {ai_score:.2f})"
                else:
                    fallback_reason = f"ai_corrected (orig: {ai_score:.2f})"

            # Add AI score to similarity scores
            scores["ai_score"] = ai_result.ai_score
            scores["ai_issues"] = len(ai_result.pattern_issues)

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
                fallback=fallback or (ai_score > self.ai_score_threshold),
                fallback_reason=fallback_reason,
            ))

        processing_time = int((time.time() - start_time) * 1000)

        # Store session for feedback collection
        import uuid
        session_id = str(uuid.uuid4())[:8]
        self.feedback_service.store_session(
            session_id=session_id,
            input_text=request.text,
            variants=[v.model_dump() for v in variants],
            section=request.section.value if request.section else None,
            model_version=f"{self.settings.vllm_model_name}@v1",
        )

        return RewriteResponse(
            variants=variants,
            model_version=f"{self.settings.vllm_model_name}@v1",
            processing_time_ms=processing_time,
        )

    async def rewrite_stream(self, request: RewriteRequest):
        """Rewrite text with streaming progress events."""
        start_time = time.time()
        strategies = self.prompt_service.get_strategies(
            request.strategies,
            request.n_variants
        )
        total_variants = len(strategies)
        max_tokens = self.settings.max_new_tokens
        variants: list[RewriteVariant] = []

        for i, strategy in enumerate(strategies):
            yield {
                "type": "variant_start",
                "variant_index": i,
                "variant_total": total_variants,
                "strategy": strategy.value,
            }

            system_prompt = self.prompt_service.build_system_prompt(
                section=request.section,
                is_fallback=False,
            )
            user_prompt = self.prompt_service.build_user_prompt(
                text=request.text,
                strategy=strategy,
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            temp = self.settings.temperature_base + (i - total_variants // 2) * self.settings.temperature_variance
            temp = max(0.1, min(1.0, temp))

            tokens_generated = 0
            parts: list[str] = []
            async for delta in self.llm_client.stream_complete(
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            ):
                parts.append(delta)
                tokens_generated += estimate_token_count(delta)
                variant_progress = min(tokens_generated / max_tokens, 1.0) if max_tokens else 0.0
                overall_progress = min((i + variant_progress) / total_variants, 1.0)
                yield {
                    "type": "progress",
                    "variant_index": i,
                    "variant_total": total_variants,
                    "tokens_generated": tokens_generated,
                    "tokens_target": max_tokens,
                    "variant_progress": variant_progress,
                    "overall_progress": overall_progress,
                    "is_fallback": False,
                }

            generated_text = "".join(parts)
            should_fallback, scores = self.similarity_service.check_similarity(generated_text)

            fallback = False
            fallback_reason = None

            if should_fallback:
                fallback = True
                fallback_reason = self._get_fallback_reason(scores)
                yield {
                    "type": "status",
                    "message": f"Fallback triggered for variant {i + 1}, regenerating...",
                    "variant_index": i,
                }

                system_prompt = self.prompt_service.build_system_prompt(
                    section=request.section,
                    is_fallback=True,
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                tokens_generated = 0
                parts = []
                yield {
                    "type": "progress",
                    "variant_index": i,
                    "variant_total": total_variants,
                    "tokens_generated": tokens_generated,
                    "tokens_target": max_tokens,
                    "variant_progress": 0.0,
                    "overall_progress": min(i / total_variants, 1.0),
                    "is_fallback": True,
                }

                async for delta in self.llm_client.stream_complete(
                    messages=messages,
                    temperature=temp + 0.1,
                    max_tokens=max_tokens,
                ):
                    parts.append(delta)
                    tokens_generated += estimate_token_count(delta)
                    variant_progress = min(tokens_generated / max_tokens, 1.0) if max_tokens else 0.0
                    overall_progress = min((i + variant_progress) / total_variants, 1.0)
                    yield {
                        "type": "progress",
                        "variant_index": i,
                        "variant_total": total_variants,
                        "tokens_generated": tokens_generated,
                        "tokens_target": max_tokens,
                        "variant_progress": variant_progress,
                        "overall_progress": overall_progress,
                        "is_fallback": True,
                    }

                generated_text = "".join(parts)
                _, scores = self.similarity_service.check_similarity(generated_text)

            yield {
                "type": "progress",
                "variant_index": i,
                "variant_total": total_variants,
                "tokens_generated": max_tokens,
                "tokens_target": max_tokens,
                "variant_progress": 1.0,
                "overall_progress": min((i + 1) / total_variants, 1.0),
                "is_fallback": fallback,
            }

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
        response = RewriteResponse(
            variants=variants,
            model_version=f"{self.settings.vllm_model_name}@v1",
            processing_time_ms=processing_time,
        )
        yield {
            "type": "result",
            "data": response.model_dump(mode="json"),
        }

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
