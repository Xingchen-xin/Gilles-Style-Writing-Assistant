"""API Routes.

FastAPI route definitions for GSWA.
"""
import json
import logging
import re
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from gswa.api.schemas import (
    RewriteRequest, RewriteResponse,
    ReplyRequest, ReplyResponse,
    HealthResponse, ModelsResponse, ModelInfo,
    FeedbackRequest, FeedbackResponse, FeedbackStats,
    StyleAnalysisRequest, StyleAnalysisResponse, StyleDimension
)
from gswa.services.rewriter import get_rewriter_service
from gswa.services.llm_client import get_llm_client
from gswa.services.similarity import get_similarity_service
from gswa.services.feedback import get_feedback_service
from gswa.services.model_registry import get_model_registry


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["v1"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health.

    Returns:
        Health status including LLM server connection and corpus info
    """
    llm_client = get_llm_client()
    similarity_service = get_similarity_service()
    registry = get_model_registry()

    # Check LLM server
    llm_status = await llm_client.check_health()

    return HealthResponse(
        status="healthy" if llm_status["status"] == "connected" else "degraded",
        llm_server=llm_status["status"],
        model_loaded=llm_status.get("models", [None])[0] if llm_status.get("models") else None,
        corpus_paragraphs=similarity_service.corpus_size,
        index_loaded=similarity_service.is_loaded,
        available_models=len(registry.models),
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available trained models/adapters.

    Returns:
        List of trained LoRA adapters with metadata
    """
    registry = get_model_registry()
    llm_client = get_llm_client()

    models = [ModelInfo(**m.to_dict()) for m in registry.models]
    return ModelsResponse(
        models=models,
        active_model=llm_client.model_name,
        base_model=llm_client.model_name,
    )


@router.post("/models/refresh")
async def refresh_models():
    """Re-scan models directory for new adapters.

    Call this after training completes to discover new models.
    """
    registry = get_model_registry()
    registry.refresh()
    return {"status": "ok", "count": len(registry.models)}


@router.post("/rewrite/variants", response_model=RewriteResponse)
async def rewrite_variants(request: RewriteRequest):
    """Generate multiple rewrite variants.

    Each variant uses a different organizational strategy.
    Similarity is checked against corpus; fallback triggered if too similar.

    Args:
        request: Rewrite request with text and options

    Returns:
        Response with variants and similarity scores

    Raises:
        HTTPException: On processing errors
    """
    try:
        rewriter = await get_rewriter_service()
        return await rewriter.rewrite(request)
    except Exception as e:
        logger.exception("Error in rewrite_variants")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rewrite/variants/stream")
async def rewrite_variants_stream(request: RewriteRequest):
    """Generate rewrite variants with streaming progress."""
    async def event_generator():
        try:
            rewriter = await get_rewriter_service()
            async for event in rewriter.rewrite_stream(request):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.exception("Error in rewrite_variants_stream")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/reply", response_model=ReplyResponse)
async def reply(request: ReplyRequest):
    """Simple chat endpoint for testing/debugging.

    Args:
        request: Chat request with messages

    Returns:
        Model response

    Raises:
        HTTPException: On processing errors
    """
    try:
        llm_client = get_llm_client()
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        content = await llm_client.complete(
            messages=messages,
            max_tokens=request.max_tokens,
        )
        return ReplyResponse(
            content=content,
            model=llm_client.model_name,
        )
    except Exception as e:
        logger.exception("Error in reply")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on generated variants.

    Used to collect preference data for DPO training.

    Args:
        request: Feedback with ratings for each variant

    Returns:
        Confirmation of feedback submission
    """
    try:
        feedback_service = get_feedback_service()
        return feedback_service.submit_feedback(request)
    except Exception as e:
        logger.exception("Error in submit_feedback")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/stats", response_model=FeedbackStats)
async def get_feedback_stats():
    """Get feedback collection statistics.

    Returns:
        Statistics about collected feedback
    """
    try:
        feedback_service = get_feedback_service()
        return feedback_service.get_stats()
    except Exception as e:
        logger.exception("Error in get_feedback_stats")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback/export-dpo")
async def export_dpo_data():
    """Export feedback data in DPO training format.

    Exports collected feedback as (prompt, chosen, rejected) pairs
    suitable for DPO fine-tuning.

    Returns:
        Export status with pair count and output path
    """
    try:
        feedback_service = get_feedback_service()
        output_path = "data/training/dpo_pairs.jsonl"
        pairs_count = feedback_service.export_for_dpo(output_path)
        return {
            "success": True,
            "pairs_count": pairs_count,
            "output_path": output_path,
            "message": f"Exported {pairs_count} DPO training pairs"
        }
    except Exception as e:
        logger.exception("Error in export_dpo_data")
        raise HTTPException(status_code=500, detail=str(e))


# === Style Analysis Endpoints ===

STYLE_ANALYSIS_PROMPT = """You are an expert in scientific writing style analysis. Analyze the following text and compare it to Gilles's academic writing style.

Gilles's writing is characterized by:
- Complex, information-dense sentences (avg 20-30 words)
- Frequent use of discourse markers (indeed, notably, interestingly, remarkably, strikingly)
- Academic hedging (may, might, could, suggests, indicates, appears)
- Scientific precision markers (specifically, exclusively, predominantly, substantially)
- Subordinate clauses with "which", "that", "whereby"
- Transition phrases (taken together, in contrast, of particular interest)
- Balance of passive and active voice

Analyze the text and provide a JSON response with EXACTLY this structure:
{
    "overall_score": <0-100 integer>,
    "summary": "<2-3 sentence assessment>",
    "dimensions": [
        {"name": "Sentence Complexity", "score": <0-10>, "feedback": "<specific feedback>"},
        {"name": "Discourse Markers", "score": <0-10>, "feedback": "<specific feedback>"},
        {"name": "Academic Hedging", "score": <0-10>, "feedback": "<specific feedback>"},
        {"name": "Scientific Precision", "score": <0-10>, "feedback": "<specific feedback>"},
        {"name": "Clause Structure", "score": <0-10>, "feedback": "<specific feedback>"},
        {"name": "Transitions", "score": <0-10>, "feedback": "<specific feedback>"}
    ],
    "suggestions": ["<suggestion 1>", "<suggestion 2>", "<suggestion 3>"]
}

TEXT TO ANALYZE:
"""


@router.post("/style/analyze", response_model=StyleAnalysisResponse)
async def analyze_style(request: StyleAnalysisRequest):
    """Analyze text style using the LLM.

    Compares the input text against Gilles's writing style
    and provides detailed feedback on multiple dimensions.

    Args:
        request: Style analysis request with text

    Returns:
        Detailed style analysis with scores and suggestions
    """
    try:
        llm_client = get_llm_client()

        messages = [
            {"role": "system", "content": "You are a scientific writing style analyzer. Always respond with valid JSON only, no markdown."},
            {"role": "user", "content": STYLE_ANALYSIS_PROMPT + request.text}
        ]

        response_text = await llm_client.complete(
            messages=messages,
            max_tokens=1024,
            temperature=0.3,  # Lower temperature for more consistent analysis
        )

        # Parse JSON response - extract from potential markdown code blocks
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            raise ValueError("No JSON found in model response")

        result = json.loads(json_match.group())

        # Validate and build response
        dimensions = [
            StyleDimension(
                name=d["name"],
                score=max(0, min(10, int(d["score"]))),
                feedback=d["feedback"]
            )
            for d in result.get("dimensions", [])
        ]

        return StyleAnalysisResponse(
            overall_score=max(0, min(100, int(result.get("overall_score", 50)))),
            summary=result.get("summary", "Analysis complete."),
            dimensions=dimensions,
            suggestions=result.get("suggestions", [])[:5],  # Limit to 5 suggestions
            model_used=llm_client.model_name,
        )

    except json.JSONDecodeError as e:
        logger.exception("Failed to parse style analysis JSON")
        raise HTTPException(status_code=500, detail=f"Model returned invalid JSON: {str(e)}")
    except Exception as e:
        logger.exception("Error in analyze_style")
        raise HTTPException(status_code=500, detail=str(e))
