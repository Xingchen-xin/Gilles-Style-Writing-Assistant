"""API Routes.

FastAPI route definitions for GSWA.
"""
import logging
from fastapi import APIRouter, HTTPException

from gswa.api.schemas import (
    RewriteRequest, RewriteResponse,
    ReplyRequest, ReplyResponse,
    HealthResponse
)
from gswa.services.rewriter import get_rewriter_service
from gswa.services.llm_client import get_llm_client
from gswa.services.similarity import get_similarity_service


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

    # Check LLM server
    llm_status = await llm_client.check_health()

    return HealthResponse(
        status="healthy" if llm_status["status"] == "connected" else "degraded",
        llm_server=llm_status["status"],
        model_loaded=llm_status.get("models", [None])[0] if llm_status.get("models") else None,
        corpus_paragraphs=similarity_service.corpus_size,
        index_loaded=similarity_service.is_loaded,
    )


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
