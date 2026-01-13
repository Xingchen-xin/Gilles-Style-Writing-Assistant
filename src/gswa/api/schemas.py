"""API Request/Response Schemas.

All data structures for the GSWA API endpoints.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Section(str, Enum):
    """Paper section types."""
    ABSTRACT = "Abstract"
    INTRODUCTION = "Introduction"
    METHODS = "Methods"
    RESULTS = "Results"
    DISCUSSION = "Discussion"
    CONCLUSION = "Conclusion"


class Strategy(str, Enum):
    """Rewriting strategies for diversification."""
    A = "A"  # Conclusion-first
    B = "B"  # Background-first
    C = "C"  # Methods-first
    D = "D"  # Cautious-first


class RewriteConstraints(BaseModel):
    """Constraints for rewriting."""
    preserve_numbers: bool = True
    no_new_facts: bool = True


class RewriteRequest(BaseModel):
    """Request body for /v1/rewrite/variants."""
    text: str = Field(..., min_length=10, max_length=10000)
    section: Optional[Section] = None
    n_variants: int = Field(default=3, ge=1, le=5)
    strategies: Optional[list[Strategy]] = None
    constraints: RewriteConstraints = Field(default_factory=RewriteConstraints)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class SimilarityScores(BaseModel):
    """Similarity metrics for a variant."""
    ngram_max_match: int = Field(..., description="Longest consecutive n-gram match")
    ngram_overlap: float = Field(..., description="N-gram overlap ratio")
    embed_top1: float = Field(..., description="Cosine similarity to most similar corpus paragraph")
    top1_doc_id: Optional[str] = None
    top1_para_id: Optional[str] = None


class RewriteVariant(BaseModel):
    """A single rewrite variant."""
    text: str
    strategy: Strategy
    scores: SimilarityScores
    fallback: bool = False
    fallback_reason: Optional[str] = None


class RewriteResponse(BaseModel):
    """Response body for /v1/rewrite/variants."""
    variants: list[RewriteVariant]
    model_version: str
    processing_time_ms: int


class ChatMessage(BaseModel):
    """A chat message."""
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ReplyRequest(BaseModel):
    """Request body for /v1/reply."""
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)


class ReplyResponse(BaseModel):
    """Response body for /v1/reply."""
    content: str
    model: str


class HealthResponse(BaseModel):
    """Response body for /v1/health."""
    status: str
    llm_server: str
    model_loaded: Optional[str]
    corpus_paragraphs: int
    index_loaded: bool
