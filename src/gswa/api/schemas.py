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
    model: Optional[str] = Field(None, description="Model/adapter ID to use (from /v1/models)")

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
    ai_score: Optional[float] = Field(None, description="AI trace detection score (0=human-like, 1=AI-like)")
    ai_issues: Optional[int] = Field(None, description="Number of AI trace issues detected")


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
    available_models: int = 0


class ModelInfo(BaseModel):
    """Information about a trained model/adapter."""
    id: str = Field(..., description="Unique model identifier (directory name)")
    display_name: str = Field(..., description="Human-readable name")
    base_model: str = Field(..., description="HuggingFace base model ID")
    model_short: str = Field(..., description="Short base model name")
    adapter_path: str = Field(..., description="Path to LoRA adapter")
    quantization: str = Field(default="4bit")
    lora_r: int = Field(default=16)
    epochs: int = Field(default=3)
    started_at: str = Field(default="")
    completed_at: str = Field(default="")


class ModelsResponse(BaseModel):
    """Response body for /v1/models."""
    models: list[ModelInfo]
    active_model: Optional[str] = Field(None, description="Currently active model/adapter")
    base_model: Optional[str] = Field(None, description="Base model being served")


# === Feedback Schemas (for DPO training) ===

class FeedbackType(str, Enum):
    """Types of feedback."""
    BEST = "best"          # User selected this as the best variant
    GOOD = "good"          # Acceptable but not the best
    BAD = "bad"            # Not acceptable
    EDITED = "edited"      # User edited the output
    AI_LIKE = "ai_like"    # Sounds too AI-generated (for DPO rejection)


class VariantFeedback(BaseModel):
    """Feedback for a single variant."""
    variant_index: int = Field(..., ge=0, le=4, description="Index of the variant (0-4)")
    feedback_type: FeedbackType
    edited_text: Optional[str] = Field(None, description="User's edited version (if edited)")


class FeedbackRequest(BaseModel):
    """Request body for /v1/feedback."""
    session_id: str = Field(..., min_length=1, description="Unique session identifier")
    input_text: str = Field(..., min_length=10, description="Original input text")
    section: Optional[Section] = None
    variants: list[VariantFeedback] = Field(..., min_length=1, description="Feedback for each variant")

    # Optional metadata
    user_notes: Optional[str] = Field(None, max_length=1000, description="Additional notes from user")


class FeedbackResponse(BaseModel):
    """Response body for /v1/feedback."""
    success: bool
    feedback_id: str
    message: str


class FeedbackStats(BaseModel):
    """Statistics about collected feedback."""
    total_sessions: int
    total_variants_rated: int
    best_count: int
    good_count: int
    bad_count: int
    edited_count: int
    ai_like_count: int = 0  # Count of AI-like flagged variants


# === Style Analysis Schemas (Model-based evaluation) ===

class StyleAnalysisRequest(BaseModel):
    """Request body for /v1/style/analyze."""
    text: str = Field(..., min_length=50, max_length=10000)
    reference_style: str = Field(
        default="gilles",
        description="Reference style to compare against (currently only 'gilles')"
    )


class StyleDimension(BaseModel):
    """Score for a single style dimension."""
    name: str = Field(..., description="Dimension name (e.g., 'Sentence Complexity')")
    score: int = Field(..., ge=0, le=10, description="Score from 0-10")
    feedback: str = Field(..., description="Specific feedback for this dimension")


class StyleAnalysisResponse(BaseModel):
    """Response body for /v1/style/analyze."""
    overall_score: int = Field(..., ge=0, le=100, description="Overall style match score 0-100")
    summary: str = Field(..., description="Brief overall assessment")
    dimensions: list[StyleDimension] = Field(..., description="Scores for each style dimension")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    model_used: str = Field(..., description="Model used for analysis")
