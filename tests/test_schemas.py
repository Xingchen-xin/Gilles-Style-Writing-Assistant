"""Tests for Pydantic schemas."""
import pytest
from pydantic import ValidationError
from gswa.api.schemas import (
    Section, Strategy, RewriteConstraints, RewriteRequest,
    SimilarityScores, RewriteVariant, RewriteResponse,
    ChatMessage, ReplyRequest, ReplyResponse, HealthResponse
)


class TestRewriteRequest:
    """Tests for RewriteRequest schema."""

    def test_valid_request(self):
        """Test valid request creation."""
        req = RewriteRequest(
            text="This is a sample paragraph for rewriting.",
            section=Section.RESULTS,
            n_variants=3
        )
        assert req.text == "This is a sample paragraph for rewriting."
        assert req.section == Section.RESULTS
        assert req.n_variants == 3

    def test_text_stripped(self):
        """Test that text is stripped of whitespace."""
        req = RewriteRequest(text="  Some text with whitespace.  ")
        assert req.text == "Some text with whitespace."

    def test_text_min_length(self):
        """Test minimum text length validation."""
        with pytest.raises(ValidationError):
            RewriteRequest(text="Short")

    def test_text_empty_rejected(self):
        """Test that empty/whitespace-only text is rejected."""
        with pytest.raises(ValidationError):
            RewriteRequest(text="          ")

    def test_n_variants_bounds(self):
        """Test n_variants bounds."""
        # Valid range
        req = RewriteRequest(text="Valid paragraph text.", n_variants=1)
        assert req.n_variants == 1

        req = RewriteRequest(text="Valid paragraph text.", n_variants=5)
        assert req.n_variants == 5

        # Invalid range
        with pytest.raises(ValidationError):
            RewriteRequest(text="Valid paragraph text.", n_variants=0)

        with pytest.raises(ValidationError):
            RewriteRequest(text="Valid paragraph text.", n_variants=6)

    def test_default_constraints(self):
        """Test default constraints."""
        req = RewriteRequest(text="Valid paragraph text.")
        assert req.constraints.preserve_numbers is True
        assert req.constraints.no_new_facts is True


class TestSimilarityScores:
    """Tests for SimilarityScores schema."""

    def test_valid_scores(self):
        """Test valid scores creation."""
        scores = SimilarityScores(
            ngram_max_match=5,
            ngram_overlap=0.03,
            embed_top1=0.71
        )
        assert scores.ngram_max_match == 5
        assert scores.ngram_overlap == 0.03
        assert scores.embed_top1 == 0.71

    def test_optional_fields(self):
        """Test optional fields."""
        scores = SimilarityScores(
            ngram_max_match=5,
            ngram_overlap=0.03,
            embed_top1=0.71,
            top1_doc_id="doc_001",
            top1_para_id="para_003"
        )
        assert scores.top1_doc_id == "doc_001"
        assert scores.top1_para_id == "para_003"


class TestRewriteVariant:
    """Tests for RewriteVariant schema."""

    def test_variant_creation(self):
        """Test variant creation."""
        variant = RewriteVariant(
            text="Rewritten text here.",
            strategy=Strategy.A,
            scores=SimilarityScores(
                ngram_max_match=3,
                ngram_overlap=0.02,
                embed_top1=0.65
            )
        )
        assert variant.text == "Rewritten text here."
        assert variant.strategy == Strategy.A
        assert variant.fallback is False

    def test_variant_with_fallback(self):
        """Test variant with fallback."""
        variant = RewriteVariant(
            text="Rewritten after fallback.",
            strategy=Strategy.B,
            scores=SimilarityScores(
                ngram_max_match=8,
                ngram_overlap=0.05,
                embed_top1=0.72
            ),
            fallback=True,
            fallback_reason="n-gram match (15 tokens)"
        )
        assert variant.fallback is True
        assert variant.fallback_reason == "n-gram match (15 tokens)"


class TestChatMessage:
    """Tests for ChatMessage schema."""

    def test_valid_roles(self):
        """Test valid roles."""
        for role in ["user", "assistant", "system"]:
            msg = ChatMessage(role=role, content="Hello")
            assert msg.role == role

    def test_invalid_role(self):
        """Test invalid role rejection."""
        with pytest.raises(ValidationError):
            ChatMessage(role="admin", content="Hello")


class TestReplyRequest:
    """Tests for ReplyRequest schema."""

    def test_valid_request(self):
        """Test valid request."""
        req = ReplyRequest(
            messages=[
                ChatMessage(role="user", content="Hello")
            ],
            max_tokens=256
        )
        assert len(req.messages) == 1
        assert req.max_tokens == 256

    def test_max_tokens_bounds(self):
        """Test max_tokens bounds."""
        with pytest.raises(ValidationError):
            ReplyRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                max_tokens=0
            )

        with pytest.raises(ValidationError):
            ReplyRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                max_tokens=5000
            )


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_health_response(self):
        """Test health response creation."""
        resp = HealthResponse(
            status="healthy",
            llm_server="connected",
            model_loaded="mistral-7b-instruct",
            corpus_paragraphs=1234,
            index_loaded=True
        )
        assert resp.status == "healthy"
        assert resp.corpus_paragraphs == 1234


class TestEnums:
    """Tests for enum types."""

    def test_section_values(self):
        """Test Section enum values."""
        assert Section.ABSTRACT.value == "Abstract"
        assert Section.INTRODUCTION.value == "Introduction"
        assert Section.METHODS.value == "Methods"
        assert Section.RESULTS.value == "Results"
        assert Section.DISCUSSION.value == "Discussion"
        assert Section.CONCLUSION.value == "Conclusion"

    def test_strategy_values(self):
        """Test Strategy enum values."""
        assert Strategy.A.value == "A"
        assert Strategy.B.value == "B"
        assert Strategy.C.value == "C"
        assert Strategy.D.value == "D"
