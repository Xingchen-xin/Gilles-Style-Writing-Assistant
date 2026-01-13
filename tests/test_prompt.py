"""Tests for prompt service."""
import pytest
from gswa.services.prompt import (
    PromptService, SYSTEM_PROMPT, FALLBACK_PROMPT,
    STRATEGY_TEMPLATES, SECTION_GUIDANCE
)
from gswa.api.schemas import Section, Strategy


@pytest.fixture
def prompt_service():
    """Create prompt service for tests."""
    return PromptService()


class TestBuildSystemPrompt:
    """Tests for build_system_prompt method."""

    def test_basic_system_prompt(self, prompt_service):
        """Test basic system prompt without section."""
        prompt = prompt_service.build_system_prompt()
        assert SYSTEM_PROMPT in prompt
        assert FALLBACK_PROMPT not in prompt

    def test_system_prompt_with_section(self, prompt_service):
        """Test system prompt with section guidance."""
        prompt = prompt_service.build_system_prompt(section=Section.RESULTS)
        assert SYSTEM_PROMPT in prompt
        assert "Results" in prompt
        assert SECTION_GUIDANCE[Section.RESULTS] in prompt

    def test_system_prompt_with_fallback(self, prompt_service):
        """Test system prompt with fallback enabled."""
        prompt = prompt_service.build_system_prompt(is_fallback=True)
        assert SYSTEM_PROMPT in prompt
        assert FALLBACK_PROMPT in prompt

    def test_system_prompt_with_section_and_fallback(self, prompt_service):
        """Test system prompt with both section and fallback."""
        prompt = prompt_service.build_system_prompt(
            section=Section.DISCUSSION,
            is_fallback=True
        )
        assert SYSTEM_PROMPT in prompt
        assert "Discussion" in prompt
        assert FALLBACK_PROMPT in prompt


class TestBuildUserPrompt:
    """Tests for build_user_prompt method."""

    def test_user_prompt_strategy_a(self, prompt_service):
        """Test user prompt with strategy A."""
        text = "This is a test paragraph."
        prompt = prompt_service.build_user_prompt(text, Strategy.A)

        assert STRATEGY_TEMPLATES[Strategy.A] in prompt
        assert text in prompt

    def test_user_prompt_strategy_b(self, prompt_service):
        """Test user prompt with strategy B."""
        text = "Another test paragraph."
        prompt = prompt_service.build_user_prompt(text, Strategy.B)

        assert STRATEGY_TEMPLATES[Strategy.B] in prompt
        assert text in prompt

    def test_user_prompt_all_strategies(self, prompt_service):
        """Test user prompt with all strategies."""
        text = "Test paragraph."
        for strategy in Strategy:
            prompt = prompt_service.build_user_prompt(text, strategy)
            assert STRATEGY_TEMPLATES[strategy] in prompt
            assert text in prompt


class TestGetStrategies:
    """Tests for get_strategies method."""

    def test_default_strategies(self, prompt_service):
        """Test default strategy selection."""
        strategies = prompt_service.get_strategies(None, 3)
        assert len(strategies) == 3
        assert strategies == [Strategy.A, Strategy.B, Strategy.C]

    def test_default_strategies_max(self, prompt_service):
        """Test default strategy selection for max variants."""
        strategies = prompt_service.get_strategies(None, 5)
        # Only 4 strategies defined, so returns all 4
        assert len(strategies) == 4
        assert strategies == [Strategy.A, Strategy.B, Strategy.C, Strategy.D]

    def test_requested_strategies(self, prompt_service):
        """Test with requested strategies."""
        requested = [Strategy.B, Strategy.D]
        strategies = prompt_service.get_strategies(requested, 4)

        assert len(strategies) == 4
        # Should cycle through requested strategies
        assert strategies[0] == Strategy.B
        assert strategies[1] == Strategy.D
        assert strategies[2] == Strategy.B
        assert strategies[3] == Strategy.D

    def test_single_requested_strategy(self, prompt_service):
        """Test with single requested strategy."""
        requested = [Strategy.C]
        strategies = prompt_service.get_strategies(requested, 3)

        assert len(strategies) == 3
        assert all(s == Strategy.C for s in strategies)


class TestPromptContent:
    """Tests for prompt content quality."""

    def test_system_prompt_contains_constraints(self, prompt_service):
        """Test that system prompt contains key constraints."""
        prompt = prompt_service.build_system_prompt()

        # Check for critical constraints
        assert "Preserve meaning" in prompt
        assert "numbers" in prompt.lower()
        assert "facts" in prompt.lower()

    def test_fallback_prompt_contains_diversification(self, prompt_service):
        """Test that fallback prompt mentions diversification."""
        prompt = prompt_service.build_system_prompt(is_fallback=True)

        assert "different sentence structures" in prompt.lower()
        assert "6 consecutive words" in prompt or "six consecutive words" in prompt.lower()

    def test_all_sections_have_guidance(self, prompt_service):
        """Test that all sections have guidance defined."""
        for section in Section:
            assert section in SECTION_GUIDANCE
            prompt = prompt_service.build_system_prompt(section=section)
            assert section.value in prompt
