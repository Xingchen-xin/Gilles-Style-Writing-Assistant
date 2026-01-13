"""Prompt Construction Service.

Builds prompts for different rewriting strategies and fallback.
"""
from typing import Optional
from gswa.api.schemas import Section, Strategy


# System prompt (style card)
SYSTEM_PROMPT = """You are a scientific paper rewriter. Rewrite the user's paragraph in a style consistent with Gilles's published papers.

HARD CONSTRAINTS (MUST follow):
- Preserve meaning EXACTLY: do not change numbers, units, experimental conditions, comparisons, or conclusion strength
- Do not introduce new facts not present in the input
- Do not strengthen hedged claims (may/suggest → demonstrate) or weaken strong claims
- Avoid copying long phrases (>8 words) from reference corpus; prefer new sentence structures

Output ONLY the rewritten paragraph, no explanations or preamble."""


# Strategy templates
STRATEGY_TEMPLATES = {
    Strategy.A: "Rewrite with the main claim in the FIRST sentence, then provide supporting details and qualifiers.",
    Strategy.B: "Rewrite starting with brief context/motivation, then introduce the main claim, then qualifiers.",
    Strategy.C: "Rewrite starting with the experimental setup/approach, then report the key finding, then interpretation.",
    Strategy.D: "Rewrite starting with cautious framing/limitations, then state the key claim conservatively, then implications.",
}


# Section-specific guidance
SECTION_GUIDANCE = {
    Section.ABSTRACT: "Keep it concise and self-contained. Include background, objective, key results, and conclusion.",
    Section.INTRODUCTION: "Build from broad context to specific hypothesis. End with clear objectives.",
    Section.METHODS: "Be precise and reproducible. Use past tense and passive voice appropriately.",
    Section.RESULTS: "Present findings objectively. Let data speak without over-interpretation.",
    Section.DISCUSSION: "Interpret results in context. Acknowledge limitations. Connect to broader implications.",
    Section.CONCLUSION: "Summarize key findings. Emphasize significance. Suggest future directions.",
}


# Fallback prompt for stronger diversification
FALLBACK_PROMPT = """IMPORTANT: Your previous rewrite was too similar to existing text.

Rewrite again with SIGNIFICANTLY different sentence structures:
- Split or merge sentences differently
- Change active/passive voice
- Reorder clauses and ideas
- Use different transition words
- AVOID any phrase longer than 6 consecutive words from the original

Preserve the exact same meaning and all numerical values."""


class PromptService:
    """Builds prompts for rewriting tasks."""

    def build_system_prompt(
        self,
        section: Optional[Section] = None,
        is_fallback: bool = False,
    ) -> str:
        """Build the system prompt.

        Args:
            section: Paper section type for section-specific guidance
            is_fallback: Whether this is a fallback regeneration

        Returns:
            Complete system prompt string
        """
        parts = [SYSTEM_PROMPT]

        if section and section in SECTION_GUIDANCE:
            parts.append(f"\nSection-specific guidance ({section.value}):")
            parts.append(SECTION_GUIDANCE[section])

        if is_fallback:
            parts.append(f"\n{FALLBACK_PROMPT}")

        return "\n".join(parts)

    def build_user_prompt(
        self,
        text: str,
        strategy: Strategy,
    ) -> str:
        """Build the user prompt with strategy.

        Args:
            text: The paragraph text to rewrite
            strategy: The rewriting strategy to use

        Returns:
            Complete user prompt string
        """
        strategy_instruction = STRATEGY_TEMPLATES.get(strategy, STRATEGY_TEMPLATES[Strategy.A])

        return f"""{strategy_instruction}

Here is the paragraph to rewrite:

{text}"""

    def get_strategies(
        self,
        requested: Optional[list[Strategy]],
        n: int,
    ) -> list[Strategy]:
        """Get list of strategies to use for n variants.

        Args:
            requested: Optional list of requested strategies
            n: Number of variants needed

        Returns:
            List of strategies to use
        """
        all_strategies = [Strategy.A, Strategy.B, Strategy.C, Strategy.D]

        if requested:
            # Use requested strategies, cycling if needed
            strategies = []
            for i in range(n):
                strategies.append(requested[i % len(requested)])
            return strategies

        # Default: use first n strategies
        return all_strategies[:n]


# Singleton instance
_prompt_service: Optional[PromptService] = None


def get_prompt_service() -> PromptService:
    """Get or create prompt service singleton."""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = PromptService()
    return _prompt_service
