"""Prompt Construction Service.

Builds prompts for different rewriting strategies and fallback.
Includes anti-AI detection rules and author style fingerprint support.
"""
import json
import logging
from pathlib import Path
from typing import Optional
from gswa.api.schemas import Section, Strategy


logger = logging.getLogger(__name__)

# Path to style fingerprint
STYLE_FINGERPRINT_PATH = Path("data/style/author_fingerprint.json")


# ==============================================================================
# Anti-AI Detection Rules
# ==============================================================================

ANTI_AI_RULES = """
CRITICAL ANTI-AI DETECTION RULES (MUST follow to avoid AI detection):

1. SENTENCE VARIATION:
   - Mix sentence lengths dramatically: include some SHORT sentences (5-10 words) and some LONG ones (30-40 words)
   - Do NOT write all sentences with similar length (15-25 words) - this is a major AI signature
   - Occasionally use a sentence fragment for emphasis

2. FORBIDDEN WORDS/PHRASES (these trigger AI detectors):
   - NEVER use: "Furthermore", "Moreover", "Additionally", "Consequently", "Nevertheless"
   - NEVER use: "It is worth noting that", "It is important to note", "It should be noted"
   - NEVER use: "utilize" (use "use"), "leverage", "facilitate", "elucidate"
   - NEVER use: "a wide range of", "plays a crucial/vital/pivotal role"
   - NEVER start with: "In conclusion,", "To summarize,", "In summary,"

3. STRUCTURE VARIATION:
   - Do NOT use "First... Second... Third..." enumeration
   - Do NOT use perfect parallel structures repeatedly
   - Occasionally start a sentence with "And" or "But" (acceptable in academic writing)
   - Vary paragraph openings - don't always start with the main claim

4. HEDGE WORDS:
   - Limit to MAX 2 hedge words (may/might/could/potentially) per paragraph
   - Don't stack hedges: "may potentially" or "could possibly" sound AI-generated

5. HUMAN TOUCHES:
   - Use simpler words when possible: "show" not "demonstrate", "use" not "utilize"
   - Include occasional contractions in appropriate contexts
   - Don't over-explain - trust the reader

6. PREFERRED TRANSITIONS:
   - Use: "However", "Also", "Yet", "Still", "So", "But", "And"
   - For emphasis: just state the point directly without announcing it"""


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

    def __init__(self):
        """Initialize prompt service."""
        self._style_fingerprint: Optional[dict] = None
        self._load_style_fingerprint()

    def _load_style_fingerprint(self) -> None:
        """Load author style fingerprint if available."""
        if STYLE_FINGERPRINT_PATH.exists():
            try:
                with open(STYLE_FINGERPRINT_PATH, "r", encoding="utf-8") as f:
                    self._style_fingerprint = json.load(f)
                logger.info(f"Loaded style fingerprint for {self._style_fingerprint.get('author_name', 'unknown')}")
            except Exception as e:
                logger.warning(f"Could not load style fingerprint: {e}")

    def _build_style_guidance(self) -> str:
        """Build style guidance from fingerprint."""
        if not self._style_fingerprint:
            return ""

        parts = ["\nAUTHOR STYLE GUIDANCE (from corpus analysis):"]

        # Sentence stats
        ss = self._style_fingerprint.get("sentence_stats", {})
        if ss.get("avg_length"):
            parts.append(f"- Target sentence length: {ss['avg_length']:.0f} words (vary between {ss.get('min_length', 5)}-{ss.get('max_length', 50)})")

        # Structure stats
        st = self._style_fingerprint.get("structure_stats", {})
        if st.get("passive_voice_ratio"):
            pv_pct = st["passive_voice_ratio"] * 100
            if pv_pct > 30:
                parts.append(f"- Use passive voice moderately (~{pv_pct:.0f}% of sentences)")
            else:
                parts.append("- Prefer active voice")

        if st.get("hedge_frequency"):
            parts.append(f"- Hedge frequency: ~{st['hedge_frequency']:.1f} per 100 words")

        # Vocabulary
        vs = self._style_fingerprint.get("vocabulary_stats", {})
        if vs.get("top_verbs"):
            parts.append(f"- Preferred verbs: {', '.join(vs['top_verbs'][:5])}")
        if vs.get("favorite_transitions"):
            parts.append(f"- Preferred transitions: {', '.join(vs['favorite_transitions'][:5])}")
        if vs.get("avoided_words"):
            parts.append(f"- Avoid these words: {', '.join(vs['avoided_words'])}")

        # Style rules
        style_rules = self._style_fingerprint.get("style_rules", [])
        if style_rules:
            parts.append("\nStyle rules:")
            for rule in style_rules[:5]:
                parts.append(f"- {rule}")

        return "\n".join(parts)

    def build_system_prompt(
        self,
        section: Optional[Section] = None,
        is_fallback: bool = False,
        include_anti_ai: bool = True,
        include_style: bool = True,
    ) -> str:
        """Build the system prompt.

        Args:
            section: Paper section type for section-specific guidance
            is_fallback: Whether this is a fallback regeneration
            include_anti_ai: Include anti-AI detection rules
            include_style: Include author style guidance from fingerprint

        Returns:
            Complete system prompt string
        """
        parts = [SYSTEM_PROMPT]

        # Add anti-AI rules (critical for avoiding detection)
        if include_anti_ai:
            parts.append(ANTI_AI_RULES)

        # Add author style guidance from fingerprint
        if include_style:
            style_guidance = self._build_style_guidance()
            if style_guidance:
                parts.append(style_guidance)

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
