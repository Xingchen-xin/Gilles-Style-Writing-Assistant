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
CRITICAL: WRITE LIKE A HUMAN (Based on AI detection research)

AI detectors analyze: (1) sentence variation, (2) vocabulary predictability, (3) style consistency.
To avoid detection, follow these research-backed strategies:

== MOST IMPORTANT: SENTENCE BURSTINESS (30% of detection weight) ==
Human writing has HIGH variation in sentence length. AI writing is UNIFORM.

REQUIRED pattern:
- Include at least ONE short sentence (5-10 words) per paragraph
- Include at least ONE longer sentence (25-35 words) per paragraph
- Coefficient of variation should be > 0.4 (standard deviation / mean)

Example (GOOD - varied):
"The results were striking. We observed a 47% increase in binding affinity when the modified peptide was introduced, suggesting that the conformational change plays a key role in receptor recognition. This matters."

Example (BAD - uniform):
"The results showed a significant increase in binding affinity. The modified peptide demonstrated enhanced receptor recognition. The conformational change appeared to be important."

== VOCABULARY UNPREDICTABILITY (25% weight) ==
- Avoid the most statistically likely next word
- Use unexpected but accurate synonyms occasionally
- Don't always use the "perfect" academic word

== STYLE CONSISTENCY WITH AUTHOR (15% weight) ==
- Match the sentence length distribution of the author's corpus
- Use the author's preferred verbs and transitions
- Match passive/active voice ratio

== LOWER PRIORITY: Word Substitutions (10% weight) ==
Prefer simpler words:
- "use" not "utilize"
- "show" not "demonstrate"
- "help" not "facilitate"
- "Also" not "Furthermore/Moreover"

Remove filler phrases entirely:
- Delete "It is worth noting that"
- Delete "It is important to note"
- Just state the point directly"""


# System prompt (style card)
SYSTEM_PROMPT = """You are a scientific paper rewriter. Rewrite the user's paragraph in a style consistent with Gilles's published papers.

HARD CONSTRAINTS (MUST follow):
- Preserve meaning EXACTLY: do not change numbers, units, experimental conditions, comparisons, or conclusion strength
- Do not introduce new facts not present in the input
- Do not strengthen hedged claims (may/suggest → demonstrate) or weaken strong claims
- Avoid copying long phrases (>8 words) from reference corpus; prefer new sentence structures

Output ONLY the rewritten paragraph, no explanations or preamble."""

# Simplified instructions for LoRA fine-tuned models
# Must match the training data format exactly
# Training used: instruction + "\n\n" + input (no system prompt)
LORA_INSTRUCTIONS = [
    "Enhance the academic tone and precision of this text:",
    "Paraphrase this research paragraph in formal academic English:",
    "Edit this scientific text for publication quality:",
    "Rewrite this scientific text in a more polished academic style:",
]


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
        for_lora: bool = False,
    ) -> str:
        """Build the system prompt.

        Args:
            section: Paper section type for section-specific guidance
            is_fallback: Whether this is a fallback regeneration
            include_anti_ai: Include anti-AI detection rules
            include_style: Include author style guidance from fingerprint
            for_lora: Use simplified prompt for LoRA fine-tuned models

        Returns:
            Complete system prompt string
        """
        # For LoRA models, return empty - we'll use a direct instruction in user message
        # This matches the training format: instruction + input (no system prompt)
        if for_lora:
            return ""

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
        for_lora: bool = False,
        variant_index: int = 0,
    ) -> str:
        """Build the user prompt with strategy.

        Args:
            text: The paragraph text to rewrite
            strategy: The rewriting strategy to use
            for_lora: Use simple format matching LoRA training data
            variant_index: Index for selecting instruction variation

        Returns:
            Complete user prompt string
        """
        if for_lora:
            # Match exact training format: instruction + "\n\n" + input
            # No strategy, no extra formatting - just like training data
            instruction = LORA_INSTRUCTIONS[variant_index % len(LORA_INSTRUCTIONS)]
            return f"{instruction}\n\n{text}"

        strategy_instruction = STRATEGY_TEMPLATES.get(
            strategy, STRATEGY_TEMPLATES[Strategy.A]
        )

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
