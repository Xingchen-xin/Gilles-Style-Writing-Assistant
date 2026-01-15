"""AI Trace Detection and Correction.

Detects typical AI-generated text patterns and suggests corrections.
This helps avoid AI detection tools by identifying and fixing common
AI writing signatures.
"""
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AITraceResult:
    """Result of AI trace detection."""
    has_ai_traces: bool
    score: float  # 0.0 = human-like, 1.0 = very AI-like
    issues: list[dict] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    corrected_text: Optional[str] = None


# ==============================================================================
# AI Writing Pattern Definitions
# ==============================================================================

# Overused transition words that AI loves
AI_TRANSITION_WORDS = [
    # High frequency AI markers
    (r'\bFurthermore\b', 'Furthermore', 'Also / In addition / [Remove]', 0.15),
    (r'\bMoreover\b', 'Moreover', 'Also / Besides / [Remove]', 0.15),
    (r'\bAdditionally\b', 'Additionally', 'Also / [Remove]', 0.12),
    (r'\bConsequently\b', 'Consequently', 'So / Thus / As a result', 0.10),
    (r'\bNevertheless\b', 'Nevertheless', 'Still / Yet / However', 0.10),
    (r'\bNonetheless\b', 'Nonetheless', 'Still / Yet', 0.10),
    (r'\bHence\b', 'Hence', 'So / Thus', 0.08),
    (r'\bThereby\b', 'Thereby', '[Restructure sentence]', 0.08),
    (r'\bThus\b,?\s', 'Thus', 'So / [Remove]', 0.05),
]

# Phrases that scream "I am AI"
AI_PHRASE_PATTERNS = [
    # Opening phrases
    (r'^In conclusion,?\s', 'In conclusion', '[Just conclude without announcing]', 0.20),
    (r'^To summarize,?\s', 'To summarize', '[Summarize without announcing]', 0.18),
    (r'^In summary,?\s', 'In summary', '[Summarize without announcing]', 0.18),
    (r'^It is worth noting that\s', 'It is worth noting that', '[Remove, just state it]', 0.25),
    (r'^It is important to note that\s', 'It is important to note that', '[Remove, just state it]', 0.25),
    (r'^It should be noted that\s', 'It should be noted that', '[Remove, just state it]', 0.20),
    (r'^Notably,?\s', 'Notably', '[Remove or use "Importantly"]', 0.12),
    (r'^Interestingly,?\s', 'Interestingly', '[Remove, let reader decide]', 0.15),
    (r'^Importantly,?\s', 'Importantly', '[Remove, just state it]', 0.10),

    # Mid-sentence AI markers
    (r'\bplays a (crucial|vital|pivotal|key) role\b', 'plays a crucial/vital role', 'is important for / affects', 0.15),
    (r'\b(a |the )?wide (range|variety|array) of\b', 'a wide range of', 'many / various / different', 0.12),
    (r'\bin the context of\b', 'in the context of', 'for / in / regarding', 0.10),
    (r'\bwith respect to\b', 'with respect to', 'for / about / regarding', 0.08),
    (r'\bin terms of\b', 'in terms of', 'for / by / regarding', 0.08),
    (r'\bhas been shown to\b', 'has been shown to', 'can / does', 0.08),
    (r'\bhas been demonstrated to\b', 'has been demonstrated to', 'can / does', 0.10),
    (r'\bprovides (valuable|important|significant) insights?\b', 'provides valuable insights', 'shows / reveals / explains', 0.15),

    # Filler phrases
    (r'\bit is (essential|crucial|important|vital) to\b', 'it is essential to', '[Restructure: "We must" or remove]', 0.12),
    (r'\bthis (study|paper|research|work) (aims|seeks) to\b', 'this study aims to', '[Use "We" + verb]', 0.10),
    (r'\bthe (findings|results) (suggest|indicate|demonstrate) that\b', 'the findings suggest that', '[Direct statement]', 0.08),
]

# Sentence structure patterns typical of AI
AI_STRUCTURE_PATTERNS = [
    # Enumeration addiction
    (r'First(?:ly)?,.*Second(?:ly)?,.*Third(?:ly)?,', 'First...Second...Third...', 'Vary enumeration style', 0.20),
    (r'\bOn one hand,.*on the other hand\b', 'On one hand...on the other hand', 'Use but/however/while', 0.15),

    # Perfect parallelism (too perfect)
    (r'not only.*but also', 'not only...but also', '[Use occasionally, not repeatedly]', 0.08),

    # Over-hedging
    (r'\b(may|might|could)\b.*\b(may|might|could)\b.*\b(may|might|could)\b', 'Multiple hedges', 'Limit to 2 hedge words per paragraph', 0.25),
    (r'\b(potentially|possibly)\b.*\b(potentially|possibly)\b', 'Multiple "potentially/possibly"', 'Use only once per paragraph', 0.20),
]

# Words AI overuses as "fancy" replacements
AI_FANCY_WORDS = [
    (r'\butilize[sd]?\b', 'utilize', 'use', 0.10),
    (r'\bleverage[sd]?\b', 'leverage', 'use', 0.12),
    (r'\bfacilitate[sd]?\b', 'facilitate', 'help / enable / allow', 0.10),
    (r'\bimplement(?:ed|s|ing)?\b', 'implement', 'use / apply / do', 0.05),
    (r'\bdemonstrate[sd]?\b', 'demonstrate', 'show', 0.08),
    (r'\belucidates?\b', 'elucidate', 'explain / show / clarify', 0.15),
    (r'\belicit[sd]?\b', 'elicit', 'cause / produce / get', 0.12),
    (r'\bcommenc(?:e[sd]?|ing)\b', 'commence', 'start / begin', 0.15),
    (r'\bascertain(?:ed|s)?\b', 'ascertain', 'find / determine / learn', 0.15),
    (r'\bprocure[sd]?\b', 'procure', 'get / obtain', 0.12),
    (r'\bsubsequently\b', 'subsequently', 'then / later / after', 0.10),
    (r'\bprior to\b', 'prior to', 'before', 0.08),
    (r'\bpursuant to\b', 'pursuant to', 'under / following', 0.15),
    (r'\bin order to\b', 'in order to', 'to', 0.08),
]


class AITraceDetector:
    """Detects and corrects AI-generated text patterns."""

    def __init__(
        self,
        transition_threshold: float = 0.3,
        phrase_threshold: float = 0.4,
        structure_threshold: float = 0.3,
        overall_threshold: float = 0.35,
    ):
        """Initialize detector with thresholds.

        Args:
            transition_threshold: Max score for transition words
            phrase_threshold: Max score for AI phrases
            structure_threshold: Max score for structure patterns
            overall_threshold: Max overall score to pass
        """
        self.transition_threshold = transition_threshold
        self.phrase_threshold = phrase_threshold
        self.structure_threshold = structure_threshold
        self.overall_threshold = overall_threshold

    def detect(self, text: str) -> AITraceResult:
        """Detect AI traces in text.

        Args:
            text: Text to analyze

        Returns:
            AITraceResult with detection details
        """
        issues = []
        suggestions = []
        total_score = 0.0

        # Check transition words
        transition_score, transition_issues = self._check_patterns(
            text, AI_TRANSITION_WORDS, "transition_word"
        )
        issues.extend(transition_issues)
        total_score += transition_score * 0.25

        # Check AI phrases
        phrase_score, phrase_issues = self._check_patterns(
            text, AI_PHRASE_PATTERNS, "ai_phrase"
        )
        issues.extend(phrase_issues)
        total_score += phrase_score * 0.35

        # Check structure patterns
        structure_score, structure_issues = self._check_patterns(
            text, AI_STRUCTURE_PATTERNS, "structure"
        )
        issues.extend(structure_issues)
        total_score += structure_score * 0.20

        # Check fancy words
        fancy_score, fancy_issues = self._check_patterns(
            text, AI_FANCY_WORDS, "fancy_word"
        )
        issues.extend(fancy_issues)
        total_score += fancy_score * 0.20

        # Check sentence length uniformity
        uniformity_score = self._check_sentence_uniformity(text)
        if uniformity_score > 0.5:
            issues.append({
                "type": "uniformity",
                "description": "Sentences too uniform in length",
                "score": uniformity_score,
            })
            suggestions.append("Vary sentence length: mix short (5-10 words) and long (30-40 words) sentences")
        total_score += uniformity_score * 0.10

        # Normalize score
        total_score = min(1.0, total_score)

        # Generate suggestions based on issues
        for issue in issues:
            if "suggestion" in issue and issue["suggestion"] not in suggestions:
                suggestions.append(f"{issue['found']}: {issue['suggestion']}")

        # Determine if text has AI traces
        has_traces = total_score >= self.overall_threshold

        return AITraceResult(
            has_ai_traces=has_traces,
            score=round(total_score, 3),
            issues=issues,
            suggestions=suggestions[:10],  # Limit suggestions
            corrected_text=None,  # Set by correct() method
        )

    def correct(self, text: str, result: Optional[AITraceResult] = None) -> str:
        """Auto-correct common AI patterns in text.

        Args:
            text: Text to correct
            result: Optional pre-computed detection result

        Returns:
            Corrected text with reduced AI traces
        """
        if result is None:
            result = self.detect(text)

        corrected = text

        # Apply automatic corrections for clear-cut cases
        corrections = [
            # Remove filler phrases
            (r'\bIt is worth noting that\s+', ''),
            (r'\bIt is important to note that\s+', ''),
            (r'\bIt should be noted that\s+', ''),

            # Simplify fancy words
            (r'\butilize\b', 'use'),
            (r'\butilized\b', 'used'),
            (r'\butilizes\b', 'uses'),
            (r'\bleverage\b', 'use'),
            (r'\bleveraged\b', 'used'),
            (r'\bcommence\b', 'start'),
            (r'\bcommenced\b', 'started'),
            (r'\bprior to\b', 'before'),
            (r'\bin order to\b', 'to'),
            (r'\bsubsequently\b', 'then'),

            # Fix over-formality
            (r'\bFurthermore,\s+', 'Also, '),
            (r'\bMoreover,\s+', 'Also, '),
            (r'\bAdditionally,\s+', 'Also, '),
            (r'\bConsequently,\s+', 'So, '),
            (r'\bNevertheless,\s+', 'However, '),
            (r'\bNonetheless,\s+', 'Still, '),

            # Simplify verbose phrases
            (r'\bplays a crucial role in\b', 'is important for'),
            (r'\bplays a vital role in\b', 'is important for'),
            (r'\bplays a pivotal role in\b', 'is important for'),
            (r'\ba wide range of\b', 'many'),
            (r'\ba wide variety of\b', 'many'),
            (r'\bthe wide array of\b', 'many'),
            (r'\bwith respect to\b', 'for'),
            (r'\bin terms of\b', 'for'),
            (r'\bin the context of\b', 'in'),
        ]

        for pattern, replacement in corrections:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

        # Fix capitalization after removals
        corrected = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), corrected)

        return corrected

    def _check_patterns(
        self,
        text: str,
        patterns: list[tuple],
        pattern_type: str
    ) -> tuple[float, list[dict]]:
        """Check text against pattern list.

        Args:
            text: Text to check
            patterns: List of (regex, name, suggestion, weight) tuples
            pattern_type: Type label for issues

        Returns:
            Tuple of (total_score, list_of_issues)
        """
        issues = []
        total_score = 0.0

        for pattern, name, suggestion, weight in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            if matches:
                # Score increases with frequency
                match_score = weight * min(len(matches), 3)  # Cap at 3 occurrences
                total_score += match_score

                issues.append({
                    "type": pattern_type,
                    "pattern": name,
                    "found": name,
                    "suggestion": suggestion,
                    "count": len(matches),
                    "score": round(match_score, 3),
                    "positions": [m.start() for m in matches[:3]],
                })

        return min(1.0, total_score), issues

    def _check_sentence_uniformity(self, text: str) -> float:
        """Check if sentence lengths are too uniform (AI signature).

        AI tends to produce sentences of similar length.
        Humans vary more.

        Args:
            text: Text to analyze

        Returns:
            Uniformity score (0.0 = varied, 1.0 = very uniform)
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', text.strip())
        sentences = [s for s in sentences if len(s) > 10]

        if len(sentences) < 3:
            return 0.0

        # Calculate word counts
        lengths = [len(s.split()) for s in sentences]

        # Calculate coefficient of variation (CV)
        mean_len = sum(lengths) / len(lengths)
        if mean_len == 0:
            return 0.0

        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        cv = std_dev / mean_len

        # Low CV = uniform (AI-like), High CV = varied (human-like)
        # Human writing typically has CV > 0.4
        # AI writing often has CV < 0.3
        if cv < 0.2:
            return 0.8
        elif cv < 0.3:
            return 0.5
        elif cv < 0.4:
            return 0.2
        else:
            return 0.0

    def get_humanization_tips(self, result: AITraceResult) -> list[str]:
        """Get specific tips to make text more human-like.

        Args:
            result: Detection result

        Returns:
            List of actionable tips
        """
        tips = []

        # Based on score
        if result.score > 0.6:
            tips.append("HIGH AI SCORE: Consider significant restructuring")
        elif result.score > 0.4:
            tips.append("MODERATE AI SCORE: Apply suggested corrections")

        # Specific tips based on issues
        issue_types = set(i["type"] for i in result.issues)

        if "transition_word" in issue_types:
            tips.append("Vary transitions: use 'Also', 'And', or simply remove")

        if "ai_phrase" in issue_types:
            tips.append("Remove filler phrases: state points directly")

        if "fancy_word" in issue_types:
            tips.append("Use simpler words: 'use' not 'utilize', 'show' not 'demonstrate'")

        if "structure" in issue_types:
            tips.append("Break enumeration patterns: don't always use First/Second/Third")

        if "uniformity" in issue_types:
            tips.append("Vary sentence length: add some short punchy sentences")

        # General tips
        tips.extend([
            "Add one minor grammatical quirk (acceptable in academic writing)",
            "Use contractions sparingly in less formal sections",
            "Include a personal observation or interpretation",
        ])

        return tips[:7]  # Limit to 7 tips


# ==============================================================================
# Singleton
# ==============================================================================

_ai_detector: Optional[AITraceDetector] = None


def get_ai_detector() -> AITraceDetector:
    """Get or create AI detector singleton."""
    global _ai_detector
    if _ai_detector is None:
        _ai_detector = AITraceDetector()
    return _ai_detector


# ==============================================================================
# Convenience Functions
# ==============================================================================

def detect_ai_traces(text: str) -> AITraceResult:
    """Detect AI traces in text (convenience function)."""
    return get_ai_detector().detect(text)


def correct_ai_traces(text: str) -> str:
    """Auto-correct AI traces in text (convenience function)."""
    return get_ai_detector().correct(text)


def get_ai_score(text: str) -> float:
    """Get AI score for text (convenience function)."""
    return get_ai_detector().detect(text).score
