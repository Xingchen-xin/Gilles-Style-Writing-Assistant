"""Scientific AI Detection Module.

Based on academic research on AI text detection:
- Perplexity: How predictable is the text (lower = more AI-like)
- Burstiness: Sentence length variation (lower = more AI-like)
- Vocabulary Diversity: Type-token ratio and lexical richness
- Style Consistency: Match against author fingerprint

References:
- GPTZero: Perplexity and Burstiness metrics
- DetectGPT: Probability curvature analysis
- StyloAI: 31 stylometric features for AI detection
- DIPPER: Paraphrase attack research
"""
import re
import math
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ==============================================================================
# Configuration
# ==============================================================================

STYLE_FINGERPRINT_PATH = Path("data/style/author_fingerprint.json")

# Thresholds based on research
# Human text typically: perplexity 20-80, burstiness > 0.4
# AI text typically: perplexity 5-15, burstiness < 0.3
THRESHOLDS = {
    "perplexity_low": 15.0,      # Below this = likely AI
    "perplexity_high": 60.0,     # Above this = likely human
    "burstiness_low": 0.25,      # Below this = likely AI
    "burstiness_high": 0.45,     # Above this = likely human
    "ttr_low": 0.4,              # Below this = repetitive (AI)
    "ttr_high": 0.7,             # Above this = diverse (human)
    "style_match_good": 0.7,     # Above this = matches author well
}


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class AIDetectionResult:
    """Comprehensive AI detection result."""
    # Overall scores (0 = human-like, 1 = AI-like)
    ai_score: float = 0.0
    confidence: float = 0.0

    # Individual metrics
    perplexity: float = 0.0
    perplexity_score: float = 0.0  # Normalized 0-1

    burstiness: float = 0.0
    burstiness_score: float = 0.0  # Normalized 0-1

    vocabulary_diversity: float = 0.0
    vocabulary_score: float = 0.0  # Normalized 0-1

    style_consistency: float = 0.0
    style_score: float = 0.0  # Normalized 0-1

    # Pattern-based detection (legacy, lower weight)
    pattern_score: float = 0.0
    pattern_issues: list = field(default_factory=list)

    # Analysis details
    sentence_lengths: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)

    @property
    def is_likely_ai(self) -> bool:
        """Whether text is likely AI-generated."""
        return self.ai_score > 0.5

    @property
    def risk_level(self) -> str:
        """Human-readable risk level."""
        if self.ai_score < 0.25:
            return "low"
        elif self.ai_score < 0.5:
            return "moderate"
        elif self.ai_score < 0.75:
            return "high"
        else:
            return "very_high"


# ==============================================================================
# Text Processing Utilities
# ==============================================================================

def tokenize(text: str) -> list[str]:
    """Simple word tokenization."""
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Handle common abbreviations
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|et al|i\.e|e\.g|vs|etc|Fig|Eq)\.\s', r'\1<DOT> ', text)
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences if len(s.strip()) > 3]
    return sentences


# ==============================================================================
# Core Metrics (Based on Research)
# ==============================================================================

def calculate_perplexity_approx(text: str) -> float:
    """Approximate perplexity using character-level n-gram model.

    True perplexity requires a trained LM. This approximation uses
    character-level entropy as a proxy, which correlates with AI detection.

    Lower perplexity = more predictable = more AI-like
    Human text: 20-80, AI text: 5-15
    """
    if len(text) < 50:
        return 30.0  # Default for short text

    # Character bigram entropy
    text_lower = text.lower()
    bigrams = [text_lower[i:i+2] for i in range(len(text_lower)-1)]

    if not bigrams:
        return 30.0

    # Count bigram frequencies
    bigram_counts = Counter(bigrams)
    total = len(bigrams)

    # Calculate entropy
    entropy = 0.0
    for count in bigram_counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)

    # Convert entropy to perplexity-like score
    # Higher entropy = higher perplexity = more human-like
    perplexity = 2 ** entropy

    # Scale to typical range (5-80)
    perplexity = perplexity * 3.5

    return round(perplexity, 2)


def calculate_burstiness(text: str) -> float:
    """Calculate burstiness (sentence length variation).

    Burstiness = coefficient of variation of sentence lengths
    Human writing has high burstiness (varied sentences)
    AI writing has low burstiness (uniform sentences)

    Human: > 0.4, AI: < 0.3
    """
    sentences = split_sentences(text)

    if len(sentences) < 3:
        return 0.35  # Default for short text

    # Get sentence lengths (word count)
    lengths = [len(tokenize(s)) for s in sentences]
    lengths = [l for l in lengths if l > 0]

    if len(lengths) < 3:
        return 0.35

    # Calculate coefficient of variation
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.35

    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    std_dev = variance ** 0.5
    cv = std_dev / mean_len

    return round(cv, 3)


def calculate_vocabulary_diversity(text: str) -> float:
    """Calculate vocabulary diversity (Type-Token Ratio).

    TTR = unique words / total words
    Higher = more diverse vocabulary = more human-like

    Human: 0.5-0.8, AI: 0.3-0.5 (AI tends to repeat phrases)
    """
    words = tokenize(text)

    if len(words) < 20:
        return 0.5  # Default for short text

    # Use moving TTR for longer texts (more stable)
    window_size = min(100, len(words))

    if len(words) <= window_size:
        ttr = len(set(words)) / len(words)
    else:
        # Average TTR over windows
        ttrs = []
        for i in range(0, len(words) - window_size + 1, window_size // 2):
            window = words[i:i + window_size]
            ttrs.append(len(set(window)) / len(window))
        ttr = sum(ttrs) / len(ttrs)

    return round(ttr, 3)


def calculate_style_consistency(text: str, fingerprint: Optional[dict]) -> float:
    """Calculate how well text matches author's style fingerprint.

    Higher = better match to author style

    Returns:
        Score from 0 (no match) to 1 (perfect match)
    """
    if not fingerprint:
        return 0.5  # Neutral if no fingerprint

    scores = []

    # 1. Sentence length match
    sentences = split_sentences(text)
    if sentences:
        lengths = [len(tokenize(s)) for s in sentences]
        avg_len = sum(lengths) / len(lengths) if lengths else 20

        target_len = fingerprint.get("sentence_stats", {}).get("avg_length", 20)
        target_std = fingerprint.get("sentence_stats", {}).get("std_dev", 8)

        # Score based on how close to target
        len_diff = abs(avg_len - target_len)
        len_score = max(0, 1 - len_diff / (target_std * 2))
        scores.append(len_score)

    # 2. Vocabulary overlap
    words = set(tokenize(text))
    fav_verbs = set(fingerprint.get("vocabulary_stats", {}).get("top_verbs", []))
    fav_trans = set(fingerprint.get("vocabulary_stats", {}).get("favorite_transitions", []))

    if fav_verbs:
        verb_overlap = len(words & fav_verbs) / len(fav_verbs)
        scores.append(min(1.0, verb_overlap * 2))  # Scale up

    if fav_trans:
        trans_overlap = len(words & fav_trans) / len(fav_trans)
        scores.append(min(1.0, trans_overlap * 2))

    # 3. Passive voice ratio match
    structure = fingerprint.get("structure_stats", {})
    if structure.get("passive_voice_ratio"):
        # Simple passive detection
        passive_patterns = len(re.findall(
            r'\b(was|were|been|being|is|are)\s+\w+ed\b',
            text, re.IGNORECASE
        ))
        total_sentences = len(sentences) if sentences else 1
        text_passive_ratio = passive_patterns / total_sentences

        target_passive = structure["passive_voice_ratio"]
        passive_diff = abs(text_passive_ratio - target_passive)
        passive_score = max(0, 1 - passive_diff * 2)
        scores.append(passive_score)

    if not scores:
        return 0.5

    return round(sum(scores) / len(scores), 3)


# ==============================================================================
# Pattern Detection (Supplementary)
# ==============================================================================

# Common AI patterns (lower weight in final score)
AI_PATTERNS = [
    # Overused transitions
    (r'\bFurthermore\b', 'Furthermore', 0.08),
    (r'\bMoreover\b', 'Moreover', 0.08),
    (r'\bAdditionally\b', 'Additionally', 0.06),
    (r'\bConsequently\b', 'Consequently', 0.05),

    # Filler phrases
    (r'It is worth noting that', 'It is worth noting', 0.10),
    (r'It is important to note', 'It is important to note', 0.10),
    (r'It should be noted', 'It should be noted', 0.08),

    # Fancy words AI loves
    (r'\butilize[sd]?\b', 'utilize', 0.05),
    (r'\bleverage[sd]?\b', 'leverage', 0.06),
    (r'\bfacilitate[sd]?\b', 'facilitate', 0.05),

    # Perfect enumeration
    (r'First(?:ly)?,.*Second(?:ly)?,.*Third(?:ly)?,', 'First/Second/Third', 0.08),

    # Verbose phrases
    (r'\bplays a (?:crucial|vital|pivotal) role\b', 'plays a crucial role', 0.06),
    (r'\ba wide (?:range|variety|array) of\b', 'a wide range of', 0.05),
]


def detect_patterns(text: str) -> tuple[float, list[dict]]:
    """Detect AI-typical patterns in text.

    Returns:
        Tuple of (pattern_score, list of issues)
    """
    issues = []
    total_score = 0.0

    for pattern, name, weight in AI_PATTERNS:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            count = len(matches)
            score = weight * min(count, 3)  # Cap at 3 occurrences
            total_score += score

            issues.append({
                "pattern": name,
                "count": count,
                "weight": weight,
                "score": round(score, 3),
            })

    return min(1.0, total_score), issues


# ==============================================================================
# Main Detector Class
# ==============================================================================

class ScientificAIDetector:
    """Research-based AI detection system.

    Uses multiple signals weighted by research effectiveness:
    - Perplexity (25%): Statistical predictability
    - Burstiness (30%): Sentence variation
    - Vocabulary (20%): Lexical diversity
    - Style (15%): Author fingerprint match
    - Patterns (10%): Known AI phrases (lowest weight)
    """

    WEIGHTS = {
        "perplexity": 0.25,
        "burstiness": 0.30,
        "vocabulary": 0.20,
        "style": 0.15,
        "patterns": 0.10,
    }

    def __init__(self):
        """Initialize detector."""
        self._fingerprint: Optional[dict] = None
        self._load_fingerprint()

    def _load_fingerprint(self) -> None:
        """Load author style fingerprint if available."""
        if STYLE_FINGERPRINT_PATH.exists():
            try:
                with open(STYLE_FINGERPRINT_PATH, "r", encoding="utf-8") as f:
                    self._fingerprint = json.load(f)
            except Exception:
                pass

    def reload_fingerprint(self) -> None:
        """Reload fingerprint from disk."""
        self._load_fingerprint()

    def detect(self, text: str) -> AIDetectionResult:
        """Perform comprehensive AI detection.

        Args:
            text: Text to analyze

        Returns:
            AIDetectionResult with all metrics
        """
        if len(text.strip()) < 50:
            return AIDetectionResult(
                ai_score=0.5,
                confidence=0.2,
                suggestions=["Text too short for reliable detection"]
            )

        # Calculate core metrics
        perplexity = calculate_perplexity_approx(text)
        burstiness = calculate_burstiness(text)
        vocabulary = calculate_vocabulary_diversity(text)
        style = calculate_style_consistency(text, self._fingerprint)
        pattern_score, pattern_issues = detect_patterns(text)

        # Normalize metrics to 0-1 (higher = more AI-like)
        perplexity_score = self._normalize_perplexity(perplexity)
        burstiness_score = self._normalize_burstiness(burstiness)
        vocabulary_score = self._normalize_vocabulary(vocabulary)
        style_score = 1 - style  # Invert: low style match = AI-like

        # Weighted combination
        ai_score = (
            self.WEIGHTS["perplexity"] * perplexity_score +
            self.WEIGHTS["burstiness"] * burstiness_score +
            self.WEIGHTS["vocabulary"] * vocabulary_score +
            self.WEIGHTS["style"] * style_score +
            self.WEIGHTS["patterns"] * pattern_score
        )

        # Calculate confidence based on text length and metric agreement
        confidence = self._calculate_confidence(
            text, perplexity_score, burstiness_score, vocabulary_score
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(
            perplexity_score, burstiness_score, vocabulary_score,
            style_score, pattern_issues
        )

        # Get sentence lengths for analysis
        sentences = split_sentences(text)
        sentence_lengths = [len(tokenize(s)) for s in sentences]

        return AIDetectionResult(
            ai_score=round(ai_score, 3),
            confidence=round(confidence, 2),
            perplexity=perplexity,
            perplexity_score=round(perplexity_score, 3),
            burstiness=burstiness,
            burstiness_score=round(burstiness_score, 3),
            vocabulary_diversity=vocabulary,
            vocabulary_score=round(vocabulary_score, 3),
            style_consistency=style,
            style_score=round(style_score, 3),
            pattern_score=round(pattern_score, 3),
            pattern_issues=pattern_issues,
            sentence_lengths=sentence_lengths,
            suggestions=suggestions,
        )

    def _normalize_perplexity(self, perplexity: float) -> float:
        """Normalize perplexity to 0-1 (higher = more AI-like)."""
        low = THRESHOLDS["perplexity_low"]
        high = THRESHOLDS["perplexity_high"]

        if perplexity <= low:
            return 1.0  # Very low perplexity = AI
        elif perplexity >= high:
            return 0.0  # High perplexity = human
        else:
            # Linear interpolation
            return 1 - (perplexity - low) / (high - low)

    def _normalize_burstiness(self, burstiness: float) -> float:
        """Normalize burstiness to 0-1 (higher = more AI-like)."""
        low = THRESHOLDS["burstiness_low"]
        high = THRESHOLDS["burstiness_high"]

        if burstiness <= low:
            return 1.0  # Low burstiness = AI
        elif burstiness >= high:
            return 0.0  # High burstiness = human
        else:
            return 1 - (burstiness - low) / (high - low)

    def _normalize_vocabulary(self, ttr: float) -> float:
        """Normalize vocabulary diversity to 0-1 (higher = more AI-like)."""
        low = THRESHOLDS["ttr_low"]
        high = THRESHOLDS["ttr_high"]

        if ttr <= low:
            return 1.0  # Low diversity = AI
        elif ttr >= high:
            return 0.0  # High diversity = human
        else:
            return 1 - (ttr - low) / (high - low)

    def _calculate_confidence(
        self,
        text: str,
        perplexity_score: float,
        burstiness_score: float,
        vocabulary_score: float
    ) -> float:
        """Calculate confidence in the detection result."""
        # Base confidence from text length
        word_count = len(tokenize(text))
        length_conf = min(1.0, word_count / 200)

        # Agreement between metrics
        scores = [perplexity_score, burstiness_score, vocabulary_score]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        agreement_conf = max(0.3, 1 - variance * 2)

        return (length_conf * 0.4 + agreement_conf * 0.6)

    def _generate_suggestions(
        self,
        perplexity_score: float,
        burstiness_score: float,
        vocabulary_score: float,
        style_score: float,
        pattern_issues: list
    ) -> list[str]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []

        # Perplexity suggestions
        if perplexity_score > 0.6:
            suggestions.append(
                "Text is highly predictable. Add unexpected word choices "
                "or unconventional phrasing."
            )

        # Burstiness suggestions (most important!)
        if burstiness_score > 0.6:
            suggestions.append(
                "CRITICAL: Sentence lengths too uniform. Mix short (5-10 words) "
                "and long (30-40 words) sentences."
            )
        elif burstiness_score > 0.4:
            suggestions.append(
                "Vary sentence length more. Add some punchy short sentences."
            )

        # Vocabulary suggestions
        if vocabulary_score > 0.6:
            suggestions.append(
                "Vocabulary too repetitive. Use more varied word choices."
            )

        # Style suggestions
        if style_score > 0.6:
            suggestions.append(
                "Text doesn't match author's writing style. Review the "
                "style fingerprint and adjust vocabulary/structure."
            )

        # Pattern-specific suggestions
        if pattern_issues:
            top_issues = sorted(pattern_issues, key=lambda x: -x["score"])[:3]
            for issue in top_issues:
                suggestions.append(f"Remove/replace: '{issue['pattern']}'")

        return suggestions[:7]


# ==============================================================================
# Humanizer - Automatic Text Improvement
# ==============================================================================

class TextHumanizer:
    """Automatically improve text to be more human-like.

    Based on research findings:
    1. Vary sentence length (most effective)
    2. Remove AI-typical phrases
    3. Use simpler vocabulary
    4. Break perfect parallel structures
    """

    # Phrase replacements
    REPLACEMENTS = [
        # Remove filler phrases entirely
        (r'It is worth noting that\s+', ''),
        (r'It is important to note that\s+', ''),
        (r'It should be noted that\s+', ''),
        (r'Notably,\s+', ''),
        (r'Importantly,\s+', ''),
        (r'Interestingly,\s+', ''),

        # Replace formal transitions
        (r'\bFurthermore,\s+', 'Also, '),
        (r'\bMoreover,\s+', 'Also, '),
        (r'\bAdditionally,\s+', 'Also, '),
        (r'\bConsequently,\s+', 'So '),
        (r'\bNevertheless,\s+', 'Still, '),
        (r'\bNonetheless,\s+', 'Yet '),
        (r'\bHence,\s+', 'So '),

        # Simplify vocabulary
        (r'\butilize\b', 'use'),
        (r'\butilized\b', 'used'),
        (r'\butilizes\b', 'uses'),
        (r'\butilizing\b', 'using'),
        (r'\bleverage\b', 'use'),
        (r'\bleveraged\b', 'used'),
        (r'\bfacilitate\b', 'help'),
        (r'\bfacilitated\b', 'helped'),
        (r'\bcommence\b', 'start'),
        (r'\bcommenced\b', 'started'),
        (r'\bprior to\b', 'before'),
        (r'\bsubsequently\b', 'then'),
        (r'\bin order to\b', 'to'),
        (r'\bdemonstrate\b', 'show'),
        (r'\bdemonstrated\b', 'showed'),
        (r'\bdemonstrates\b', 'shows'),

        # Simplify verbose phrases
        (r'\bplays a crucial role in\b', 'is key to'),
        (r'\bplays a vital role in\b', 'is vital for'),
        (r'\bplays a pivotal role in\b', 'is central to'),
        (r'\ba wide range of\b', 'many'),
        (r'\ba wide variety of\b', 'various'),
        (r'\bwith respect to\b', 'for'),
        (r'\bin terms of\b', 'for'),
        (r'\bin the context of\b', 'in'),
    ]

    def humanize(self, text: str, target_burstiness: float = 0.45) -> str:
        """Humanize text to reduce AI detection score.

        Args:
            text: Text to humanize
            target_burstiness: Target sentence length variation

        Returns:
            Humanized text
        """
        # Step 1: Apply phrase replacements
        result = text
        for pattern, replacement in self.REPLACEMENTS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Step 2: Fix capitalization after removals
        result = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), result)
        result = re.sub(r'^\s*([a-z])', lambda m: m.group(1).upper(), result)

        # Step 3: Vary sentence lengths if needed
        current_burstiness = calculate_burstiness(result)
        if current_burstiness < target_burstiness:
            result = self._vary_sentences(result)

        # Step 4: Clean up
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s+([.,;:])', r'\1', result)

        return result.strip()

    def _vary_sentences(self, text: str) -> str:
        """Add variation to sentence lengths."""
        sentences = split_sentences(text)

        if len(sentences) < 3:
            return text

        # Find sentences that could be split or combined
        modified = []
        i = 0

        while i < len(sentences):
            sent = sentences[i]
            words = tokenize(sent)
            word_count = len(words)

            # Very long sentence: try to split
            if word_count > 35 and '; ' in sent:
                parts = sent.split('; ')
                modified.extend(parts)
                i += 1
                continue

            # Medium sentence followed by short: might combine
            if (i + 1 < len(sentences) and
                15 < word_count < 25 and
                len(tokenize(sentences[i + 1])) < 12):
                # Sometimes combine with "and" or "—"
                if len(modified) % 3 == 0:  # Every third opportunity
                    combined = f"{sent.rstrip('.')} — {sentences[i + 1].lower()}"
                    modified.append(combined)
                    i += 2
                    continue

            modified.append(sent)
            i += 1

        return ' '.join(modified)


# ==============================================================================
# Singleton Instances
# ==============================================================================

_detector: Optional[ScientificAIDetector] = None
_humanizer: Optional[TextHumanizer] = None


def get_ai_detector() -> ScientificAIDetector:
    """Get or create AI detector singleton."""
    global _detector
    if _detector is None:
        _detector = ScientificAIDetector()
    return _detector


def get_humanizer() -> TextHumanizer:
    """Get or create humanizer singleton."""
    global _humanizer
    if _humanizer is None:
        _humanizer = TextHumanizer()
    return _humanizer


# ==============================================================================
# Convenience Functions
# ==============================================================================

def detect_ai_traces(text: str) -> AIDetectionResult:
    """Detect AI traces in text."""
    return get_ai_detector().detect(text)


def humanize_text(text: str) -> str:
    """Humanize text to reduce AI detection score."""
    return get_humanizer().humanize(text)


def get_ai_score(text: str) -> float:
    """Get AI score for text (0 = human, 1 = AI)."""
    return get_ai_detector().detect(text).ai_score


def correct_ai_traces(text: str) -> str:
    """Alias for humanize_text for backward compatibility."""
    return humanize_text(text)


# ==============================================================================
# CLI Interface
# ==============================================================================

def analyze_text_detailed(text: str) -> dict:
    """Get detailed analysis for display."""
    result = detect_ai_traces(text)

    return {
        "overall": {
            "ai_score": result.ai_score,
            "risk_level": result.risk_level,
            "confidence": result.confidence,
            "is_likely_ai": result.is_likely_ai,
        },
        "metrics": {
            "perplexity": {
                "value": result.perplexity,
                "score": result.perplexity_score,
                "interpretation": "lower = more predictable = more AI-like"
            },
            "burstiness": {
                "value": result.burstiness,
                "score": result.burstiness_score,
                "interpretation": "lower = more uniform = more AI-like"
            },
            "vocabulary_diversity": {
                "value": result.vocabulary_diversity,
                "score": result.vocabulary_score,
                "interpretation": "lower = more repetitive = more AI-like"
            },
            "style_consistency": {
                "value": result.style_consistency,
                "score": result.style_score,
                "interpretation": "higher = better match to author"
            },
        },
        "sentence_lengths": result.sentence_lengths,
        "pattern_issues": result.pattern_issues,
        "suggestions": result.suggestions,
    }
