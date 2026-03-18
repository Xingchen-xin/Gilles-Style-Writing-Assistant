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
    # NOTE: Do NOT remove Gilles's preferred transitions (However, Thus,
    # Importantly, Interestingly, Indeed) - these are part of his style.
    REPLACEMENTS = [
        # Remove filler phrases entirely
        (r'It is worth noting that\s+', ''),
        (r'It is important to note that\s+', ''),
        (r'It should be noted that\s+', ''),

        # Replace AI-typical formal transitions (NOT Gilles's preferred ones)
        (r'\bFurthermore,\s+', 'Also, '),
        (r'\bMoreover,\s+', 'Also, '),
        (r'\bAdditionally,\s+', 'Also, '),
        (r'\bConsequently,\s+', 'Thus, '),
        (r'\bHence,\s+', 'Thus, '),

        # Simplify AI-typical vocabulary
        (r'\butilize\b', 'use'),
        (r'\butilized\b', 'used'),
        (r'\butilizes\b', 'uses'),
        (r'\butilizing\b', 'using'),
        (r'\bleverage\b', 'use'),
        (r'\bleveraged\b', 'used'),
        (r'\bfacilitate\b', 'allow'),
        (r'\bfacilitated\b', 'allowed'),
        (r'\bfacilitates\b', 'allows'),
        (r'\bcommence\b', 'start'),
        (r'\bcommenced\b', 'started'),
        (r'\bprior to\b', 'before'),
        (r'\bin order to\b', 'to'),
        (r'\belucidate\b', 'clarify'),
        (r'\belucidated\b', 'clarified'),
        (r'\bunderscore[sd]?\b', 'highlight'),
        (r'\bunderscoring\b', 'highlighting'),
        (r'\bunveil[sed]*\b', 'reveal'),
        (r'\bunveiling\b', 'revealing'),

        # Simplify verbose phrases
        (r'\bplays a crucial role in\b', 'is key to'),
        (r'\bplays a vital role in\b', 'is important for'),
        (r'\bplays a pivotal role in\b', 'is central to'),
        (r'\ba wide range of\b', 'many'),
        (r'\ba wide variety of\b', 'various'),
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
        # Step 0: Strip markdown formatting from LLM output
        result = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        result = re.sub(r'\*(.+?)\*', r'\1', result)     # Italic

        # Step 1: Apply phrase replacements
        for pattern, replacement in self.REPLACEMENTS:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Step 2: Fix capitalization after removals
        result = re.sub(r'\.\s+([a-z])', lambda m: '. ' + m.group(1).upper(), result)
        result = re.sub(r'^\s*([a-z])', lambda m: m.group(1).upper(), result)

        # Step 3: Vary sentence lengths if needed
        current_burstiness = calculate_burstiness(result)
        if current_burstiness < target_burstiness:
            result = self._vary_sentences(result)

        # Step 4: Fix fragments AFTER vary_sentences (which may create new ones)
        result = self._fix_fragments(result)

        # Step 5: Clean up
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s+([.,;:])', r'\1', result)

        return result.strip()

    def _fix_fragments(self, text: str) -> str:
        """Detect and merge sentence fragments back into adjacent sentences.

        Fragments are incomplete sentences that lack a subject+verb. Common
        patterns from LLMs trying to write short sentences:
        - "Many of which remain silent." (relative clause)
        - "Notably the bld genes." (adverb + noun)
        - "Given the potential." (participle phrase)
        """
        sentences = split_sentences(text)
        if len(sentences) < 2:
            return text

        # Pattern: sentence starts with a fragment indicator
        fragment_starts = re.compile(
            r'^(?:'
            r'(?:many|some|most|all|none|each|several) of (?:which|whom|these|those)|'  # relative
            r'(?:notably|particularly|especially|given|including) [a-z]|'  # participle/adverb
            r'(?:such as|rather than|as well as|along with) |'  # comparative
            r'(?:transitioning|resulting|leading|followed by) |'  # dangling participle
            r'(?:achievable|observable|detectable) '  # adjective fragment
            r')',
            re.IGNORECASE
        )
        # Verbs that need a subject (fragments when starting a sentence)
        subjectless_verb = re.compile(
            r'^(?:undergo|exhibit|transition|reveal|encode|produce|suggest|indicate|demonstrate|show|require|control|remain|is|are|was|were|has|have|had)\b',
            re.IGNORECASE
        )
        # Common finite verbs for detecting if a sentence has a verb
        has_verb = re.compile(
            r'\b(?:is|are|was|were|has|have|had|do|does|did|can|will|may|shall|could|would|might|should|'
            r'\w+ed|'  # past tense
            r'\w+es|'  # third person
            r'\w+ates?|'  # -ate verbs
            r'\w+izes?'  # -ize verbs
            r')\b'
        )

        merged: list[str] = []
        skip_next = False
        for i, sent in enumerate(sentences):
            if skip_next:
                skip_next = False
                continue

            words = sent.split()
            word_count = len(words)

            # Check if this sentence is a fragment
            is_fragment = False
            # Relative clauses, dangling participles, etc.
            if word_count <= 12 and fragment_starts.match(sent):
                is_fragment = True
            # Verb without subject (< 10 words)
            elif word_count <= 10 and subjectless_verb.match(sent):
                is_fragment = True
            # Short sentence without any verb → likely a fragment
            # e.g. "Streptomyces, renowned for their production."
            elif word_count <= 8 and not has_verb.search(sent):
                is_fragment = True
            # Very short noun phrase with appositive: "Noun, appositive."
            elif word_count <= 4 and ',' in sent:
                is_fragment = True

            if is_fragment and merged:
                # Merge with previous sentence (restore as relative clause)
                prev = merged[-1].rstrip('.')
                merged[-1] = f"{prev}, {sent[0].lower()}{sent[1:]}."
            elif is_fragment and i + 1 < len(sentences):
                # Merge with next sentence
                next_sent = sentences[i + 1]
                merged.append(f"{sent}, {next_sent[0].lower()}{next_sent[1:]}.")
                skip_next = True
            else:
                # Ensure sentence ends with period
                sent_clean = sent.rstrip('.')
                merged.append(f"{sent_clean}.")

        return ' '.join(merged)

    def _vary_sentences(self, text: str) -> str:
        """Add variation to sentence lengths to increase CV.

        Strategy (split-only, never combine):
        1. Split sentences >25 words at natural break points
        2. For runs of 3+ similar-length sentences, split the longest one
        3. Never combine sentences — that creates longer uniform ones
        Goal: CV > 0.4.
        """
        sentences = split_sentences(text)

        if len(sentences) < 3:
            return text

        # Phase 1: Split long sentences at natural break points
        split_result = []
        for sent in sentences:
            parts = self._try_split_sentence(sent)
            split_result.extend(parts)

        # Phase 2: Break up runs of similar-length sentences
        # If 3+ consecutive sentences have similar word counts (within ±5),
        # try to split the longest one in the run
        modified = list(split_result)
        changed = True
        max_passes = 3
        while changed and max_passes > 0:
            changed = False
            max_passes -= 1
            lengths = [len(s.split()) for s in modified]
            i = 0
            new_modified = []
            while i < len(modified):
                # Check for a run of 3+ similar-length sentences
                run_end = i + 1
                while run_end < len(modified):
                    if abs(lengths[run_end] - lengths[i]) <= 5:
                        run_end += 1
                    else:
                        break
                run_len = run_end - i
                if run_len >= 3:
                    # Try to split ANY sentence in the run (longest first)
                    run_sents = modified[i:run_end]
                    # Sort indices by sentence length (longest first)
                    sorted_indices = sorted(
                        range(len(run_sents)),
                        key=lambda j: len(run_sents[j].split()),
                        reverse=True
                    )
                    split_done = False
                    for try_idx in sorted_indices:
                        sent = run_sents[try_idx]
                        if len(sent.split()) < 8:
                            continue
                        # Try normal split first, then comma split
                        parts = self._try_split_sentence(sent, threshold=12)
                        if len(parts) == 1:
                            parts = self._split_at_comma(sent)
                        if len(parts) > 1:
                            for j, s in enumerate(run_sents):
                                if j == try_idx:
                                    new_modified.extend(parts)
                                else:
                                    new_modified.append(s)
                            i = run_end
                            changed = True
                            split_done = True
                            break
                    if split_done:
                        continue
                # No run or couldn't split — pass through
                new_modified.append(modified[i])
                i += 1
            modified = new_modified

        # Ensure proper sentence endings
        result = []
        for s in modified:
            s = s.strip()
            if s and not s.endswith(('.', '!', '?')):
                s += '.'
            result.append(s)

        return ' '.join(result)

    def _try_split_sentence(self, sent: str, threshold: int = 25) -> list[str]:
        """Try to split a sentence at a natural break point.

        Args:
            sent: The sentence to split
            threshold: Word count above which to attempt splitting

        Returns:
            List of 1 or more sentence fragments
        """
        words = sent.split()
        if len(words) <= threshold:
            return [sent]

        # Try semicolon first (cleanest split)
        if '; ' in sent:
            parts = sent.split('; ', 1)
            first = parts[0].rstrip('.') + '.'
            second = parts[1].strip()
            if second and second[0].islower():
                second = second[0].upper() + second[1:]
            return [first, second]

        # Try clause-level splitters
        splitters = [
            (', which ', 'This '),
            (', whereas ', 'In contrast, '),
            (', while ', 'Meanwhile, '),
            (', and this ', 'This '),
            (', suggesting ', 'This suggests '),
            (', indicating ', 'This indicates '),
            (', resulting in ', 'This resulted in '),
        ]
        for marker, replacement in splitters:
            if marker in sent:
                idx = sent.index(marker)
                # Only split if both halves are substantial (>5 words each)
                first_words = len(sent[:idx].split())
                rest_text = sent[idx + len(marker):]
                rest_words = len(rest_text.split())
                if first_words >= 5 and rest_words >= 4:
                    first_part = sent[:idx] + '.'
                    rest = replacement + rest_text
                    if rest and rest[0].islower():
                        rest = rest[0].upper() + rest[1:]
                    return [first_part, rest]

        # Try splitting at ", and " for longer sentences
        if len(words) > 20 and ', and ' in sent:
            idx = sent.index(', and ')
            first_words = len(sent[:idx].split())
            rest_text = sent[idx + 6:]  # skip ', and '
            rest_words = len(rest_text.split())
            if first_words >= 6 and rest_words >= 6:
                first_part = sent[:idx] + '.'
                rest = rest_text.strip()
                if rest and rest[0].islower():
                    rest = rest[0].upper() + rest[1:]
                return [first_part, rest]

        return [sent]

    def _split_at_comma(self, sent: str) -> list[str]:
        """Last-resort split at comma for uniform-run breaking.

        Looks for introductory phrases (3-7 words before comma) where the
        rest forms a complete clause. Restructures into two sentences.

        Returns [sent] unchanged if no good split found.
        """
        # Find commas
        comma_positions = [m.start() for m in re.finditer(r',\s', sent)]
        if not comma_positions:
            return [sent]

        for pos in comma_positions:
            before = sent[:pos].strip()
            after = sent[pos + 2:].strip()  # skip ", "
            before_words = len(before.split())
            after_words = len(after.split())

            # Good split: short intro (3-7 words) + substantial rest (6+ words)
            # Rest must NOT start with a dependent-clause word or preposition
            dependent_starts = r'^(?:which|that|because|although|though|unless|if|when|where|while|as|since|after|before|during|through|via|by|with|without|despite|for|from|into|onto|upon|including|according|between|among|having|being)\b'
            # Before must NOT start with a preposition (would create a fragment)
            prep_starts = r'^(?:in|on|at|under|during|after|before|with|by|for|from|through|between|among|upon|over|into|across|along|around|behind|below|beneath|beside|beyond|near|toward|against|within|without|despite|via)\b'
            if (3 <= before_words <= 7 and after_words >= 6 and
                    not re.match(dependent_starts, after, re.IGNORECASE) and
                    not re.match(prep_starts, before, re.IGNORECASE)):
                # Keep intro as short sentence + main clause as longer sentence
                intro_sent = before.rstrip('.') + '.'
                main_clause = after
                if main_clause[0].islower():
                    main_clause = main_clause[0].upper() + main_clause[1:]
                main_sent = main_clause.rstrip('.') + '.'
                return [intro_sent, main_sent]

        # Alternative: split at comma between two balanced halves
        for pos in comma_positions:
            before = sent[:pos].strip()
            after = sent[pos + 2:].strip()
            before_words = len(before.split())
            after_words = len(after.split())

            # Balanced split where after starts with a conjunction or clause
            if (before_words >= 5 and after_words >= 5 and
                    re.match(r'(?:and |but |or |yet |so |which |where |when )', after)):
                first_part = before + '.'
                rest = after
                # Strip leading conjunction for independent sentence
                rest = re.sub(r'^(?:and |but |or |yet |so )', '', rest)
                if rest and rest[0].islower():
                    rest = rest[0].upper() + rest[1:]
                rest = rest.rstrip('.') + '.'
                return [first_part, rest]

        return [sent]


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
