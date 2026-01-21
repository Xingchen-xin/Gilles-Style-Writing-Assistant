"""Feedback Collection Service.

Stores user feedback on generated variants for DPO training.
Includes automatic AI trace detection for enhanced rejection sampling.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

from gswa.api.schemas import (
    FeedbackRequest, FeedbackResponse, FeedbackStats, FeedbackType
)
from gswa.config import get_settings
from gswa.utils.ai_detector import get_ai_detector, AIDetectionResult

logger = logging.getLogger(__name__)


class FeedbackService:
    """Manages feedback collection and storage."""

    def __init__(self, feedback_dir: Optional[str] = None):
        """Initialize feedback service.

        Args:
            feedback_dir: Directory to store feedback data
        """
        settings = get_settings()
        self.feedback_dir = Path(feedback_dir or settings.log_path) / "feedback"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        # Store generated variants for pairing with feedback
        self._session_cache: dict[str, dict] = {}

        # AI detector for automatic trace detection
        self._ai_detector = get_ai_detector()

    def analyze_ai_traces(self, text: str) -> dict:
        """Analyze text for AI traces.

        Args:
            text: Text to analyze

        Returns:
            Dict with AI score and detection details
        """
        result = self._ai_detector.detect(text)
        return {
            "ai_score": result.ai_score,
            "has_ai_traces": result.is_likely_ai,
            "issue_count": len(result.pattern_issues),
            "top_issues": [
                {"type": i["pattern"], "found": i.get("pattern", "")}
                for i in result.pattern_issues[:3]
            ],
        }

    def store_session(
        self,
        session_id: str,
        input_text: str,
        variants: list[dict],
        section: Optional[str] = None,
        model_version: str = ""
    ) -> None:
        """Store a rewrite session for later feedback.

        Automatically analyzes variants for AI traces and stores scores.

        Args:
            session_id: Unique session identifier
            input_text: Original input text
            variants: List of generated variants
            section: Paper section type
            model_version: Model version used
        """
        # Analyze each variant for AI traces
        variants_with_ai_scores = []
        for v in variants:
            variant_copy = dict(v)
            text = v.get("text", "")
            if text:
                ai_analysis = self.analyze_ai_traces(text)
                variant_copy["ai_analysis"] = ai_analysis
            variants_with_ai_scores.append(variant_copy)

        self._session_cache[session_id] = {
            "input_text": input_text,
            "variants": variants_with_ai_scores,
            "section": section,
            "model_version": model_version,
            "timestamp": datetime.utcnow().isoformat()
        }

    def submit_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """Submit feedback for a session.

        Args:
            request: Feedback request with ratings

        Returns:
            Feedback response with status
        """
        feedback_id = str(uuid.uuid4())[:8]

        # Get session data if available
        session_data = self._session_cache.get(request.session_id, {})

        # Build feedback record
        record = {
            "feedback_id": feedback_id,
            "session_id": request.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "input_text": request.input_text,
            "section": request.section.value if request.section else None,
            "variants": [],
            "user_notes": request.user_notes,
            "model_version": session_data.get("model_version", "unknown")
        }

        # Add variant feedback
        for vf in request.variants:
            variant_data = {
                "index": vf.variant_index,
                "feedback_type": vf.feedback_type.value,
                "edited_text": vf.edited_text,
            }

            # Include original variant text if available
            if session_data and vf.variant_index < len(session_data.get("variants", [])):
                original = session_data["variants"][vf.variant_index]
                variant_data["original_text"] = original.get("text", "")
                variant_data["strategy"] = original.get("strategy", "")
                variant_data["scores"] = original.get("scores", {})
                # Include AI trace analysis
                variant_data["ai_analysis"] = original.get("ai_analysis", {})

            record["variants"].append(variant_data)

        # Save to file
        feedback_file = self.feedback_dir / f"feedback_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved feedback {feedback_id} for session {request.session_id}")

        # Clean up session cache
        if request.session_id in self._session_cache:
            del self._session_cache[request.session_id]

        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Feedback saved successfully"
        )

    def get_stats(self) -> FeedbackStats:
        """Get feedback statistics.

        Returns:
            Statistics about collected feedback
        """
        total_sessions = 0
        total_variants = 0
        counts = {
            FeedbackType.BEST: 0,
            FeedbackType.GOOD: 0,
            FeedbackType.BAD: 0,
            FeedbackType.EDITED: 0,
            FeedbackType.AI_LIKE: 0,
        }

        for feedback_file in self.feedback_dir.glob("feedback_*.jsonl"):
            with open(feedback_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        total_sessions += 1
                        for variant in record.get("variants", []):
                            total_variants += 1
                            ft = variant.get("feedback_type")
                            if ft in [e.value for e in FeedbackType]:
                                counts[FeedbackType(ft)] += 1
                    except json.JSONDecodeError:
                        continue

        return FeedbackStats(
            total_sessions=total_sessions,
            total_variants_rated=total_variants,
            best_count=counts[FeedbackType.BEST],
            good_count=counts[FeedbackType.GOOD],
            bad_count=counts[FeedbackType.BAD],
            edited_count=counts[FeedbackType.EDITED],
            ai_like_count=counts[FeedbackType.AI_LIKE],
        )

    def export_for_dpo(self, output_path: str) -> int:
        """Export feedback data in DPO training format.

        DPO format requires pairs of (prompt, chosen, rejected).

        Args:
            output_path: Path to save DPO training data

        Returns:
            Number of training pairs exported
        """
        pairs = []

        for feedback_file in self.feedback_dir.glob("feedback_*.jsonl"):
            with open(feedback_file, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        pairs.extend(self._extract_dpo_pairs(record))
                    except json.JSONDecodeError:
                        continue

        # Save to output file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

        logger.info(f"Exported {len(pairs)} DPO training pairs to {output_path}")
        return len(pairs)

    def _extract_dpo_pairs(self, record: dict) -> list[dict]:
        """Extract DPO training pairs from a feedback record.

        Uses both explicit feedback and automatic AI trace detection.

        Args:
            record: Feedback record

        Returns:
            List of DPO training pairs
        """
        pairs = []
        input_text = record.get("input_text", "")
        variants = record.get("variants", [])

        # Find best and worst variants based on explicit feedback
        best_variants = [v for v in variants if v.get("feedback_type") == "best"]
        good_variants = [v for v in variants if v.get("feedback_type") == "good"]
        bad_variants = [v for v in variants if v.get("feedback_type") == "bad"]
        edited_variants = [v for v in variants if v.get("feedback_type") == "edited"]
        ai_like_variants = [v for v in variants if v.get("feedback_type") == "ai_like"]

        # Also use automatic AI detection for rejection candidates
        # Variants with high AI scores (>0.5) that weren't marked as best/good
        auto_rejected = []
        for v in variants:
            ai_analysis = v.get("ai_analysis", {})
            ai_score = ai_analysis.get("ai_score", 0)
            feedback_type = v.get("feedback_type", "")

            # High AI score + not marked as good = potential rejection
            if ai_score > 0.5 and feedback_type not in ["best", "good", "edited"]:
                auto_rejected.append(v)

        # Preferred outputs (in order of preference):
        # 1. Edited by user (highest value)
        # 2. Best (user's choice)
        # 3. Good with low AI score
        preferred = []
        preferred.extend(edited_variants)
        preferred.extend(best_variants)

        # Only add good variants with low AI scores
        for v in good_variants:
            ai_score = v.get("ai_analysis", {}).get("ai_score", 0)
            if ai_score < 0.4:
                preferred.append(v)

        # Rejected outputs:
        # 1. Explicitly marked as bad
        # 2. Explicitly marked as ai_like
        # 3. Auto-detected high AI score variants
        rejected = bad_variants + ai_like_variants + auto_rejected

        # Remove duplicates (by text)
        seen_rejected = set()
        unique_rejected = []
        for v in rejected:
            text = v.get("original_text", "")
            if text and text not in seen_rejected:
                seen_rejected.add(text)
                unique_rejected.append(v)
        rejected = unique_rejected

        # Create pairs: each preferred paired with each rejected
        for pref in preferred:
            pref_text = pref.get("edited_text") or pref.get("original_text", "")
            if not pref_text:
                continue

            for rej in rejected:
                rej_text = rej.get("original_text", "")
                if not rej_text or rej_text == pref_text:
                    continue

                # Get AI scores for metadata
                pref_ai_score = pref.get("ai_analysis", {}).get("ai_score", 0)
                rej_ai_score = rej.get("ai_analysis", {}).get("ai_score", 0)

                pairs.append({
                    "prompt": input_text,
                    "chosen": pref_text,
                    "rejected": rej_text,
                    "section": record.get("section"),
                    "chosen_strategy": pref.get("strategy"),
                    "rejected_strategy": rej.get("strategy"),
                    "chosen_ai_score": pref_ai_score,
                    "rejected_ai_score": rej_ai_score,
                    "rejection_reason": self._get_rejection_reason(rej),
                })

        return pairs

    def _get_rejection_reason(self, variant: dict) -> str:
        """Get human-readable rejection reason for a variant.

        Args:
            variant: Variant dict

        Returns:
            Rejection reason string
        """
        reasons = []

        feedback_type = variant.get("feedback_type", "")
        if feedback_type == "bad":
            reasons.append("user_marked_bad")
        elif feedback_type == "ai_like":
            reasons.append("user_marked_ai_like")

        ai_analysis = variant.get("ai_analysis", {})
        if ai_analysis.get("ai_score", 0) > 0.5:
            reasons.append(f"high_ai_score({ai_analysis['ai_score']:.2f})")

        top_issues = ai_analysis.get("top_issues", [])
        if top_issues:
            issue_types = [i.get("type", "") for i in top_issues[:2]]
            reasons.append(f"issues:{','.join(issue_types)}")

        return "; ".join(reasons) if reasons else "unspecified"


# Singleton instance
_feedback_service: Optional[FeedbackService] = None


def get_feedback_service() -> FeedbackService:
    """Get or create feedback service singleton."""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service
