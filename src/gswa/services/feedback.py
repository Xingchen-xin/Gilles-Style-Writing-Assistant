"""Feedback Collection Service.

Stores user feedback on generated variants for DPO training.
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

    def store_session(
        self,
        session_id: str,
        input_text: str,
        variants: list[dict],
        section: Optional[str] = None,
        model_version: str = ""
    ) -> None:
        """Store a rewrite session for later feedback.

        Args:
            session_id: Unique session identifier
            input_text: Original input text
            variants: List of generated variants
            section: Paper section type
            model_version: Model version used
        """
        self._session_cache[session_id] = {
            "input_text": input_text,
            "variants": variants,
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
            FeedbackType.EDITED: 0
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
            edited_count=counts[FeedbackType.EDITED]
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

        Args:
            record: Feedback record

        Returns:
            List of DPO training pairs
        """
        pairs = []
        input_text = record.get("input_text", "")
        variants = record.get("variants", [])

        # Find best and worst variants
        best_variants = [v for v in variants if v.get("feedback_type") == "best"]
        good_variants = [v for v in variants if v.get("feedback_type") == "good"]
        bad_variants = [v for v in variants if v.get("feedback_type") == "bad"]
        edited_variants = [v for v in variants if v.get("feedback_type") == "edited"]

        # Preferred outputs (in order of preference)
        preferred = edited_variants + best_variants + good_variants
        rejected = bad_variants

        # Create pairs: each preferred paired with each rejected
        for pref in preferred:
            pref_text = pref.get("edited_text") or pref.get("original_text", "")
            if not pref_text:
                continue

            for rej in rejected:
                rej_text = rej.get("original_text", "")
                if not rej_text:
                    continue

                pairs.append({
                    "prompt": input_text,
                    "chosen": pref_text,
                    "rejected": rej_text,
                    "section": record.get("section"),
                    "chosen_strategy": pref.get("strategy"),
                    "rejected_strategy": rej.get("strategy"),
                })

        return pairs


# Singleton instance
_feedback_service: Optional[FeedbackService] = None


def get_feedback_service() -> FeedbackService:
    """Get or create feedback service singleton."""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service
