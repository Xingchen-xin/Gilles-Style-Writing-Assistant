"""Logging utilities for GSWA.

Provides audit logging for rewrite operations.
"""
import logging
import hashlib
from datetime import datetime
from typing import Optional
from pathlib import Path

from gswa.config import get_settings


def get_audit_logger() -> logging.Logger:
    """Get or create audit logger.

    Returns:
        Logger configured for audit logging
    """
    logger = logging.getLogger("gswa.audit")

    if not logger.handlers:
        settings = get_settings()
        log_path = Path(settings.log_path)
        log_path.mkdir(parents=True, exist_ok=True)

        # File handler for audit logs
        handler = logging.FileHandler(log_path / "audit.log")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def hash_text(text: str) -> str:
    """Create SHA256 hash of text for logging.

    We log hashes instead of full text to protect confidential corpus.

    Args:
        text: Text to hash

    Returns:
        First 16 characters of SHA256 hash
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def log_rewrite_request(
    input_text: str,
    n_variants: int,
    section: Optional[str] = None,
) -> None:
    """Log a rewrite request (without sensitive content).

    Args:
        input_text: Input text (will be hashed)
        n_variants: Number of variants requested
        section: Paper section
    """
    logger = get_audit_logger()
    logger.info(
        f"REWRITE_REQUEST | "
        f"input_hash={hash_text(input_text)} | "
        f"input_len={len(input_text)} | "
        f"n_variants={n_variants} | "
        f"section={section}"
    )


def log_rewrite_result(
    input_hash: str,
    variant_hashes: list[str],
    fallback_count: int,
    processing_time_ms: int,
) -> None:
    """Log a rewrite result.

    Args:
        input_hash: Hash of input text
        variant_hashes: Hashes of output variants
        fallback_count: Number of fallbacks triggered
        processing_time_ms: Processing time in milliseconds
    """
    logger = get_audit_logger()
    logger.info(
        f"REWRITE_RESULT | "
        f"input_hash={input_hash} | "
        f"n_variants={len(variant_hashes)} | "
        f"fallback_count={fallback_count} | "
        f"processing_time_ms={processing_time_ms}"
    )


def log_similarity_check(
    text_hash: str,
    ngram_max_match: int,
    embed_top1: float,
    triggered_fallback: bool,
) -> None:
    """Log a similarity check result.

    Args:
        text_hash: Hash of checked text
        ngram_max_match: Maximum n-gram match length
        embed_top1: Top embedding similarity score
        triggered_fallback: Whether fallback was triggered
    """
    logger = get_audit_logger()
    logger.info(
        f"SIMILARITY_CHECK | "
        f"text_hash={text_hash} | "
        f"ngram_max_match={ngram_max_match} | "
        f"embed_top1={embed_top1:.4f} | "
        f"triggered_fallback={triggered_fallback}"
    )
