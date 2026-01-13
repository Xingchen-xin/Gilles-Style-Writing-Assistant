"""GSWA Configuration Module.

This module handles all configuration with security-first defaults.
External API calls are DISABLED by default and cannot be enabled
without explicit override.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with security defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === SECURITY (HARDCODED DEFAULTS) ===
    allow_external_api: bool = False  # MUST remain False

    # === LLM Server ===
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "mistral-7b-instruct"
    vllm_api_key: str = "dummy"  # vLLM doesn't need real key locally

    # === Similarity Thresholds ===
    threshold_ngram_max_match: int = 12
    threshold_ngram_overlap: float = 0.15
    threshold_embed_top1: float = 0.88

    # === Generation ===
    default_n_variants: int = 3
    max_n_variants: int = 5
    temperature_base: float = 0.3
    temperature_variance: float = 0.15
    max_new_tokens: int = 1024

    # === Paths ===
    corpus_path: str = "./data/corpus/parsed"
    index_path: str = "./data/index"
    log_path: str = "./logs"

    # === Embedding ===
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    def validate_security(self) -> None:
        """Validate security constraints at startup."""
        if self.allow_external_api:
            raise ValueError(
                "CRITICAL: ALLOW_EXTERNAL_API=true is FORBIDDEN. "
                "This system MUST run fully offline. "
                "Remove or set ALLOW_EXTERNAL_API=false."
            )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.validate_security()
    return settings
