"""GSWA Configuration Module.

This module handles all configuration with security-first defaults.
External API calls are DISABLED by default and cannot be enabled
without explicit override.

Supports multiple LLM backends:
- vllm: For Linux/NVIDIA GPU servers
- ollama: For Mac (Apple Silicon) and Linux
- lm-studio: For desktop users with LM Studio
"""
from functools import lru_cache
from typing import Literal
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

    # === Authentication (enabled by default for security) ===
    # Override with GSWA_AUTH_USER and GSWA_AUTH_PASS environment variables
    auth_enabled: bool = True  # Authentication enabled by default
    auth_user: str = "gilles"  # Default username
    auth_pass: str = "IBLGilles2026"  # Default password

    # === LLM Backend ===
    # Supported: "vllm" (Linux/NVIDIA), "ollama" (Mac/Linux), "lm-studio" (Desktop)
    llm_backend: Literal["vllm", "ollama", "lm-studio"] = "vllm"

    # === LLM Server ===
    # Default URLs by backend:
    # - vllm: http://localhost:8000/v1
    # - ollama: http://localhost:11434/v1
    # - lm-studio: http://localhost:1234/v1
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "mistralai/Mistral-Nemo-Instruct-2407"  # Match start_vllm.sh default
    vllm_api_key: str = "dummy"  # Local servers don't need real keys

    @property
    def llm_base_url(self) -> str:
        """Get the LLM base URL based on backend type."""
        # If explicitly set, use that
        if self.vllm_base_url != "http://localhost:8000/v1":
            return self.vllm_base_url
        # Otherwise use backend defaults
        defaults = {
            "vllm": "http://localhost:8000/v1",
            "ollama": "http://localhost:11434/v1",
            "lm-studio": "http://localhost:1234/v1",
        }
        return defaults.get(self.llm_backend, self.vllm_base_url)

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
    models_dir: str = "./models"

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
