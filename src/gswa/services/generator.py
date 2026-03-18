"""Unified Generator Facade.

Routes to local LLMClient or AnthropicClient based on config.
Provides a single interface for all LLM operations.
"""
import logging
from typing import Optional

from gswa.config import get_settings

logger = logging.getLogger(__name__)


class Generator:
    """Routes to the appropriate LLM backend based on config."""

    def __init__(self):
        """Initialize generator with the configured backend."""
        self.settings = get_settings()
        self._backend_type = self.settings.llm_backend

        if self._backend_type == "anthropic":
            from gswa.services.api_client import get_anthropic_client
            self._client = get_anthropic_client()
        elif self._backend_type in ("groq", "openrouter"):
            from gswa.services.external_client import get_external_client
            self._client = get_external_client()
        else:
            from gswa.services.llm_client import get_llm_client
            self._client = get_llm_client()

        logger.info(f"Generator initialized: backend={self._backend_type}")

    @property
    def is_api(self) -> bool:
        """Whether using an external API backend."""
        return self._backend_type in ("anthropic", "groq", "openrouter")

    async def check_health(self) -> dict:
        """Check backend health."""
        return await self._client.check_health()

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate completion via the configured backend.

        Args:
            messages: Chat messages [{"role": "...", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            model: Override model name (ignored for API backend)

        Returns:
            Generated text
        """
        # For API backend, ignore LoRA model overrides
        if self.is_api:
            model = None

        return await self._client.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            model=model,
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._client.model_name

    @property
    def backend(self) -> str:
        """Get the backend type."""
        return self._backend_type


# Singleton instance
_generator: Optional[Generator] = None


def get_generator() -> Generator:
    """Get or create Generator singleton."""
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


def reset_generator() -> None:
    """Reset the Generator singleton."""
    global _generator
    _generator = None
