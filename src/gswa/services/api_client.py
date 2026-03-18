"""Anthropic Claude API Client.

Uses the Anthropic Python SDK for external API calls.
Only active when LLM_BACKEND=anthropic and ALLOW_EXTERNAL_API=true.
"""
import logging
from typing import Optional

import anthropic

from gswa.config import get_settings

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Client for Anthropic Claude API.

    Same interface as LLMClient for drop-in compatibility.
    """

    def __init__(self):
        """Initialize Anthropic client."""
        self.settings = get_settings()
        self._client = anthropic.AsyncAnthropic(api_key=self.settings.api_key)
        self._model = self.settings.api_model
        logger.info(f"Anthropic client initialized: model={self._model}")

    async def check_health(self) -> dict:
        """Check if Anthropic API is accessible."""
        try:
            # Small test message to verify API key works
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return {
                "status": "connected",
                "backend": "anthropic",
                "models": [self._model],
            }
        except anthropic.AuthenticationError:
            return {
                "status": "error",
                "backend": "anthropic",
                "error": "Invalid API key",
            }
        except Exception as e:
            return {
                "status": "error",
                "backend": "anthropic",
                "error": str(e),
            }

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate completion from Anthropic API.

        Extracts system prompt from messages (Claude API takes system as top-level param).

        Args:
            messages: Chat messages [{"role": "...", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            model: Override model name

        Returns:
            Generated text
        """
        # Extract system prompt from messages
        system_text = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                user_messages.append(msg)

        # Ensure at least one user message
        if not user_messages:
            user_messages = [{"role": "user", "content": ""}]

        kwargs = {
            "model": model or self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_text:
            kwargs["system"] = system_text
        if stop:
            kwargs["stop_sequences"] = stop

        resp = await self._client.messages.create(**kwargs)
        return resp.content[0].text

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model

    @property
    def backend(self) -> str:
        """Get the backend type."""
        return "anthropic"


# Singleton instance
_anthropic_client: Optional[AnthropicClient] = None


def get_anthropic_client() -> AnthropicClient:
    """Get or create Anthropic client singleton."""
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AnthropicClient()
    return _anthropic_client


def reset_anthropic_client() -> None:
    """Reset the Anthropic client singleton."""
    global _anthropic_client
    _anthropic_client = None
