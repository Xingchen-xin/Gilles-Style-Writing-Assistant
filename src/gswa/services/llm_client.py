"""LLM Client for vLLM server.

Uses OpenAI-compatible API to communicate with local vLLM server.
"""
import httpx
import logging
from typing import Optional
from urllib.parse import urlparse

from gswa.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for vLLM OpenAI-compatible API."""

    # Allowed hosts for local-only operation
    ALLOWED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}

    def __init__(self):
        """Initialize LLM client with security validation."""
        self.settings = get_settings()
        self._base_url = self.settings.vllm_base_url.rstrip("/")
        self._model = self.settings.vllm_model_name

        # Security: Ensure we're connecting locally
        self._validate_local_only()

    def _validate_local_only(self) -> None:
        """Ensure we only connect to local server.

        Raises:
            ValueError: If attempting to connect to non-local host
        """
        parsed = urlparse(self._base_url)

        if parsed.hostname not in self.ALLOWED_HOSTS:
            if not self.settings.allow_external_api:
                raise ValueError(
                    f"External API calls forbidden. "
                    f"Host '{parsed.hostname}' is not localhost. "
                    f"Only local vLLM server is allowed."
                )

    async def check_health(self) -> dict:
        """Check if vLLM server is healthy.

        Returns:
            Dictionary with status and model info
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                resp = await client.get(f"{self._base_url}/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", [])
                    return {
                        "status": "connected",
                        "models": [m.get("id") for m in models]
                    }
            except httpx.ConnectError:
                return {"status": "disconnected", "error": "Cannot connect to vLLM server"}
            except Exception as e:
                return {"status": "error", "error": str(e)}
        return {"status": "disconnected"}

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate completion from vLLM.

        Args:
            messages: Chat messages [{"role": "...", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            Generated text

        Raises:
            httpx.HTTPError: On API errors
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.settings.vllm_api_key}"}
            )
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"]

    async def generate_variants(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int = 3,
        temperature_base: float = 0.3,
        temperature_variance: float = 0.15,
    ) -> list[str]:
        """Generate multiple variants with slightly different temperatures.

        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt with the text to rewrite
            n: Number of variants to generate
            temperature_base: Base temperature
            temperature_variance: Variance to add/subtract per variant

        Returns:
            List of generated texts
        """
        variants = []

        for i in range(n):
            # Vary temperature slightly for each variant
            temp = temperature_base + (i - n // 2) * temperature_variance
            temp = max(0.1, min(1.0, temp))  # Clamp to valid range

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            result = await self.complete(
                messages=messages,
                temperature=temp,
                max_tokens=self.settings.max_new_tokens,
            )
            variants.append(result)

        return variants

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
