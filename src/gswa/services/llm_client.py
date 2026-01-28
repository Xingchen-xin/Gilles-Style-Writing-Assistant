"""LLM Client for local inference servers.

Uses OpenAI-compatible API to communicate with local LLM servers.
Supports multiple backends: vLLM, Ollama, LM Studio.
"""
import httpx
import json
import logging
from typing import Optional
from urllib.parse import urlparse

from gswa.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for OpenAI-compatible local LLM servers.

    Supports:
    - vLLM (Linux/NVIDIA GPU)
    - Ollama (Mac Apple Silicon / Linux)
    - LM Studio (Desktop)
    """

    # Allowed hosts for local-only operation
    ALLOWED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}

    # Stop tokens for LoRA models to prevent generating training data artifacts
    # The model tends to continue with references (". J ", ". (") after rewriting
    LORA_STOP_TOKENS = [". J ", ". (", "\n\n", "1. ", "2. ", "[1]", "[2]"]
    LORA_MAX_TOKENS = 512  # Limit tokens for single paragraph rewrites

    def __init__(self):
        """Initialize LLM client with security validation."""
        self.settings = get_settings()
        self._backend = self.settings.llm_backend
        self._base_url = self.settings.llm_base_url.rstrip("/")
        self._model = self.settings.vllm_model_name

        # Security: Ensure we're connecting locally
        self._validate_local_only()

        logger.info(f"LLM client initialized: backend={self._backend}, url={self._base_url}")

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
                    f"Only local LLM servers are allowed."
                )

    async def check_health(self) -> dict:
        """Check if LLM server is healthy.

        Returns:
            Dictionary with status and model info
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                # Try OpenAI-compatible /models endpoint
                resp = await client.get(f"{self._base_url}/models")
                if resp.status_code == 200:
                    data = resp.json()
                    models = data.get("data", [])
                    return {
                        "status": "connected",
                        "backend": self._backend,
                        "models": [m.get("id") for m in models]
                    }
            except httpx.ConnectError:
                return {
                    "status": "disconnected",
                    "backend": self._backend,
                    "error": f"Cannot connect to {self._backend} server at {self._base_url}"
                }
            except Exception as e:
                return {"status": "error", "backend": self._backend, "error": str(e)}
        return {"status": "disconnected", "backend": self._backend}

    def _is_lora_model(self, model: Optional[str]) -> bool:
        """Check if the model name refers to a LoRA adapter."""
        if not model:
            return False
        return model.startswith("gswa-") or model.startswith("lora-")

    def _format_for_lora(self, messages: list[dict]) -> list[dict]:
        """Format messages for LoRA models trained on alpaca format.

        LoRA adapters were trained with format: instruction + input (no system).
        The user message should already contain the instruction + text.
        We just need to remove any system message.

        Args:
            messages: Original messages

        Returns:
            Messages with only user content (system removed)
        """
        user_content = None
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break

        if user_content:
            return [{"role": "user", "content": user_content}]
        return messages

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate completion from LLM server.

        Args:
            messages: Chat messages [{"role": "...", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            model: Override model/adapter name (for LoRA selection)

        Returns:
            Generated text

        Raises:
            httpx.HTTPError: On API errors
        """
        # For LoRA adapters, use simpler format and add stop tokens
        effective_messages = messages
        effective_max_tokens = max_tokens
        effective_stop = stop

        if self._is_lora_model(model):
            effective_messages = self._format_for_lora(messages)
            effective_max_tokens = min(max_tokens, self.LORA_MAX_TOKENS)
            # Add LoRA stop tokens to prevent generating training artifacts
            effective_stop = list(set((stop or []) + self.LORA_STOP_TOKENS))
            logger.debug(f"Using LoRA format for model: {model}")

        payload = {
            "model": model or self._model,
            "messages": effective_messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        if effective_stop:
            payload["stop"] = effective_stop

        # Add stream: false for Ollama compatibility
        if self._backend == "ollama":
            payload["stream"] = False

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.settings.vllm_api_key}"}
            )
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"]

    async def stream_complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
        model: Optional[str] = None,
    ):
        """Stream completion deltas from LLM server.

        Args:
            messages: Chat messages [{"role": "...", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            model: Override model/adapter name (for LoRA selection)

        Yields:
            Delta text chunks
        """
        # For LoRA adapters, use simpler format and add stop tokens
        effective_messages = messages
        effective_max_tokens = max_tokens
        effective_stop = stop

        if self._is_lora_model(model):
            effective_messages = self._format_for_lora(messages)
            effective_max_tokens = min(max_tokens, self.LORA_MAX_TOKENS)
            effective_stop = list(set((stop or []) + self.LORA_STOP_TOKENS))
            logger.debug(f"Using LoRA format for streaming model: {model}")

        payload = {
            "model": model or self._model,
            "messages": effective_messages,
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "stream": True,
        }
        if effective_stop:
            payload["stop"] = effective_stop

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.settings.vllm_api_key}"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = (
                        choice.get("delta", {}).get("content")
                        or choice.get("message", {}).get("content")
                    )
                    if delta:
                        yield delta

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

    @property
    def backend(self) -> str:
        """Get the backend type."""
        return self._backend


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def reset_llm_client() -> None:
    """Reset the LLM client singleton (useful for testing)."""
    global _llm_client
    _llm_client = None
