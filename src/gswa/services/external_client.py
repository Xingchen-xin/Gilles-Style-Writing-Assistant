"""External OpenAI-Compatible API Client.

Supports Groq, OpenRouter, and any other OpenAI-compatible external API.
Only active when LLM_BACKEND is an external type and ALLOW_EXTERNAL_API=true.
"""
import httpx
import json
import logging
from typing import Optional, AsyncIterator

from gswa.config import get_settings

logger = logging.getLogger(__name__)

# Endpoint configs for known providers
PROVIDER_CONFIGS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
}


class ExternalOpenAIClient:
    """OpenAI-compatible client for external APIs (Groq, OpenRouter, etc.).

    Same interface as LLMClient/AnthropicClient for drop-in use via Generator.
    """

    def __init__(self):
        self.settings = get_settings()
        self._provider = self.settings.llm_backend  # "groq" or "openrouter"
        cfg = PROVIDER_CONFIGS[self._provider]

        self._base_url = cfg["base_url"].rstrip("/")
        self._model = self.settings.api_model or cfg["default_model"]
        self._api_key = self.settings.api_key

        logger.info(f"ExternalOpenAIClient: provider={self._provider}, model={self._model}")

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def backend(self) -> str:
        return self._provider

    async def check_health(self) -> dict:
        """Check API connectivity with a minimal request."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                if resp.status_code == 200:
                    return {"status": "connected", "backend": self._provider, "models": [self._model]}
                else:
                    return {"status": "error", "backend": self._provider,
                            "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"status": "error", "backend": self._provider, "error": str(e)}

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate completion via OpenAI-compatible API."""
        payload = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
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
    ) -> AsyncIterator[str]:
        """Stream completion via OpenAI-compatible API."""
        payload = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue


# Singleton
_external_client: Optional[ExternalOpenAIClient] = None


def get_external_client() -> ExternalOpenAIClient:
    global _external_client
    if _external_client is None:
        _external_client = ExternalOpenAIClient()
    return _external_client


def reset_external_client() -> None:
    global _external_client
    _external_client = None
