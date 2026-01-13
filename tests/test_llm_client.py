"""Tests for LLM client."""
import os
import pytest
from gswa.services.llm_client import LLMClient, get_llm_client
from gswa.config import get_settings


class TestLLMClientSecurity:
    """Security tests for LLM client."""

    def test_localhost_allowed(self):
        """Test that localhost is allowed."""
        # Default config uses localhost
        client = LLMClient()
        assert "localhost" in client._base_url or "127.0.0.1" in client._base_url

    def test_external_host_blocked(self):
        """Test that external hosts are blocked."""
        os.environ["VLLM_BASE_URL"] = "http://api.openai.com/v1"
        get_settings.cache_clear()

        try:
            with pytest.raises(ValueError, match="External API calls forbidden"):
                LLMClient()
        finally:
            os.environ.pop("VLLM_BASE_URL", None)
            get_settings.cache_clear()

    def test_127_0_0_1_allowed(self):
        """Test that 127.0.0.1 is allowed."""
        os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
        get_settings.cache_clear()

        try:
            client = LLMClient()
            assert "127.0.0.1" in client._base_url
        finally:
            os.environ.pop("VLLM_BASE_URL", None)
            get_settings.cache_clear()


class TestLLMClientConfig:
    """Configuration tests for LLM client."""

    def test_model_name(self):
        """Test model name configuration."""
        client = LLMClient()
        assert client.model_name == get_settings().vllm_model_name

    def test_base_url_trailing_slash_removed(self):
        """Test that trailing slash is removed from base URL."""
        os.environ["VLLM_BASE_URL"] = "http://localhost:8000/v1/"
        get_settings.cache_clear()

        try:
            client = LLMClient()
            assert not client._base_url.endswith("/")
        finally:
            os.environ.pop("VLLM_BASE_URL", None)
            get_settings.cache_clear()


class TestLLMClientSingleton:
    """Tests for singleton pattern."""

    def test_get_llm_client_returns_same_instance(self):
        """Test that get_llm_client returns singleton."""
        # Reset singleton
        import gswa.services.llm_client as llm_module
        llm_module._llm_client = None

        client1 = get_llm_client()
        client2 = get_llm_client()

        assert client1 is client2


class TestLLMClientMethods:
    """Tests for LLM client methods (mocked)."""

    @pytest.mark.asyncio
    async def test_check_health_handles_connection_error(self):
        """Test health check handles connection errors gracefully."""
        client = LLMClient()
        # This will fail to connect since no server is running
        result = await client.check_health()

        assert result["status"] in ["disconnected", "error"]
