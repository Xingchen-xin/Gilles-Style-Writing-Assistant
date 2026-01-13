"""Tests for configuration module."""
import os
import pytest
from gswa.config import Settings, get_settings


def test_default_settings():
    """Test that default settings are secure."""
    settings = Settings()
    assert settings.allow_external_api is False
    assert "localhost" in settings.vllm_base_url or "127.0.0.1" in settings.vllm_base_url


def test_security_validation_blocks_external_api():
    """Test that ALLOW_EXTERNAL_API=true raises an error."""
    # Set environment variable
    os.environ["ALLOW_EXTERNAL_API"] = "true"

    try:
        settings = Settings()
        with pytest.raises(ValueError, match="FORBIDDEN"):
            settings.validate_security()
    finally:
        # Clean up
        os.environ.pop("ALLOW_EXTERNAL_API", None)
        # Clear the cache
        get_settings.cache_clear()


def test_get_settings_validates_security():
    """Test that get_settings validates security on creation."""
    # Ensure clean state
    os.environ.pop("ALLOW_EXTERNAL_API", None)
    get_settings.cache_clear()

    # Should not raise
    settings = get_settings()
    assert settings.allow_external_api is False


def test_threshold_defaults():
    """Test that thresholds have correct defaults."""
    settings = Settings()
    assert settings.threshold_ngram_max_match == 12
    assert settings.threshold_ngram_overlap == 0.15
    assert settings.threshold_embed_top1 == 0.88


def test_generation_defaults():
    """Test that generation parameters have correct defaults."""
    settings = Settings()
    assert settings.default_n_variants == 3
    assert settings.max_n_variants == 5
    assert settings.temperature_base == 0.3
    assert settings.max_new_tokens == 1024
