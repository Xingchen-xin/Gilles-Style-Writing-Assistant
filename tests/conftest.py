"""Pytest configuration and fixtures."""
import os
import pytest

# Ensure tests run in offline mode
os.environ.setdefault("ALLOW_EXTERNAL_API", "false")


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset settings cache before each test."""
    from gswa.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
