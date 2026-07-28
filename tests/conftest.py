"""Shared test fixtures.

The single most important thing this file does is make the suite hermetic.
A settings test that passes only because the developer happens to have a
``.env`` with a valid key is not a test. The autouse fixture below strips
every application environment variable, disables ``.env`` loading and clears
the settings cache before each test, so results are identical on a laptop, in
CI and on a reviewer's machine.
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from src.config.settings import Settings, get_settings

#: Every environment variable the application reads.
APP_ENV_VARS = [
    "APP_NAME", "APP_ENV",
    "LOG_LEVEL", "LOG_FORMAT", "LOG_USER_CONTENT",
    "LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL",
    "GROQ_API_KEY", "GROQ_MODEL",
    "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
    "LLM_TEMPERATURE", "LLM_MAX_OUTPUT_TOKENS",
    "LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES",
    "RAG_ENABLED", "CHROMA_COLLECTION", "EMBEDDING_MODEL",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_TOP_K", "RETRIEVAL_MIN_SCORE",
    "SAFETY_STRICT_MODE", "MAX_INPUT_CHARS",
    "MAX_HISTORY_TURNS", "HISTORY_SUMMARY_THRESHOLD_CHARS",
]


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove ambient configuration so every test starts from known defaults."""
    for name in APP_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    # Prevent the developer's real .env from bleeding into assertions.
    monkeypatch.setitem(Settings.model_config, "env_file", None)

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def build_settings() -> Any:
    """Return a factory that builds Settings with sensible test defaults.

    Supplies a dummy Gemini key so the provider-key validator is satisfied,
    while allowing any field to be overridden per test.
    """

    def _build(**overrides: Any) -> Settings:
        defaults: dict[str, Any] = {
            "llm_provider": "gemini",
            "gemini_api_key": "test-key-not-real",
        }
        defaults.update(overrides)
        return Settings(**defaults)

    return _build
