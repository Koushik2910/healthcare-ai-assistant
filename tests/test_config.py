"""Configuration tests.

These assert the *fail-fast* contract: an invalid configuration must be
rejected at construction with an actionable message, never accepted and
allowed to misbehave later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.config.settings import PROJECT_ROOT, Settings, get_settings
from src.utils.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


def test_defaults_are_valid_when_active_key_present(build_settings: Any) -> None:
    settings = build_settings()

    assert settings.llm_provider == "gemini"
    assert settings.llm_temperature == 0.3
    assert settings.rag_enabled is True
    assert settings.safety_strict_mode is True


def test_missing_active_provider_key_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(llm_provider="gemini", gemini_api_key=None)

    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_inactive_provider_key_may_be_absent(build_settings: Any) -> None:
    """Only the selected provider's key is mandatory.

    gemini_api_key is explicitly cleared here because the fixture defaults
    it to a dummy value so gemini-provider tests don't need to repeat it;
    this test must override that default to exercise the groq-only path.
    """
    settings = build_settings(
        llm_provider="groq", gemini_api_key=None, groq_api_key="groq-test-key"
    )

    assert settings.groq_api_key is not None
    assert settings.gemini_api_key is None
    assert settings.active_model == settings.groq_model


def test_chunk_overlap_must_be_smaller_than_chunk_size(build_settings: Any) -> None:
    with pytest.raises(ValidationError) as exc_info:
        build_settings(chunk_size=400, chunk_overlap=400)

    assert "chunk_overlap" in str(exc_info.value)


def test_verbatim_user_logging_rejected_outside_local(build_settings: Any) -> None:
    """Logging health questions verbatim is a local-development-only affordance."""
    with pytest.raises(ValidationError) as exc_info:
        build_settings(app_env="production", log_user_content=True)

    assert "log_user_content" in str(exc_info.value)


def test_verbatim_user_logging_allowed_locally(build_settings: Any) -> None:
    settings = build_settings(app_env="local", log_user_content=True)

    assert settings.log_user_content is True


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_bounds_enforced(build_settings: Any, temperature: float) -> None:
    with pytest.raises(ValidationError):
        build_settings(llm_temperature=temperature)


def test_active_key_and_model_follow_provider(build_settings: Any) -> None:
    gemini = build_settings(llm_provider="gemini", gemini_api_key="g-key")
    assert gemini.active_api_key is not None
    assert gemini.active_api_key.get_secret_value() == "g-key"
    assert gemini.active_model == gemini.gemini_model

    groq = build_settings(
        llm_provider="groq", gemini_api_key=None, groq_api_key="q-key"
    )
    assert groq.active_api_key is not None
    assert groq.active_api_key.get_secret_value() == "q-key"
    assert groq.active_model == groq.groq_model


def test_secrets_are_not_exposed_by_repr_or_dump(build_settings: Any) -> None:
    settings = build_settings(gemini_api_key="super-secret-value")

    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.safe_dump())
    assert settings.safe_dump()["gemini_api_key"] == "***set***"


def test_relative_paths_resolve_against_project_root(build_settings: Any) -> None:
    settings = build_settings(session_dir=Path("data/custom_sessions"))

    assert settings.session_dir.is_absolute()
    assert settings.session_dir == (PROJECT_ROOT / "data" / "custom_sessions").resolve()


def test_get_settings_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "cached-key")

    first = get_settings()
    second = get_settings()

    assert first is second


def test_get_settings_raises_actionable_configuration_error() -> None:
    """A first-time user with no key should get instructions, not a traceback."""
    with pytest.raises(ConfigurationError) as exc_info:
        get_settings()

    error = exc_info.value
    assert error.code == "configuration_error"
    assert "GEMINI_API_KEY" in error.user_message
    assert ".env" in error.user_message
