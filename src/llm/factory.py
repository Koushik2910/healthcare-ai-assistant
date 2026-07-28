"""Provider factory: the single place ``LLM_PROVIDER`` is switched on.

Every other module in the application depends on :class:`LLMProvider`, never
on a concrete subclass. This function is therefore the only place that needs
to change to add a fourth provider, and the only place a test needs to patch
to exercise the chat service against a fake backend.
"""

from __future__ import annotations

from src.config.settings import Settings, get_settings
from src.llm.base import LLMProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.openrouter_provider import OpenRouterProvider
from src.utils.exceptions import ConfigurationError


def get_llm(settings: Settings | None = None) -> LLMProvider:
    """Construct the active LLM provider from configuration.

    Args:
        settings: Overrides the process-wide settings singleton. Tests use
            this to construct a provider under controlled configuration
            without touching real environment variables.

    Returns:
        A ready-to-use provider instance for ``settings.llm_provider``.

    Raises:
        ConfigurationError: If the selected provider's API key is missing.
            :class:`~src.config.settings.Settings` validation already
            guarantees this cannot happen for the *active* provider at
            startup; this is a defensive second check for callers that
            construct a ``Settings`` object directly rather than through
            :func:`~src.config.settings.get_settings`.
    """
    settings = settings or get_settings()
    common = {
        "temperature": settings.llm_temperature,
        "max_output_tokens": settings.llm_max_output_tokens,
        "timeout_seconds": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }

    if settings.llm_provider == "gemini":
        if settings.gemini_api_key is None:
            raise ConfigurationError("GEMINI_API_KEY is not set.")
        return GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
            **common,
        )

    if settings.llm_provider == "groq":
        if settings.groq_api_key is None:
            raise ConfigurationError("GROQ_API_KEY is not set.")
        return GroqProvider(
            api_key=settings.groq_api_key.get_secret_value(),
            model=settings.groq_model,
            **common,
        )

    if settings.llm_provider == "openrouter":
        if settings.openrouter_api_key is None:
            raise ConfigurationError("OPENROUTER_API_KEY is not set.")
        return OpenRouterProvider(
            api_key=settings.openrouter_api_key.get_secret_value(),
            model=settings.openrouter_model,
            **common,
        )

    raise ConfigurationError(f"Unknown llm_provider: {settings.llm_provider!r}")
