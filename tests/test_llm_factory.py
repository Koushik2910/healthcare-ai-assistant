"""Tests for the provider factory.

These confirm ``get_llm()`` wires configuration to the correct concrete
class and that generation parameters flow through unchanged. Constructing
each concrete provider here does not make a network call -- the SDK client
constructors only store the key -- so these tests need the packages
installed but not a valid key or connectivity.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.llm.factory import get_llm
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.openrouter_provider import OpenRouterProvider
from src.models.llm import ProviderName

pytestmark = pytest.mark.unit


def test_factory_selects_gemini_by_default(build_settings: Any) -> None:
    settings = build_settings()

    provider = get_llm(settings)

    assert isinstance(provider, GeminiProvider)
    assert provider.name is ProviderName.GEMINI
    assert provider.model == settings.gemini_model


def test_factory_selects_groq_when_configured(build_settings: Any) -> None:
    settings = build_settings(
        llm_provider="groq", gemini_api_key=None, groq_api_key="groq-test-key"
    )

    provider = get_llm(settings)

    assert isinstance(provider, GroqProvider)
    assert provider.name is ProviderName.GROQ
    assert provider.model == settings.groq_model


def test_factory_selects_openrouter_when_configured(build_settings: Any) -> None:
    settings = build_settings(
        llm_provider="openrouter",
        gemini_api_key=None,
        openrouter_api_key="or-test-key",
    )

    provider = get_llm(settings)

    assert isinstance(provider, OpenRouterProvider)
    assert provider.name is ProviderName.OPENROUTER
    assert provider.model == settings.openrouter_model


def test_factory_propagates_generation_parameters(build_settings: Any) -> None:
    settings = build_settings(
        llm_temperature=0.7, llm_max_output_tokens=512, llm_timeout_seconds=10.0
    )

    provider = get_llm(settings)

    assert provider.temperature == 0.7
    assert provider.max_output_tokens == 512
    assert provider.timeout_seconds == 10.0
