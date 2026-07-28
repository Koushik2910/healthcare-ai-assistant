"""Tests for the shared ``generate()`` implementation on ``LLMProvider``.

``generate()`` is deliberately not overridden per provider -- it is built
once, on top of ``stream()``, so the streaming and non-streaming paths
cannot silently diverge. These tests exercise that shared implementation
through the fake provider defined in ``tests/fakes.py``.
"""

from __future__ import annotations

import pytest

from src.models.chat import Message, Role
from src.models.llm import ProviderName
from tests.fakes import FakeLLMProvider

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_generate_concatenates_stream_chunks_in_order() -> None:
    provider = FakeLLMProvider(chunks=["The capital ", "of France ", "is Paris."])

    result = await provider.generate(
        [Message(role=Role.USER, content="What is the capital of France?")],
        system_prompt="You are a helpful assistant.",
    )

    assert result.text == "The capital of France is Paris."
    assert result.provider is ProviderName.GEMINI
    assert result.model == "fake-model"
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_generate_reports_empty_when_stream_yields_nothing() -> None:
    provider = FakeLLMProvider(chunks=[])

    result = await provider.generate([], system_prompt="system")

    assert result.is_empty is True


@pytest.mark.asyncio
async def test_generate_reports_empty_for_whitespace_only_output() -> None:
    """is_empty strips whitespace, so a stream of only spaces is empty too."""
    provider = FakeLLMProvider(chunks=["   ", "\n"])

    result = await provider.generate([], system_prompt="system")

    assert result.is_empty is True


@pytest.mark.asyncio
async def test_generate_propagates_stream_failures() -> None:
    provider = FakeLLMProvider(
        chunks=[], fail_before_first_chunk=TimeoutError("provider down")
    )

    with pytest.raises(TimeoutError):
        await provider.generate([], system_prompt="system")


@pytest.mark.asyncio
async def test_call_count_increments_once_per_generate_call() -> None:
    """Confirms generate() opens exactly one stream, not one per chunk consumed."""
    provider = FakeLLMProvider(chunks=["a", "b", "c"])

    await provider.generate([], system_prompt="system")
    await provider.generate([], system_prompt="system")

    assert provider.call_count == 2
