"""Tests for the streaming retry policy.

These exercise ``stream_with_retry`` directly against hand-built async
generators, with no LLM provider and no SDK involved. That makes them the
highest-confidence tests in the LLM layer: the retry/no-retry boundary is
exactly the piece of logic most worth getting right, and it's fully
decoupled from any external dependency.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from src.llm.retry import stream_with_retry
from src.utils.exceptions import LLMError, LLMTimeoutError

pytestmark = pytest.mark.unit


async def _collect(agen: AsyncIterator[str]) -> list[str]:
    return [chunk async for chunk in agen]


def _translate(exc: Exception) -> LLMError:
    return LLMTimeoutError(str(exc))


@pytest.mark.asyncio
async def test_successful_stream_yields_all_chunks_on_first_attempt() -> None:
    attempts = 0

    def open_stream() -> AsyncIterator[str]:
        nonlocal attempts
        attempts += 1

        async def gen() -> AsyncIterator[str]:
            yield "Hello"
            yield " world"

        return gen()

    result = await _collect(
        stream_with_retry(open_stream, max_retries=2, translate_error=_translate)
    )

    assert result == ["Hello", " world"]
    assert attempts == 1


@pytest.mark.asyncio
async def test_failure_before_first_chunk_is_retried() -> None:
    attempts = 0

    def open_stream() -> AsyncIterator[str]:
        nonlocal attempts
        attempts += 1

        async def gen() -> AsyncIterator[str]:
            if attempts < 3:
                raise ConnectionError("transient")
                yield  # pragma: no cover -- unreachable, keeps this a generator
            yield "recovered"

        return gen()

    result = await _collect(
        stream_with_retry(
            open_stream,
            max_retries=3,
            translate_error=_translate,
            base_delay_seconds=0.0,
        )
    )

    assert result == ["recovered"]
    assert attempts == 3


@pytest.mark.asyncio
async def test_retries_exhausted_raises_translated_error() -> None:
    attempts = 0

    def open_stream() -> AsyncIterator[str]:
        nonlocal attempts
        attempts += 1

        async def gen() -> AsyncIterator[str]:
            raise ConnectionError("permanently down")
            yield  # pragma: no cover

        return gen()

    with pytest.raises(LLMTimeoutError):
        await _collect(
            stream_with_retry(
                open_stream,
                max_retries=2,
                translate_error=_translate,
                base_delay_seconds=0.0,
            )
        )

    assert attempts == 3  # initial attempt + 2 retries


@pytest.mark.asyncio
async def test_failure_after_first_chunk_is_not_retried() -> None:
    """The core safety property: no duplicated output after streaming starts."""
    attempts = 0

    def open_stream() -> AsyncIterator[str]:
        nonlocal attempts
        attempts += 1

        async def gen() -> AsyncIterator[str]:
            yield "partial answer"
            raise ConnectionError("dropped mid-stream")

        return gen()

    collected: list[str] = []
    with pytest.raises(LLMTimeoutError):
        async for chunk in stream_with_retry(
            open_stream,
            max_retries=5,
            translate_error=_translate,
            base_delay_seconds=0.0,
        ):
            collected.append(chunk)

    assert collected == ["partial answer"]
    assert attempts == 1  # never re-opened after the first chunk succeeded


@pytest.mark.asyncio
async def test_empty_stream_yields_nothing_and_does_not_raise() -> None:
    def open_stream() -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            return
            yield  # pragma: no cover -- unreachable, keeps this a generator

        return gen()

    result = await _collect(
        stream_with_retry(open_stream, max_retries=1, translate_error=_translate)
    )

    assert result == []


@pytest.mark.asyncio
async def test_zero_max_retries_still_allows_one_attempt() -> None:
    attempts = 0

    def open_stream() -> AsyncIterator[str]:
        nonlocal attempts
        attempts += 1

        async def gen() -> AsyncIterator[str]:
            yield "only try"

        return gen()

    result = await _collect(
        stream_with_retry(open_stream, max_retries=0, translate_error=_translate)
    )

    assert result == ["only try"]
    assert attempts == 1
