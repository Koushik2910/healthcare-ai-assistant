"""Provider-level tests using fakes shaped like the real SDK surfaces.

Each concrete provider is constructed with an injected fake client, so these
tests exercise the real ``stream()``/``_chunks()``/``_translate_error()``
logic in ``GeminiProvider`` and ``OpenAICompatibleProvider`` without a
network call or a real API key -- the same pattern used for the Phase 1
domain-model tests.
"""

from __future__ import annotations

import pytest

from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.openrouter_provider import OpenRouterProvider
from src.models.chat import Message, Role
from src.models.llm import ProviderName
from src.utils.exceptions import LLMRateLimitError, LLMResponseError, LLMTimeoutError
from tests.fakes import FakeGeminiClient, FakeOpenAIClient

pytestmark = pytest.mark.unit


def _messages() -> list[Message]:
    return [Message(role=Role.USER, content="What is a balanced diet?")]


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_gemini_streams_concatenated_text() -> None:
    client = FakeGeminiClient(["Eat ", "a variety ", "of foods."])
    provider = GeminiProvider(
        api_key="unused",
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=0,
        client=client,
    )

    chunks = [
        chunk
        async for chunk in provider.stream(_messages(), system_prompt="Be concise.")
    ]

    assert "".join(chunks) == "Eat a variety of foods."


@pytest.mark.asyncio
async def test_gemini_recovers_after_transient_open_failures() -> None:
    """The connection fails twice, then the same client succeeds on retry."""
    client = FakeGeminiClient(
        ["recovered"],
        fail_first_n_opens=2,
        open_error=ConnectionError("cold start"),
    )
    provider = GeminiProvider(
        api_key="unused",
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=2,
        client=client,
    )

    chunks = [
        chunk
        async for chunk in provider.stream(_messages(), system_prompt="Be concise.")
    ]

    assert "".join(chunks) == "recovered"


@pytest.mark.asyncio
async def test_gemini_raises_once_retries_are_exhausted() -> None:
    client = FakeGeminiClient(
        [], fail_first_n_opens=10, open_error=ConnectionError("permanently down")
    )
    provider = GeminiProvider(
        api_key="unused",
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=2,
        client=client,
    )

    with pytest.raises(LLMResponseError):
        async for _ in provider.stream(_messages(), system_prompt="Be concise."):
            pass


@pytest.mark.asyncio
async def test_gemini_does_not_retry_mid_stream_failure() -> None:
    client = FakeGeminiClient(
        ["partial answer"],
        fail_after=1,
        mid_stream_error=RuntimeError("connection dropped"),
    )
    provider = GeminiProvider(
        api_key="unused",
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=3,
        client=client,
    )

    collected: list[str] = []
    with pytest.raises(LLMResponseError):
        async for chunk in provider.stream(_messages(), system_prompt="Be concise."):
            collected.append(chunk)

    assert collected == ["partial answer"]


@pytest.mark.parametrize(
    ("exc", "expected_type"),
    [
        (TimeoutError("deadline exceeded"), LLMTimeoutError),
        (RuntimeError("429 Too Many Requests: quota exceeded"), LLMRateLimitError),
        (RuntimeError("500 internal error"), LLMResponseError),
    ],
)
def test_gemini_translate_error_maps_to_expected_type(exc, expected_type) -> None:
    provider = GeminiProvider(
        api_key="unused",
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=0,
        client=FakeGeminiClient([]),
    )

    translated = provider._translate_error(exc)

    assert isinstance(translated, expected_type)
    assert translated.context["provider"] == "gemini"


# --------------------------------------------------------------------------- #
# Groq / OpenRouter (shared OpenAICompatibleProvider implementation)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_groq_streams_concatenated_text() -> None:
    client = FakeOpenAIClient(["Regular exercise ", "supports overall health."])
    provider = GroqProvider(
        api_key="unused",
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=0,
        client=client,
    )

    chunks = [
        chunk
        async for chunk in provider.stream(_messages(), system_prompt="Be concise.")
    ]

    assert "".join(chunks) == "Regular exercise supports overall health."


@pytest.mark.asyncio
async def test_openrouter_streams_concatenated_text() -> None:
    """OpenRouter inherits the exact streaming implementation Groq uses."""
    client = FakeOpenAIClient(["Hydration ", "matters too."])
    provider = OpenRouterProvider(
        api_key="unused",
        model="google/gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=0,
        client=client,
    )

    chunks = [
        chunk
        async for chunk in provider.stream(_messages(), system_prompt="Be concise.")
    ]

    assert "".join(chunks) == "Hydration matters too."
    assert provider.name is ProviderName.OPENROUTER


@pytest.mark.asyncio
async def test_openai_compatible_does_not_retry_mid_stream_failure() -> None:
    client = FakeOpenAIClient(
        ["partial"], fail_after=1, mid_stream_error=RuntimeError("stream reset")
    )
    provider = GroqProvider(
        api_key="unused",
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=3,
        client=client,
    )

    collected: list[str] = []
    with pytest.raises(LLMResponseError):
        async for chunk in provider.stream(_messages(), system_prompt="Be concise."):
            collected.append(chunk)

    assert collected == ["partial"]


@pytest.mark.parametrize(
    ("exc", "expected_type"),
    [
        (TimeoutError("request timed out"), LLMTimeoutError),
        (RuntimeError("rate limit exceeded, try again later"), LLMRateLimitError),
        (RuntimeError("service unavailable"), LLMResponseError),
    ],
)
def test_openai_compatible_translate_error_maps_to_expected_type(
    exc, expected_type
) -> None:
    provider = GroqProvider(
        api_key="unused",
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=0,
        client=FakeOpenAIClient([]),
    )

    translated = provider._translate_error(exc)

    assert isinstance(translated, expected_type)
    assert translated.context["provider"] == "groq"


def test_translate_error_reads_status_code_from_response_attribute() -> None:
    """Some SDK exceptions nest status_code under `.response` rather than
    exposing it directly; the translator must check both locations."""

    class _FakeResponse:
        status_code = 429

    class _FakeSdkError(Exception):
        response = _FakeResponse()

    provider = GroqProvider(
        api_key="unused",
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_output_tokens=256,
        timeout_seconds=5.0,
        max_retries=0,
        client=FakeOpenAIClient([]),
    )

    # Message text deliberately contains no "rate limit" substring, so a
    # pass here proves the status_code branch fired, not the message match.
    translated = provider._translate_error(_FakeSdkError("request rejected by upstream"))

    assert isinstance(translated, LLMRateLimitError)
