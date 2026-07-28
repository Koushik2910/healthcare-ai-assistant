"""Shared implementation for OpenAI-compatible chat-completions APIs.

Groq and OpenRouter both expose the same OpenAI ``chat.completions`` wire
format: Groq because it built an OpenAI-compatible API by convention,
OpenRouter because it is designed as a drop-in multi-provider proxy over
that same contract. This base class captures the one implementation the
two genuinely share; the concrete subclasses differ only in how their
client is constructed (``base_url`` and key). Keeping them as two thin
subclasses -- rather than one class with an ``if provider == ...`` branch,
or duplicating this logic twice -- means each provider's construction stays
independently testable, and keeps :class:`~src.llm.gemini_provider.GeminiProvider`
(an unrelated wire format) from ever being tempted into the same branch.
"""

from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from src.llm.base import LLMProvider
from src.llm.retry import stream_with_retry
from src.models.chat import Message
from src.utils.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Streams chat completions from any OpenAI-wire-format endpoint."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        """Construct the provider around an already-configured client.

        Args:
            client: A pre-built :class:`AsyncOpenAI` client pointed at the
                concrete provider's ``base_url``. Subclasses build this;
                tests may inject a fake with the same interface.
            model: Model id in whatever form the target endpoint expects.
            temperature: See :class:`~src.llm.base.LLMProvider`.
            max_output_tokens: See :class:`~src.llm.base.LLMProvider`.
            timeout_seconds: See :class:`~src.llm.base.LLMProvider`.
            max_retries: See :mod:`src.llm.retry`.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
        self.max_retries = max_retries
        self._client = client

    async def stream(
        self, messages: list[Message], *, system_prompt: str
    ) -> AsyncIterator[str]:
        """Yield text fragments for one turn. See base class for the contract."""
        wire_messages = [{"role": "system", "content": system_prompt}] + [
            message.to_provider_dict() for message in messages
        ]

        def open_stream() -> AsyncIterator[str]:
            return self._chunks(wire_messages)

        async for chunk in stream_with_retry(
            open_stream,
            max_retries=self.max_retries,
            translate_error=self._translate_error,
        ):
            yield chunk

    async def _chunks(
        self, wire_messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Open the completion stream and yield only non-empty content deltas."""
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=wire_messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            timeout=self.timeout_seconds,
            stream=True,
        )
        async for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    def _translate_error(self, exc: Exception) -> LLMError:
        """Map an OpenAI-SDK exception to this application's error hierarchy.

        Deliberately uses duck typing (a ``status_code`` attribute, or one
        nested under ``.response``) rather than ``isinstance`` checks against
        the SDK's own exception classes. Those classes' constructors have
        changed shape across ``openai`` package versions; matching on the
        attributes the SDK's exceptions are documented to expose is more
        resilient to that churn, and is what makes this method testable with
        plain, hand-built exceptions instead of SDK internals.
        """
        provider = self.name.value
        message = str(exc)
        lowered = message.lower()
        status = getattr(exc, "status_code", None)
        if status is None:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None) if response is not None else None
        context = {"provider": provider, "model": self.model}

        if isinstance(exc, TimeoutError) or "timeout" in lowered:
            return LLMTimeoutError(f"{provider} request timed out: {message}", context=context)
        if status == 429 or "rate limit" in lowered:
            return LLMRateLimitError(f"{provider} rate limit: {message}", context=context)
        return LLMResponseError(
            f"{provider} request failed: {message}", context={**context, "status": status}
        )
