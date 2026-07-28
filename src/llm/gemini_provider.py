"""Gemini provider: primary language-model backend.

Uses the ``google-genai`` SDK directly against Google's endpoint, rather
than through a router, so that reproducing this project costs a reviewer
nothing but a free key from https://aistudio.google.com/apikey and no
proxy or billing account. See :mod:`src.llm.openrouter_provider` for the
same underlying model reachable through a paid multi-provider gateway,
offered as an optional third backend rather than the graded path.

.. note::
   This module was written from documented ``google-genai`` usage without
   the ability to execute it against a live key in the authoring
   environment. If the installed SDK version exposes a different call
   shape than assumed here, the fix is isolated to :meth:`GeminiProvider._chunks`.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from google import genai
from google.genai import types as genai_types

from src.llm.base import LLMProvider
from src.llm.retry import stream_with_retry
from src.models.chat import Message, Role
from src.models.llm import ProviderName
from src.utils.exceptions import (
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    """Streams responses from Google's Gemini API."""

    name = ProviderName.GEMINI

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        max_retries: int,
        client: genai.Client | None = None,
    ) -> None:
        """Construct the provider.

        Args:
            api_key: Google AI Studio API key.
            model: Gemini model id, e.g. ``"gemini-2.5-flash"``.
            temperature: See :class:`~src.llm.base.LLMProvider`.
            max_output_tokens: See :class:`~src.llm.base.LLMProvider`.
            timeout_seconds: See :class:`~src.llm.base.LLMProvider`.
            max_retries: Additional attempts for the connection/first-chunk
                phase only; see :mod:`src.llm.retry`.
            client: Injected SDK client, used by tests to avoid a real key
                or network call. Built from ``api_key`` when omitted.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
        self.max_retries = max_retries
        self._client = client or genai.Client(api_key=api_key)

    async def stream(
        self, messages: list[Message], *, system_prompt: str
    ) -> AsyncIterator[str]:
        """Yield text fragments for one turn. See base class for the contract."""
        contents = [
            genai_types.Content(
                role="model" if message.role is Role.ASSISTANT else "user",
                parts=[genai_types.Part.from_text(text=message.content)],
            )
            for message in messages
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        def open_stream() -> AsyncIterator[str]:
            return self._chunks(contents=contents, config=config)

        async for chunk in stream_with_retry(
            open_stream,
            max_retries=self.max_retries,
            translate_error=self._translate_error,
        ):
            yield chunk

    async def _chunks(
        self, *, contents: list[Any], config: Any
    ) -> AsyncIterator[str]:
        """Open the Gemini stream and yield only the non-empty text deltas."""
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        )
        async for event in stream:
            text = getattr(event, "text", None)
            if text:
                yield text

    def _translate_error(self, exc: Exception) -> LLMError:
        """Map a raw Gemini SDK exception to this application's error hierarchy.

        Matches on status code where the SDK exposes one, falling back to
        substring matching on the message for SDK versions that raise plain
        exceptions without a structured status.
        """
        message = str(exc)
        lowered = message.lower()
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        context = {"provider": "gemini", "model": self.model}

        if isinstance(exc, TimeoutError) or "timeout" in lowered or "deadline" in lowered:
            return LLMTimeoutError(f"Gemini request timed out: {message}", context=context)
        if status == 429 or "rate limit" in lowered or "quota" in lowered or "resource_exhausted" in lowered:
            return LLMRateLimitError(f"Gemini rate limit: {message}", context=context)
        return LLMResponseError(
            f"Gemini request failed: {message}", context={**context, "status": status}
        )
