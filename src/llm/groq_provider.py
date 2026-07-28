"""Groq provider: low-latency fallback backend.

Selected as the fallback because Groq's inference is fast enough that a
failover from Gemini doesn't visibly stall a conversation, and it has its
own generous free tier -- so exercising the fallback path in a demo or in
CI costs nothing.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from src.llm.openai_compatible import OpenAICompatibleProvider
from src.models.llm import ProviderName

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(OpenAICompatibleProvider):
    """Streams chat completions from Groq's OpenAI-compatible endpoint."""

    name = ProviderName.GROQ

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        max_retries: int,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Construct the provider.

        Args:
            api_key: Groq API key.
            model: Groq model id, e.g. ``"llama-3.3-70b-versatile"``.
            temperature: See :class:`~src.llm.base.LLMProvider`.
            max_output_tokens: See :class:`~src.llm.base.LLMProvider`.
            timeout_seconds: See :class:`~src.llm.base.LLMProvider`.
            max_retries: See :mod:`src.llm.retry`.
            client: Injected client for tests. Built from ``api_key`` when
                omitted.
        """
        super().__init__(
            client=client or AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL),
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
