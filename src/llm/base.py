"""Provider-agnostic LLM interface.

Every concrete provider (Gemini, Groq, OpenRouter) speaks a different SDK,
with different streaming shapes, different exception types and different
wire formats. Nothing else in the application -- the prompt layer, the
safety validators, the chat service -- should have to know or care which one
is active. This module is that seam: one abstract class, two methods wide,
so swapping providers is a one-line config change (``LLM_PROVIDER=groq``)
rather than a change anywhere else in the codebase.

``generate()`` (non-streaming) is provided for free on top of ``stream()``
by concatenation, rather than implemented separately per provider. Every
provider used here supports streaming natively, and a single shared
implementation removes an entire class of bug where the streaming and
non-streaming code paths quietly drift apart.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import AsyncIterator

from src.models.chat import Message
from src.models.llm import GenerationResult, ProviderName


class LLMProvider(ABC):
    """Abstract interface every language-model backend must satisfy."""

    #: Set by each concrete subclass; identifies the provider in logs,
    #: evaluation reports and generation results.
    name: ProviderName

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> None:
        """Store the generation parameters shared by every provider.

        Args:
            model: Provider-specific model identifier.
            temperature: Sampling temperature. Low by convention for a
                healthcare assistant -- see ``Settings.llm_temperature``.
            max_output_tokens: Ceiling on generated tokens.
            timeout_seconds: Per-request timeout enforced by the concrete
                provider's transport.
        """
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def stream(
        self, messages: list[Message], *, system_prompt: str
    ) -> AsyncIterator[str]:
        """Yield response text incrementally as it is generated.

        Args:
            messages: Conversation history, oldest first, in service-layer
                :class:`Message` form. Implementations convert this to
                their own wire format internally.
            system_prompt: The fully composed system prompt for this turn,
                assembled by the prompt layer.

        Yields:
            Successive text fragments; concatenated in order they form the
            complete response.

        Raises:
            LLMTimeoutError: No response within ``timeout_seconds``.
            LLMRateLimitError: The provider's quota was exhausted.
            LLMResponseError: The provider replied, but the payload was
                empty, malformed, or blocked by the provider's own safety
                filter.
        """
        raise NotImplementedError

    async def generate(
        self, messages: list[Message], *, system_prompt: str
    ) -> GenerationResult:
        """Return the complete response as a single value.

        Built on :meth:`stream`, so the streaming and non-streaming call
        paths cannot diverge in behaviour -- only in how a caller consumes
        the result.

        Args:
            messages: See :meth:`stream`.
            system_prompt: See :meth:`stream`.

        Returns:
            The concatenated text plus timing and provider metadata.
        """
        started = time.perf_counter()
        parts: list[str] = []
        async for chunk in self.stream(messages, system_prompt=system_prompt):
            parts.append(chunk)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return GenerationResult(
            text="".join(parts),
            provider=self.name,
            model=self.model,
            latency_ms=elapsed_ms,
        )
