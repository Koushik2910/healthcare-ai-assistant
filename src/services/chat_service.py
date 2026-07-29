"""ChatService — the single owner of one complete chat turn.

Design rationale
----------------
Every chat turn passes through exactly five stages, in this order:

1. **Input guard** — deterministic pattern screen. If it fires, the model is
   never called; a template response is returned immediately.
2. **Prompt build** — ``PromptBuilder`` assembles the system prompt for this
   turn, injecting RAG context when available.
3. **LLM call with failover** — primary provider first; if it raises a
   transient error *before the first token*, the fallback (Groq) is tried
   once. Mid-stream errors are not retried (see CLAUDE.md decision 3).
4. **Output guard** — validates the generated text. In strict mode (the
   default), any violation replaces the response with a safe fallback.
5. **Response assembly** — builds a ``ChatResponse`` carrying the message,
   source, citations, disclaimer flag, and latency.

**Why one class owns all five stages:**

The alternative — spreading the pipeline across the UI layer, a middleware,
and utility functions — means the ordering and error handling live in several
places and can drift. A single ``ChatService.chat()`` method means the
pipeline is always the same pipeline, whether called by Streamlit, a FastAPI
endpoint, or a test.

**Auto-failover:**

``_primary`` is the configured provider. ``_fallbacks`` is an ordered chain
provider, constructed lazily only when (a) the primary is not already Groq
and (b) a Groq key is configured. If neither condition holds, failover is
silently disabled — the primary's error propagates as normal. This avoids a
startup failure when the user has only a Gemini key.

Failover fires only for ``LLMRateLimitError`` and ``LLMTimeoutError`` — the
two transient conditions where the primary is overloaded or slow. It does
NOT fire for ``LLMResponseError`` (the model responded but the content was
bad) because retrying on a different provider for a content failure would
serve a different response to the same question, which is surprising and not
the right fix.

**RAG integration (Phase 5):**

``ChatService`` accepts an optional ``retriever`` argument. When ``None``
(the default until Phase 5), every turn is ungrounded and
``ResponseSource.MODEL_ONLY`` is recorded. When a retriever is supplied it
is called before prompt assembly and its result is threaded through the
prompt builder to inject context and later to build ``Citation`` objects.
"""

from __future__ import annotations

import logging
import re
import time
from typing import AsyncIterator

from src.config.settings import Settings, get_settings
from src.llm.base import LLMProvider
from src.llm.groq_provider import GroqProvider
from src.llm.openrouter_provider import OpenRouterProvider
from src.models.chat import (
    ChatResponse,
    Citation,
    Conversation,
    Message,
    ResponseSource,
    Role,
)
from src.models.llm import ProviderName
from src.models.rag import RetrievalResult
from src.models.safety import SafetyAction
from src.prompts import MEDICAL_DISCLAIMER, PromptBuilder, refusal_for
from src.safety import InputGuard, OutputGuard
from src.utils.exceptions import (
    EmptyInputError,
    HealthAssistantError,
    InputTooLongError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    user_facing_message,
)

log = logging.getLogger(__name__)

# Safe fallback shown when output validation blocks a response.
_OUTPUT_BLOCKED_MESSAGE = (
    "I wasn't able to provide a safe response to that question. "
    "Please rephrase, or consult a qualified healthcare professional directly."
)

# Safe fallback shown when ALL providers in the cascade are exhausted.
_LLM_FAILURE_MESSAGE = (
    "The service is busy at the moment. Please wait a few seconds and try again."
)

# Tracks which provider served the last stream — written by _prepend,
# read by the UI to show the active provider indicator.
_ACTIVE_PROVIDER: str = "gemini"

# Sentinel prefix yielded as the ONLY chunk when the output guard fires
# post-stream.  The UI detects this prefix and replaces the entire placeholder
# with just the blocked message — no partial streamed content is shown.
STREAM_BLOCKED_SENTINEL = "\x00BLOCKED\x00"


class ChatService:
    """Orchestrates one complete question-answer turn.

    Args:
        primary: The configured LLM provider (Gemini, Groq, or OpenRouter).
        settings: Application settings. Defaults to the process-wide singleton.
        retriever: Optional callable ``(query: str) -> RetrievalResult``.
            When supplied, called before prompt assembly. Omit until Phase 5.
    """

    def __init__(
        self,
        primary: LLMProvider,
        *,
        settings: Settings | None = None,
        retriever=None,
    ) -> None:
        self._primary = primary
        self._settings = settings or get_settings()
        self._retriever = retriever

        # Build the ordered fallback chain: Groq → OpenRouter.
        # Empty list means no failover. Primary is never in this list.
        self._fallbacks: list[LLMProvider] = self._build_fallback_chain()

        self._guard = InputGuard(max_chars=self._settings.max_input_chars)
        self._output_guard = OutputGuard()
        self._builder = PromptBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_text: str,
        conversation: Conversation,
    ) -> ChatResponse:
        """Process one user message and return the assistant's response.

        This method never raises. All errors — LLM failures, guard violations,
        unexpected exceptions — are caught and returned as a ``ChatResponse``
        with ``source=ResponseSource.FALLBACK`` and a user-safe message.

        Args:
            user_text: The raw text the user typed.
            conversation: The current conversation state. Not mutated here —
                the caller is responsible for appending both the user message
                and the returned assistant message.

        Returns:
            A fully assembled :class:`~src.models.chat.ChatResponse`.
        """
        started = time.perf_counter()

        # ── Stage 1: input guard ──────────────────────────────────────
        try:
            verdict = self._guard.screen(user_text)
        except EmptyInputError as exc:
            return self._empty_input_response(exc)
        except InputTooLongError as exc:
            return self._error_response(exc, latency_ms=self._elapsed(started))

        if verdict.blocks_model_call:
            return self._guardrail_response(verdict, latency_ms=self._elapsed(started))

        # ── Stage 2: retrieval (Phase 5 — no-op until retriever supplied) ──
        retrieval: RetrievalResult | None = None
        if self._retriever is not None:
            try:
                retrieval = self._retriever(user_text)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Retrieval failed; proceeding ungrounded.",
                    extra={"error": str(exc)},
                )
                retrieval = RetrievalResult(query=user_text, degraded=True)

        # ── Stage 3: prompt assembly ──────────────────────────────────
        prompt_ctx = self._builder.build(retrieval=retrieval)

        # Build the message list the LLM provider expects: recent history
        # (system messages are handled via system_prompt, not in the list)
        history = [
            msg for msg in conversation.recent(self._settings.max_history_turns)
            if msg.role != Role.SYSTEM
        ]

        # ── Stage 4: LLM call with failover ──────────────────────────
        try:
            result = await self._call_with_failover(
                history, system_prompt=prompt_ctx.system_prompt
            )
        except HealthAssistantError as exc:
            log.error("LLM call failed after failover.", extra=exc.to_log_fields())
            return self._error_response(exc, latency_ms=self._elapsed(started))
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected error during LLM call.")
            return self._error_response(exc, latency_ms=self._elapsed(started))

        if result.is_empty:
            log.warning(
                "Provider returned empty text.",
                extra={"provider": result.provider.value, "model": result.model},
            )
            return self._fallback_response(
                _LLM_FAILURE_MESSAGE, latency_ms=self._elapsed(started)
            )

        # ── Stage 5: output guard ─────────────────────────────────────
        result = result.model_copy(
            update={"text": self._strip_llm_disclaimer(result.text)}
        )
        validation = self._output_guard.validate(result.text)
        if validation.must_block or (
            self._settings.safety_strict_mode and validation.max_severity >= 2
        ):
            log.warning(
                "Output guard blocked response.",
                extra={
                    "rules": [i.rule_id for i in validation.issues],
                    "max_severity": validation.max_severity,
                },
            )
            return self._fallback_response(
                _OUTPUT_BLOCKED_MESSAGE, latency_ms=self._elapsed(started)
            )

        # ── Stage 6: assemble ChatResponse ────────────────────────────
        source = (
            ResponseSource.GROUNDED if prompt_ctx.grounded else ResponseSource.MODEL_ONLY
        )
        citations = self._build_citations(prompt_ctx.injected_chunks)
        disclaimer = self._needs_disclaimer(result.text)

        message = Message(
            role=Role.ASSISTANT,
            content=result.text,
            source=source,
            citations=citations,
        )
        return ChatResponse(
            message=message,
            source=source,
            citations=citations,
            disclaimer=MEDICAL_DISCLAIMER if disclaimer else None,
            refused=False,
            latency_ms=self._elapsed(started),
        )

    # ------------------------------------------------------------------
    # Streaming variant (for Streamlit's st.write_stream)
    # ------------------------------------------------------------------

    async def stream_chat(
        self,
        user_text: str,
        conversation: Conversation,
    ) -> AsyncIterator[str]:
        """Yield response text token by token for streaming UIs.

        Applies the same input guard and prompt assembly as :meth:`chat`, but
        yields the model's text chunks incrementally rather than waiting for
        completion. Output guard runs after the full text is assembled from
        the stream — mid-stream interruption is not supported (see CLAUDE.md
        decision 3).

        If the input guard fires, a single chunk containing the refusal text
        is yielded and the iterator closes. This keeps the Streamlit streaming
        contract simple: the caller always gets an iterator.

        Args:
            user_text: The raw text the user typed.
            conversation: Current conversation state.

        Yields:
            Text fragments in generation order.
        """
        # Stage 1: input guard
        try:
            verdict = self._guard.screen(user_text)
        except (EmptyInputError, InputTooLongError) as exc:
            yield user_facing_message(exc)
            return

        if verdict.blocks_model_call:
            yield refusal_for(verdict.category)
            return

        # Stage 2: retrieval
        retrieval: RetrievalResult | None = None
        if self._retriever is not None:
            try:
                retrieval = self._retriever(user_text)
            except Exception:  # noqa: BLE001
                retrieval = RetrievalResult(query=user_text, degraded=True)

        # Stage 3: prompt assembly
        prompt_ctx = self._builder.build(retrieval=retrieval)
        history = [
            msg for msg in conversation.recent(self._settings.max_history_turns)
            if msg.role != Role.SYSTEM
        ]

        # Stage 4: stream with failover
        chunks: list[str] = []
        try:
            async for chunk in await self._stream_with_failover(
                history, system_prompt=prompt_ctx.system_prompt
            ):
                chunks.append(chunk)
                yield chunk
        except HealthAssistantError as exc:
            log.error("Stream failed.", extra=exc.to_log_fields())
            yield f"\n\n{user_facing_message(exc)}"
            return
        except Exception:  # noqa: BLE001
            log.exception("Unexpected stream error.")
            yield f"\n\n{_LLM_FAILURE_MESSAGE}"
            return

        # Stage 5: output guard (post-stream)
        full_text = "".join(chunks)
        validation = self._output_guard.validate(full_text)
        if validation.must_block or (
            self._settings.safety_strict_mode and validation.max_severity >= 2
        ):
            log.warning(
                "Output guard blocked streamed response.",
                extra={"rules": [i.rule_id for i in validation.issues]},
            )
            # We can't un-yield what's already been sent; yield a replacement
            # notice. The Streamlit UI handles this via st.empty() rewrite.
            yield f"{STREAM_BLOCKED_SENTINEL}{_OUTPUT_BLOCKED_MESSAGE}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_fallback_chain(self) -> list[LLMProvider]:
        """Build the ordered fallback chain: Groq (tier-1) → OpenRouter (tier-2).

        Only providers whose API keys are present are included.  The primary
        is never added to its own fallback chain.  Chain order is always
        Groq → OpenRouter regardless of which provider is set as LLM_PROVIDER.

        Returns:
            Ordered list of fallback providers (may be empty).
        """
        common = {
            "temperature": self._settings.llm_temperature,
            "max_output_tokens": self._settings.llm_max_output_tokens,
            "timeout_seconds": self._settings.llm_timeout_seconds,
            "max_retries": self._settings.llm_max_retries,
        }
        chain: list[LLMProvider] = []

        # Tier-1 fallback: Groq (free tier)
        if (
            self._primary.name != ProviderName.GROQ
            and self._settings.groq_api_key is not None
        ):
            try:
                chain.append(GroqProvider(
                    api_key=self._settings.groq_api_key.get_secret_value(),
                    model=self._settings.groq_model,
                    **common,
                ))
                log.info("Fallback chain: Groq added as tier-1 fallback.")
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to build Groq fallback.", extra={"error": str(exc)})

        # Tier-2 fallback: OpenRouter (paid, $50 buffer — guaranteed availability)
        if (
            self._primary.name != ProviderName.OPENROUTER
            and self._settings.openrouter_api_key is not None
        ):
            try:
                chain.append(OpenRouterProvider(
                    api_key=self._settings.openrouter_api_key.get_secret_value(),
                    model=self._settings.openrouter_model,
                    **common,
                ))
                log.info("Fallback chain: OpenRouter added as tier-2 fallback.")
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to build OpenRouter fallback.", extra={"error": str(exc)})

        if not chain:
            log.info(
                "Auto-failover disabled: no fallback keys configured. "
                "Add GROQ_API_KEY and/or OPENROUTER_API_KEY to .env to enable."
            )
        else:
            chain_names = " → ".join(
                [self._primary.name.value.upper()] +
                [p.name.value.upper() for p in chain]
            )
            log.info("[CASCADE] Provider chain ready: %s", chain_names)
        return chain

    async def _call_with_failover(
        self,
        messages: list[Message],
        *,
        system_prompt: str,
    ):
        """Call providers in cascade order until one succeeds.

        Tries primary first, then Groq, then OpenRouter.  On rate-limit or
        timeout, moves to the next provider.  Raises the last seen error if
        all providers are exhausted.
        """
        last_exc: Exception | None = None
        for i, provider in enumerate([self._primary, *self._fallbacks]):
            try:
                return await provider.generate(messages, system_prompt=system_prompt)
            except LLMError as exc:
                remaining = len(self._fallbacks) - i
                next_name = ([self._primary, *self._fallbacks][i + 1].name.value
                             if remaining > 0 else "none")
                log.warning(
                    "[CASCADE] %s unavailable (%s) — trying %s next.",
                    provider.name.value.upper(),
                    type(exc).__name__,
                    next_name.upper() if remaining > 0 else "NO MORE PROVIDERS",
                )
                last_exc = exc
        raise last_exc  # type: ignore[misc]

    async def _stream_with_failover(
        self,
        messages: list[Message],
        *,
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """Return a stream from the first available provider in the cascade.

        Plain ``async def`` returning ``AsyncIterator`` — NOT an async
        generator.  try/except inside an async generator does not catch
        exceptions from iterated sub-generators; a plain coroutine has
        normal semantics and can catch and failover correctly.

        Cascade: primary → Groq → OpenRouter.  Each provider is probed for
        its first chunk.  On rate-limit or timeout the next is tried.  The
        probed first chunk is prepended back so the caller receives a complete
        unmodified stream.
        """
        providers = [self._primary, *self._fallbacks]

        for i, provider in enumerate(providers):
            try:
                stream = provider.stream(messages, system_prompt=system_prompt)
                first_chunk = await stream.__anext__()
            except StopAsyncIteration:
                async def _empty() -> AsyncIterator[str]:
                    return
                    yield  # pragma: no cover
                return _empty()
            except LLMError as exc:
                # Catch all LLM errors (rate limit, timeout, response error)
                # so the cascade continues to the next provider regardless of
                # the specific failure mode.  LLMResponseError (empty/blocked
                # payload from Groq) was previously not caught, causing Groq
                # to be skipped even when it could serve the next request.
                remaining = len(providers) - i - 1
                next_name = providers[i + 1].name.value if remaining > 0 else "none"
                log.warning(
                    "[CASCADE] %s unavailable (%s) — trying %s next.",
                    provider.name.value.upper(),
                    type(exc).__name__,
                    next_name.upper() if remaining > 0 else "NO MORE PROVIDERS",
                )
                if remaining == 0:
                    raise
                continue

            tier_label = ["primary", "tier-1 fallback", "tier-2 fallback"]
            label = tier_label[i] if i < len(tier_label) else f"tier-{i} fallback"
            log.info(
                "[CASCADE] ✓ Serving from %s (%s)",
                provider.name.value.upper(),
                label,
            )

            # Set active provider immediately so the sidebar reads the correct
            # value even before the first chunk is yielded.
            import src.services.chat_service as _self_mod
            _self_mod._ACTIVE_PROVIDER = provider.name.value

            async def _prepend(
                first: str,
                rest: AsyncIterator[str],
            ) -> AsyncIterator[str]:
                yield first
                async for chunk in rest:
                    yield chunk

            return _prepend(first_chunk, stream)

        raise LLMRateLimitError("All providers in the cascade are unavailable.")

    @staticmethod
    def _build_citations(
        injected_chunks: list[tuple[int, str, str]],
    ) -> list[Citation]:
        """Convert injected chunk tuples into Citation domain objects."""
        return [
            Citation(marker=marker, title=title, source="Knowledge Base", snippet=text[:200])
            for marker, title, text in injected_chunks
        ]

    @staticmethod
    def _strip_llm_disclaimer(text: str) -> str:
        """Remove any trailing disclaimer block the LLM appended.

        The prompt instructs the model not to append the disclaimer footer,
        but LLMs occasionally do anyway.  Strips the canonical pattern and
        common variants so the UI disclaimer pill is never duplicated.
        """
        import re as _re
        pattern = _re.compile(
            r"\s*-{2,}\s*\*?This information is for general educational"
            r".*?(\*?\s*-{2,})?\s*$",
            _re.IGNORECASE | _re.DOTALL,
        )
        cleaned = pattern.sub("", text).rstrip()
        return cleaned if cleaned else text

    @staticmethod
    def _needs_disclaimer(text: str) -> bool:
        """Return True when the response discusses clinical content."""
        clinical_signals = re.compile(
            r"\b(sympt\w+|dosage|medication|treatment|diagnosis|condition|"
            r"disease|surgery|prescription|therapy|drug|tablet|mg)\b",
            re.IGNORECASE,
        )
        return bool(clinical_signals.search(text))

    @staticmethod
    def _elapsed(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)

    # ------------------------------------------------------------------
    # Response constructors
    # ------------------------------------------------------------------

    @staticmethod
    def _guardrail_response(verdict, *, latency_ms: int) -> ChatResponse:
        source = (
            ResponseSource.ESCALATION
            if verdict.action == SafetyAction.ESCALATE
            else ResponseSource.GUARDRAIL
        )
        text = refusal_for(verdict.category)
        return ChatResponse(
            message=Message(role=Role.ASSISTANT, content=text, source=source),
            source=source,
            refused=True,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _fallback_response(text: str, *, latency_ms: int) -> ChatResponse:
        return ChatResponse(
            message=Message(
                role=Role.ASSISTANT, content=text, source=ResponseSource.FALLBACK
            ),
            source=ResponseSource.FALLBACK,
            refused=False,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _error_response(exc: Exception, *, latency_ms: int) -> ChatResponse:
        text = user_facing_message(exc)
        return ChatResponse(
            message=Message(
                role=Role.ASSISTANT, content=text, source=ResponseSource.FALLBACK
            ),
            source=ResponseSource.FALLBACK,
            refused=False,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _empty_input_response(exc: EmptyInputError) -> ChatResponse:
        return ChatResponse(
            message=Message(
                role=Role.ASSISTANT,
                content=exc.user_message,
                source=ResponseSource.GUARDRAIL,
            ),
            source=ResponseSource.GUARDRAIL,
            refused=True,
            latency_ms=0,
        )
