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

``_primary`` is the configured provider. ``_fallback`` is always a Groq
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

# Safe fallback shown when the LLM call fails entirely (after failover).
_LLM_FAILURE_MESSAGE = (
    "I'm having trouble reaching the AI service right now. "
    "Please try again in a moment."
)


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

        # Build the fallback provider lazily — only when Groq key exists and
        # the primary is not already Groq.
        self._fallback: LLMProvider | None = self._build_fallback()

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
            async for chunk in self._stream_with_failover(
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
            yield f"\n\n⚠️ {_OUTPUT_BLOCKED_MESSAGE}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_fallback(self) -> LLMProvider | None:
        """Construct the Groq fallback provider, or return None if unavailable."""
        if self._primary.name == ProviderName.GROQ:
            # Primary IS Groq — no separate fallback needed.
            return None
        if self._settings.groq_api_key is None:
            log.info(
                "Auto-failover disabled: GROQ_API_KEY not set. "
                "Add it to .env to enable Gemini → Groq failover."
            )
            return None
        try:
            return GroqProvider(
                api_key=self._settings.groq_api_key.get_secret_value(),
                model=self._settings.groq_model,
                temperature=self._settings.llm_temperature,
                max_output_tokens=self._settings.llm_max_output_tokens,
                timeout_seconds=self._settings.llm_timeout_seconds,
                max_retries=self._settings.llm_max_retries,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Failed to build Groq fallback provider; failover disabled.",
                extra={"error": str(exc)},
            )
            return None

    async def _call_with_failover(
        self,
        messages: list[Message],
        *,
        system_prompt: str,
    ):
        """Call the primary; fall back to Groq on transient pre-stream errors."""
        try:
            return await self._primary.generate(messages, system_prompt=system_prompt)
        except (LLMRateLimitError, LLMTimeoutError) as primary_exc:
            if self._fallback is None:
                raise

            log.warning(
                "Primary provider failed; retrying on Groq fallback.",
                extra={
                    "primary": self._primary.name.value,
                    "error_code": getattr(primary_exc, "code", "unknown"),
                },
            )
            try:
                return await self._fallback.generate(messages, system_prompt=system_prompt)
            except HealthAssistantError:
                # Fallback also failed — re-raise the original primary error
                # so the caller sees the root cause, not the fallback's error.
                raise primary_exc

    async def _stream_with_failover(
        self,
        messages: list[Message],
        *,
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """Stream from the primary; fall back to Groq on pre-stream errors only."""
        provider = self._primary
        try:
            async for chunk in provider.stream(messages, system_prompt=system_prompt):
                yield chunk
            return
        except (LLMRateLimitError, LLMTimeoutError) as primary_exc:
            if self._fallback is None:
                raise

            log.warning(
                "Primary stream failed before first token; switching to Groq.",
                extra={"primary": self._primary.name.value},
            )
            try:
                async for chunk in self._fallback.stream(
                    messages, system_prompt=system_prompt
                ):
                    yield chunk
            except HealthAssistantError:
                raise primary_exc

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
