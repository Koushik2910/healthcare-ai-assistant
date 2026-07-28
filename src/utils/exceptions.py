"""Application exception hierarchy.

Design rationale
----------------
Every exception carries two distinct payloads:

* ``detail``       -- engineer-facing. Logged, never rendered to the user.
* ``user_message`` -- user-facing. Safe, calm, actionable. Rendered in the UI.

This separation is what allows the presentation layer to satisfy the
"never expose stack traces / friendly user errors" requirement without
littering the codebase with ``try/except`` blocks that stringify raw
exceptions. The UI catches :class:`HealthAssistantError` and renders
``user_message``; the logging layer records ``detail`` and the traceback.

In a healthcare context this is not cosmetic. A leaked traceback can expose
prompt internals, file paths, model names and -- worst case -- fragments of a
user's health question echoed back in an error string.
"""

from __future__ import annotations

from typing import Any


class HealthAssistantError(Exception):
    """Base class for every error raised deliberately by this application.

    Anything that escapes as a bare :class:`Exception` is, by definition, a
    bug rather than a handled condition, and is reported to the user with the
    generic fallback message.
    """

    #: Default user-facing text. Subclasses override.
    default_user_message: str = (
        "Something went wrong on our side. Please try again in a moment."
    )

    #: Short machine-readable code, used in logs and tests.
    code: str = "internal_error"

    def __init__(
        self,
        detail: str,
        *,
        user_message: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the error.

        Args:
            detail: Engineer-facing description. Logged, never displayed.
            user_message: Overrides :attr:`default_user_message` when a more
                specific explanation is safe to show.
            context: Structured, non-sensitive key/values attached to logs
                (for example ``{"provider": "gemini", "attempt": 2}``).
        """
        super().__init__(detail)
        self.detail = detail
        self.user_message = user_message or self.default_user_message
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        return self.detail

    def to_log_fields(self) -> dict[str, Any]:
        """Return the structured fields this error contributes to a log record."""
        return {"error_code": self.code, "error_detail": self.detail, **self.context}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class ConfigurationError(HealthAssistantError):
    """Raised at startup when configuration is missing or invalid.

    Deliberately fails fast: it is better to refuse to boot than to serve a
    chatbot whose safety layer is misconfigured.
    """

    code = "configuration_error"
    default_user_message = (
        "The application is not configured correctly. "
        "Please check the setup instructions in README.md."
    )


# --------------------------------------------------------------------------- #
# LLM provider
# --------------------------------------------------------------------------- #


class LLMError(HealthAssistantError):
    """Base class for failures originating from a language-model provider."""

    code = "llm_error"
    default_user_message = (
        "I could not reach the AI service just now. Please try again shortly."
    )


class LLMTimeoutError(LLMError):
    """The provider did not respond within the configured timeout."""

    code = "llm_timeout"
    default_user_message = (
        "That took longer than expected. Please try asking again, "
        "or shorten your question."
    )


class LLMRateLimitError(LLMError):
    """The provider rejected the request because a quota was exhausted."""

    code = "llm_rate_limit"
    default_user_message = (
        "The service is busy at the moment. Please wait a few seconds "
        "and try again."
    )


class LLMResponseError(LLMError):
    """The provider replied, but the payload was empty, malformed or blocked.

    Includes provider-side safety blocks, which are logged distinctly from
    transport failures because they carry very different operational meaning.
    """

    code = "llm_response_error"


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


class RetrievalError(HealthAssistantError):
    """The knowledge base could not be queried.

    Non-fatal by design: the chat service degrades to an ungrounded answer
    with a reduced-confidence notice rather than failing the whole turn.
    """

    code = "retrieval_error"
    default_user_message = (
        "I could not consult my reference material for this answer, "
        "so please treat it as general information only."
    )


class KnowledgeBaseNotBuiltError(RetrievalError):
    """The vector store directory is absent or empty."""

    code = "knowledge_base_missing"
    default_user_message = (
        "The knowledge base has not been built yet. "
        "Run the ingestion script described in README.md."
    )


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #


class SafetyError(HealthAssistantError):
    """Base class for guardrail failures.

    Note the distinction from a *refusal*: a refusal is a normal, successful
    outcome represented by a :class:`~src.models.safety.SafetyVerdict`, not an
    exception. This class covers the guardrail machinery itself breaking.
    """

    code = "safety_error"
    default_user_message = (
        "I could not safely process that request. Please rephrase your question."
    )


class InputTooLongError(SafetyError):
    """User input exceeded the configured character ceiling.

    A cheap denial-of-wallet and prompt-stuffing control applied before any
    billable model call.
    """

    code = "input_too_long"
    default_user_message = (
        "That message is too long for me to process. "
        "Please shorten it and try again."
    )


class EmptyInputError(SafetyError):
    """User input was blank or whitespace only."""

    code = "empty_input"
    default_user_message = "Please type a question before sending."


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #


class SessionError(HealthAssistantError):
    """Conversation state could not be read or written."""

    code = "session_error"
    default_user_message = (
        "I lost track of this conversation. Starting a new chat should fix it."
    )


def user_facing_message(exc: BaseException) -> str:
    """Return a message that is always safe to display to an end user.

    Any exception the application did not anticipate collapses to the generic
    fallback, guaranteeing the UI never renders a raw exception string.

    Args:
        exc: The caught exception.

    Returns:
        Text suitable for direct rendering in the interface.
    """
    if isinstance(exc, HealthAssistantError):
        return exc.user_message
    return HealthAssistantError.default_user_message
