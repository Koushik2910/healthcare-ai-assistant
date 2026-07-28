"""Structured logging with correlation IDs and health-content redaction.

Design rationale
----------------
Three decisions worth defending in review:

1. **Structured (JSON) output by default.** Every record carries a stable set
   of fields, so logs are greppable and machine-parsable without regex
   archaeology. A human-readable ``console`` format is available for local
   development.

2. **Correlation IDs via :mod:`contextvars`.** One chat turn touches the
   safety layer, the retriever and the model provider. A ``turn_id`` bound
   once at the start of the turn is attached automatically to every record
   emitted downstream, so a single failing turn can be reconstructed from
   the log stream. ``contextvars`` (rather than a global) keeps this correct
   under concurrency, which matters once the FastAPI adapter exists.

3. **Redaction is on by default.** In a healthcare assistant the user's
   message *is* the sensitive data -- symptoms, medications, mental-health
   disclosures. Logging it verbatim would turn a debug convenience into a
   data-protection liability. :func:`redact` therefore records a stable hash
   plus a length, which is enough to correlate and reproduce a bug report
   without retaining the content itself. Verbatim logging requires an
   explicit opt-in that is never enabled in a deployed environment.

This module intentionally has **no dependency on the settings module**, so
that configuration errors can themselves be logged. The entrypoint wires the
two together.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator, Literal
from contextlib import contextmanager

#: Identifies a single chat turn across every module that handles it.
turn_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "turn_id", default="-"
)

#: Identifies the conversation a turn belongs to.
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default="-"
)

#: Module-level switch flipped by :func:`setup_logging`.
_LOG_USER_CONTENT = False

#: Attributes present on every stdlib LogRecord; anything else is treated as
#: caller-supplied structured context and merged into the JSON payload.
_STANDARD_RECORD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }
)


class ContextFilter(logging.Filter):
    """Attach the current turn and session identifiers to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.turn_id = turn_id_var.get()
        record.session_id = session_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "turn_id": getattr(record, "turn_id", "-"),
            "session_id": getattr(record, "session_id", "-"),
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in payload:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable format for local development."""

    _FMT = "%(asctime)s %(levelname)-8s [%(turn_id)s] %(name)s: %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt="%H:%M:%S")


def setup_logging(
    level: str = "INFO",
    log_format: Literal["json", "console"] = "json",
    log_user_content: bool = False,
) -> None:
    """Configure the root logger. Safe to call more than once.

    Args:
        level: Standard logging level name, for example ``"DEBUG"``.
        log_format: ``"json"`` for structured output, ``"console"`` for
            human-readable local development output.
        log_user_content: When ``True``, :func:`redact` returns message text
            verbatim. Intended only for local debugging of the prompt
            pipeline; must remain ``False`` anywhere real user data flows.
    """
    global _LOG_USER_CONTENT
    _LOG_USER_CONTENT = log_user_content

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter() if log_format == "json" else ConsoleFormatter())
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These libraries are extremely chatty at INFO and drown the signal.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if log_user_content:
        root.warning(
            "User content logging is ENABLED. Do not use this setting with "
            "real user data.",
            extra={"event": "unsafe_logging_enabled"},
        )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Args:
        name: Conventionally ``__name__`` of the calling module.
    """
    return logging.getLogger(name)


def redact(text: str | None, *, keep_prefix: int = 0) -> str:
    """Return a log-safe representation of user-supplied text.

    Produces a stable fingerprint rather than the content, so that repeated
    occurrences of the same message can be correlated across log lines
    without the message itself ever being written to disk.

    Args:
        text: The raw text, typically a user's health question.
        keep_prefix: Number of leading characters to retain in the clear.
            Use sparingly; even a short prefix of a health question can be
            identifying.

    Returns:
        Either the verbatim text (only when user-content logging was
        explicitly enabled) or a ``<redacted len=.. sha=..>`` marker.

    Examples:
        >>> setup_logging(log_user_content=False)
        >>> redact("I have a headache").startswith("<redacted")
        True
    """
    if text is None:
        return "<none>"
    if _LOG_USER_CONTENT:
        return text

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    prefix = f" prefix={text[:keep_prefix]!r}" if keep_prefix > 0 else ""
    return f"<redacted len={len(text)} sha={digest}{prefix}>"


def new_turn_id() -> str:
    """Return a short unique identifier for one chat turn."""
    return uuid.uuid4().hex[:12]


@contextmanager
def log_context(
    *, turn_id: str | None = None, session_id: str | None = None
) -> Iterator[str]:
    """Bind correlation identifiers for the duration of a block.

    Args:
        turn_id: Identifier for this turn. Generated when omitted.
        session_id: Identifier for the owning conversation. Left unchanged
            when omitted.

    Yields:
        The turn identifier in effect inside the block.

    Examples:
        >>> with log_context(session_id="abc") as tid:  # doctest: +SKIP
        ...     get_logger(__name__).info("handling turn")
    """
    resolved_turn = turn_id or new_turn_id()
    turn_token = turn_id_var.set(resolved_turn)
    session_token = session_id_var.set(session_id) if session_id else None
    try:
        yield resolved_turn
    finally:
        turn_id_var.reset(turn_token)
        if session_token is not None:
            session_id_var.reset(session_token)
