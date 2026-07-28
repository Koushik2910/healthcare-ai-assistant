"""Logging and exception tests.

The redaction tests are the ones that matter. In a healthcare assistant the
user's message is the sensitive payload, so "we do not log health questions"
has to be an assertion in CI rather than a claim in a README.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

import pytest

from src.utils import logging as app_logging
from src.utils.exceptions import (
    ConfigurationError,
    HealthAssistantError,
    LLMTimeoutError,
    user_facing_message,
)
from src.utils.logging import (
    JsonFormatter,
    log_context,
    redact,
    session_id_var,
    setup_logging,
    turn_id_var,
)

pytestmark = pytest.mark.unit

SENSITIVE = "I have been having chest pain since Tuesday"


@pytest.fixture(autouse=True)
def reset_logging_state() -> Iterator[None]:
    """Restore module-level logging state after each test."""
    yield
    app_logging._LOG_USER_CONTENT = False
    logging.getLogger().handlers.clear()


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


def test_user_content_is_redacted_by_default() -> None:
    setup_logging(log_user_content=False)

    output = redact(SENSITIVE)

    assert SENSITIVE not in output
    assert "chest pain" not in output
    assert output.startswith("<redacted")
    assert f"len={len(SENSITIVE)}" in output


def test_redaction_is_stable_for_identical_input() -> None:
    """A stable fingerprint lets repeated failures be correlated across logs."""
    setup_logging(log_user_content=False)

    assert redact(SENSITIVE) == redact(SENSITIVE)
    assert redact(SENSITIVE) != redact(SENSITIVE + "?")


def test_redaction_can_be_disabled_explicitly_for_local_debugging() -> None:
    setup_logging(log_user_content=True)

    assert redact(SENSITIVE) == SENSITIVE


def test_prefix_retention_is_opt_in_and_bounded() -> None:
    setup_logging(log_user_content=False)

    assert "prefix=" not in redact(SENSITIVE)
    assert "prefix='I have'" in redact(SENSITIVE, keep_prefix=6)


def test_none_is_handled() -> None:
    assert redact(None) == "<none>"


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #


def test_log_context_binds_and_restores_identifiers() -> None:
    assert turn_id_var.get() == "-"

    with log_context(session_id="session-abc") as turn_id:
        assert turn_id_var.get() == turn_id
        assert session_id_var.get() == "session-abc"

    assert turn_id_var.get() == "-"
    assert session_id_var.get() == "-"


def test_nested_contexts_restore_the_outer_turn() -> None:
    with log_context(turn_id="outer"):
        with log_context(turn_id="inner"):
            assert turn_id_var.get() == "inner"
        assert turn_id_var.get() == "outer"


def test_json_formatter_emits_structured_fields_and_extras() -> None:
    record = logging.LogRecord(
        name="src.services.chat",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="turn completed",
        args=(),
        exc_info=None,
    )
    record.turn_id = "turn-1"
    record.session_id = "session-1"
    record.latency_ms = 812

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "turn completed"
    assert payload["level"] == "INFO"
    assert payload["turn_id"] == "turn-1"
    assert payload["latency_ms"] == 812
    assert "timestamp" in payload


def test_log_records_carry_context_end_to_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_logging(level="INFO", log_format="json")

    with log_context(turn_id="turn-xyz", session_id="session-xyz"):
        logging.getLogger("src.test").info(
            "screening input", extra={"user_message": redact(SENSITIVE)}
        )

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    assert payload["turn_id"] == "turn-xyz"
    assert payload["session_id"] == "session-xyz"
    assert SENSITIVE not in json.dumps(payload)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


def test_user_message_defaults_per_exception_type() -> None:
    assert (
        "try asking again" in LLMTimeoutError("provider deadline exceeded").user_message
    )
    assert "README" in ConfigurationError("no key").user_message


def test_internal_detail_never_leaks_into_user_message() -> None:
    error = LLMTimeoutError(
        "gemini-2.5-flash timed out after 30.0s at https://internal.endpoint",
        context={"provider": "gemini", "attempt": 2},
    )

    assert "internal.endpoint" not in error.user_message
    assert "internal.endpoint" in error.detail
    assert error.to_log_fields()["provider"] == "gemini"
    assert error.to_log_fields()["error_code"] == "llm_timeout"


def test_unexpected_exceptions_collapse_to_generic_message() -> None:
    """An unhandled bug must never render a raw exception string in the UI."""
    leaky = ZeroDivisionError("division by zero in src/rag/retriever.py line 88")

    message = user_facing_message(leaky)

    assert message == HealthAssistantError.default_user_message
    assert "retriever.py" not in message


def test_explicit_user_message_overrides_default() -> None:
    error = LLMTimeoutError("detail", user_message="The AI service is slow right now.")

    assert user_facing_message(error) == "The AI service is slow right now."
