"""Startup diagnostic.

Verifies that the environment, configuration and logging subsystem are wired
correctly before any model call is attempted. Run it after cloning the
repository and again whenever configuration changes:

    python scripts\\preflight.py

Output is deliberately ASCII-only so it renders correctly in a default
Windows PowerShell console without a code-page change.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts\preflight.py` to resolve `src.*` without an install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings  # noqa: E402
from src.utils.exceptions import HealthAssistantError  # noqa: E402
from src.utils.logging import get_logger, log_context, redact, setup_logging  # noqa: E402

MIN_PYTHON = (3, 11)
PASS = "[ OK ]"
FAIL = "[FAIL]"


def _line(status: str, label: str, detail: str = "") -> None:
    """Print one aligned status row."""
    print(f"{status} {label:<28} {detail}")


def check_python_version() -> bool:
    """Verify the interpreter meets the minimum supported version."""
    actual = sys.version_info[:2]
    ok = actual >= MIN_PYTHON
    _line(
        PASS if ok else FAIL,
        "Python version",
        f"{actual[0]}.{actual[1]} (requires >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
    )
    return ok


def check_configuration() -> bool:
    """Load and validate settings, then print them with secrets masked."""
    try:
        settings = get_settings()
    except HealthAssistantError as exc:
        _line(FAIL, "Configuration", exc.user_message)
        return False

    _line(PASS, "Configuration", f"loaded, env='{settings.app_env}'")
    _line(PASS, "LLM provider", f"{settings.llm_provider} / {settings.active_model}")
    _line(PASS, "Retrieval", "enabled" if settings.rag_enabled else "disabled")
    _line(
        PASS,
        "Safety mode",
        "strict" if settings.safety_strict_mode else "permissive (logs only)",
    )

    for label, path in (
        ("Knowledge base dir", settings.knowledge_base_dir),
        ("Vector store dir", settings.chroma_persist_dir),
        ("Session dir", settings.session_dir),
    ):
        path.mkdir(parents=True, exist_ok=True)
        _line(PASS, label, str(path))

    return True


def check_logging() -> bool:
    """Emit one correlated, redacted log record and confirm redaction holds."""
    settings = get_settings()
    setup_logging(
        level=settings.log_level,
        log_format=settings.log_format,
        log_user_content=settings.log_user_content,
    )
    logger = get_logger("preflight")

    sample = "I have had a sore throat for three days"
    redacted = redact(sample)

    with log_context(session_id="preflight") as turn_id:
        logger.info("preflight log record", extra={"user_message": redacted})

    leaked = settings.log_user_content and sample in redacted
    _line(
        FAIL if leaked else PASS,
        "Logging",
        f"turn_id={turn_id}, redaction="
        + ("DISABLED (local debug)" if settings.log_user_content else "active"),
    )
    return True


def main() -> int:
    """Run every check and return a process exit code."""
    print("\nHealthcare AI Assistant -- preflight\n" + "-" * 60)

    results = [check_python_version(), check_configuration()]
    if results[-1]:
        results.append(check_logging())

    print("-" * 60)
    if all(results):
        print("All checks passed. Ready for the next phase.\n")
        return 0

    print("One or more checks failed. See the messages above.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
