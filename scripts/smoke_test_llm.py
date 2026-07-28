"""Manual smoke test: one real call to the currently configured LLM provider.

Every automated test in ``tests/`` runs against fakes and makes no network
call, by design -- that's what keeps the suite fast and free to run. This
script is the deliberate exception: it makes one real request using your
real ``.env`` configuration, so you can confirm the SDK call shape this
project assumes actually matches the installed provider library, and see
real token usage and latency before building anything on top of it.

Usage (PowerShell):

    python scripts\\smoke_test_llm.py
    python scripts\\smoke_test_llm.py --provider groq
    python scripts\\smoke_test_llm.py --provider openrouter

The ``--provider`` flag temporarily overrides ``LLM_PROVIDER`` for this one
run only; it does not modify your ``.env`` file. Omit it to use whichever
provider ``.env`` currently selects.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import get_settings  # noqa: E402
from src.llm.factory import get_llm  # noqa: E402
from src.models.chat import Message, Role  # noqa: E402
from src.utils.exceptions import HealthAssistantError  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

SYSTEM_PROMPT = (
    "You are a general healthcare information assistant. Answer briefly, "
    "in two sentences or fewer. Do not diagnose or prescribe."
)
TEST_QUESTION = "What are three simple habits that support good hydration?"


async def run(provider_override: str | None) -> int:
    """Make one live call and print the result, or a clear diagnostic on failure."""
    settings = get_settings()
    setup_logging(level=settings.log_level, log_format="console")
    logger = get_logger("smoke_test")

    if provider_override:
        # A copy, not a mutation -- get_settings() returns a cached, shared
        # singleton, and this override should apply only to this one run.
        settings = settings.model_copy(update={"llm_provider": provider_override})

    print("-" * 60)
    print(f"Provider under test : {settings.llm_provider}")
    print(f"Model               : {settings.active_model}")
    print(f"Question            : {TEST_QUESTION}")
    print("-" * 60)

    try:
        provider = get_llm(settings)
    except HealthAssistantError as exc:
        print(f"\n[FAIL] Could not construct provider: {exc.user_message}")
        return 1

    messages = [Message(role=Role.USER, content=TEST_QUESTION)]
    started = time.perf_counter()

    try:
        print("\nStreaming response:\n")
        chunks: list[str] = []
        async for chunk in provider.stream(messages, system_prompt=SYSTEM_PROMPT):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
    except HealthAssistantError as exc:
        logger.exception("Smoke test call failed")
        print(f"\n\n[FAIL] {exc.code}: {exc.user_message}")
        print(f"        Detail: {exc.detail}")
        return 1

    text = "".join(chunks)
    print("\n\n" + "-" * 60)
    if not text.strip():
        print("[FAIL] Provider returned an empty response.")
        return 1

    print(f"[ OK ] Received {len(text)} characters in {elapsed_ms} ms.")
    print("-" * 60 + "\n")
    return 0


def main() -> int:
    """Parse arguments and run the smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["gemini", "groq", "openrouter"],
        default=None,
        help="Override LLM_PROVIDER for this run only.",
    )
    args = parser.parse_args()
    return asyncio.run(run(args.provider))


if __name__ == "__main__":
    raise SystemExit(main())
