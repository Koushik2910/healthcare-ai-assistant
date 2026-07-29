"""Adversarial eval harness — hermetic pytest edition.

Runs the shared eval cases against a fake LLM so the suite stays
network-free and deterministic.  These tests prove that the *pipeline
routing logic* (guard → LLM decision → source assignment) is correct for
every category, without requiring a real API key or a populated ChromaDB.

What these tests cover (and don't cover)
-----------------------------------------
COVER:
  - InputGuard fires on the right inputs (CRISIS, EMERGENCY, OUT_OF_SCOPE,
    PROMPT_INJECTION cases) and sets the correct ResponseSource / refused flag.
  - Benign inputs are NOT blocked (no false positives from the guard).
  - The pipeline assembles a ChatResponse for every case without raising.

DON'T COVER (tested by ``scripts/eval.py`` against the live pipeline):
  - Whether the LLM actually produces medically accurate content.
  - Whether ChromaDB retrieves relevant chunks (source=GROUNDED path).
  - Latency.

GROUNDED cases in this harness
--------------------------------
The fake LLM returns a fixed benign string.  Without a real ChromaDB the
retriever slot is None, so ChatService sets source=MODEL_ONLY — not GROUNDED.
Rather than skipping these cases or wiring a fake ChromaDB, we assert
``expect_refused=False`` and ``source ∈ {grounded, model_only}`` — both
indicate the pipeline processed the input normally without a false refusal.
This is the right trade-off: the grounded-vs-ungrounded distinction belongs
to the live eval, not the unit harness.

Markers: ``unit``, ``eval`` (both declared in pyproject.toml).
"""

from __future__ import annotations

import pytest

from src.config.settings import Settings
from src.eval.cases import EVAL_CASES, EvalCase
from src.models.chat import Conversation, ResponseSource
from src.services.chat_service import ChatService
from tests.fakes import FakeLLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAFE_RESPONSE = (
    "Staying well-hydrated is important for overall health. "
    "Aim for about 8 cups of water per day, though individual needs vary."
)


def _service() -> ChatService:
    """ChatService with a safe fake LLM, no retriever, strict mode."""
    primary = FakeLLMProvider(chunks=[_SAFE_RESPONSE])
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key="test-key-not-real",  # type: ignore[arg-type]
        safety_strict_mode=True,
        max_input_chars=2000,
        max_history_turns=8,
    )
    svc = ChatService(primary=primary, settings=settings)
    svc._fallback = None  # disable auto-fallback — keeps tests deterministic
    return svc


# ---------------------------------------------------------------------------
# Parametrised guardrail tests (CRISIS, EMERGENCY, OUT_OF_SCOPE, INJECTION)
# ---------------------------------------------------------------------------

_GUARDRAIL_CASES = [c for c in EVAL_CASES if c.expect_refused]
_GUARDRAIL_IDS = [c.id for c in _GUARDRAIL_CASES]


@pytest.mark.unit
@pytest.mark.eval
@pytest.mark.asyncio
class TestGuardrailCases:
    """Every case that expects a guardrail/escalation response."""

    @pytest.mark.parametrize("case", _GUARDRAIL_CASES, ids=_GUARDRAIL_IDS)
    async def test_refused(self, case: EvalCase) -> None:
        """Pipeline must set refused=True for this input."""
        svc = _service()
        resp = await svc.chat(case.input, Conversation())
        assert resp.refused is True, (
            f"[{case.id}] Expected refused=True but got False. "
            f"source={resp.source.value!r}, content={resp.message.content[:80]!r}"
        )

    @pytest.mark.parametrize("case", _GUARDRAIL_CASES, ids=_GUARDRAIL_IDS)
    async def test_source(self, case: EvalCase) -> None:
        """Pipeline must set the expected ResponseSource."""
        svc = _service()
        resp = await svc.chat(case.input, Conversation())
        assert resp.source.value == case.expect_source, (
            f"[{case.id}] source={resp.source.value!r} != {case.expect_source!r}"
        )

    @pytest.mark.parametrize("case", _GUARDRAIL_CASES, ids=_GUARDRAIL_IDS)
    async def test_llm_not_called(self, case: EvalCase) -> None:
        """InputGuard must block the LLM call for refusal cases."""
        primary = FakeLLMProvider(chunks=[_SAFE_RESPONSE])
        settings = Settings(
            llm_provider="gemini",
            gemini_api_key="test-key-not-real",  # type: ignore[arg-type]
        )
        svc = ChatService(primary=primary, settings=settings)
        svc._fallback = None
        await svc.chat(case.input, Conversation())
        assert primary.call_count == 0, (
            f"[{case.id}] LLM was called ({primary.call_count} times) "
            "but should have been blocked by the input guard."
        )


# ---------------------------------------------------------------------------
# No false-positive tests (GROUNDED + BENIGN)
# ---------------------------------------------------------------------------

_PASS_THROUGH_CASES = [c for c in EVAL_CASES if not c.expect_refused]
_PASS_THROUGH_IDS = [c.id for c in _PASS_THROUGH_CASES]


@pytest.mark.unit
@pytest.mark.eval
@pytest.mark.asyncio
class TestNoFalsePositives:
    """Benign and grounded inputs must NOT be refused."""

    @pytest.mark.parametrize("case", _PASS_THROUGH_CASES, ids=_PASS_THROUGH_IDS)
    async def test_not_refused(self, case: EvalCase) -> None:
        """The pipeline must answer this question, not refuse it."""
        svc = _service()
        resp = await svc.chat(case.input, Conversation())
        assert resp.refused is False, (
            f"[{case.id}] False positive — benign question was refused. "
            f"source={resp.source.value!r}, content={resp.message.content[:80]!r}"
        )

    @pytest.mark.parametrize("case", _PASS_THROUGH_CASES, ids=_PASS_THROUGH_IDS)
    async def test_source_is_not_escalation(self, case: EvalCase) -> None:
        """Benign questions must never be escalated."""
        svc = _service()
        resp = await svc.chat(case.input, Conversation())
        assert resp.source != ResponseSource.ESCALATION, (
            f"[{case.id}] Benign input was escalated as a crisis."
        )

    @pytest.mark.parametrize("case", _PASS_THROUGH_CASES, ids=_PASS_THROUGH_IDS)
    async def test_llm_was_called(self, case: EvalCase) -> None:
        """Benign inputs must reach the LLM (guard must not block them)."""
        primary = FakeLLMProvider(chunks=[_SAFE_RESPONSE])
        settings = Settings(
            llm_provider="gemini",
            gemini_api_key="test-key-not-real",  # type: ignore[arg-type]
        )
        svc = ChatService(primary=primary, settings=settings)
        svc._fallback = None
        await svc.chat(case.input, Conversation())
        assert primary.call_count == 1, (
            f"[{case.id}] LLM call_count={primary.call_count} "
            "(benign input should have reached the LLM exactly once)."
        )

    @pytest.mark.parametrize("case", _PASS_THROUGH_CASES, ids=_PASS_THROUGH_IDS)
    async def test_response_has_content(self, case: EvalCase) -> None:
        """Response must contain non-empty text."""
        svc = _service()
        resp = await svc.chat(case.input, Conversation())
        assert resp.message.content.strip(), (
            f"[{case.id}] Response content is empty."
        )


# ---------------------------------------------------------------------------
# Crisis keyword assertions (content of the escalation response)
# ---------------------------------------------------------------------------

_CRISIS_CASES = [c for c in EVAL_CASES if c.category == "CRISIS"]
_CRISIS_IDS = [c.id for c in _CRISIS_CASES]

_EMERGENCY_CASES = [c for c in EVAL_CASES if c.category == "EMERGENCY"]
_EMERGENCY_IDS = [c.id for c in _EMERGENCY_CASES]


@pytest.mark.unit
@pytest.mark.eval
@pytest.mark.asyncio
class TestCrisisContent:
    """Crisis/emergency responses must contain appropriate hotline references."""

    @pytest.mark.parametrize("case", _CRISIS_CASES, ids=_CRISIS_IDS)
    async def test_crisis_response_contains_988(self, case: EvalCase) -> None:
        """Self-harm escalation must mention 988."""
        svc = _service()
        resp = await svc.chat(case.input, Conversation())
        assert "988" in resp.message.content, (
            f"[{case.id}] Crisis response missing '988': {resp.message.content[:120]!r}"
        )

    @pytest.mark.parametrize("case", _EMERGENCY_CASES, ids=_EMERGENCY_IDS)
    async def test_emergency_response_contains_911(self, case: EvalCase) -> None:
        """Medical emergency escalation must mention 911."""
        svc = _service()
        resp = await svc.chat(case.input, Conversation())
        assert "911" in resp.message.content, (
            f"[{case.id}] Emergency response missing '911': {resp.message.content[:120]!r}"
        )
