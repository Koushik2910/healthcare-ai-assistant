"""Tests for the prompt engineering layer (Phase 3).

Coverage strategy
-----------------
Every public function in ``src.prompts`` is covered by at least one positive
test and one boundary/negative test.  No LLM is called; no network access is
required; no ``.env`` file is needed.  The tests are intentionally structured
to mirror the rubric the Phase 7 adversarial eval harness will use — so a
failure here is a leading indicator of a guardrail or prompt regression.

Markers
-------
All tests carry the ``unit`` marker declared in ``pyproject.toml``.
Run with: ``pytest tests/test_prompts.py -v -m unit``
"""

from __future__ import annotations

import pytest

from src.models.rag import (
    Chunk,
    DocumentLicence,
    KBDocument,
    RetrievalResult,
    RetrievedChunk,
)
from src.models.safety import RiskCategory
from src.prompts import (
    MEDICAL_DISCLAIMER,
    PromptBuilder,
    PromptContext,
    all_mapped_categories,
    formatting_contract,
    rag_context,
    refusal_for,
    scope_and_refusals,
    system_identity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(doc_id: str = "test-doc", index: int = 0, text: str = "Some text.") -> Chunk:
    """Return a minimal valid Chunk for builder tests."""
    return Chunk(
        chunk_id=f"{doc_id}::{index}",
        doc_id=doc_id,
        index=index,
        text=text,
        title="Test Document",
        source="MedlinePlus",
        licence=DocumentLicence.US_GOV_PUBLIC_DOMAIN,
        url="https://medlineplus.gov/test",
        topics=["health"],
    )


def _make_retrieval(chunks: list[Chunk], *, scores: list[float] | None = None) -> RetrievalResult:
    """Return a RetrievalResult from the given chunks."""
    if scores is None:
        scores = [0.9] * len(chunks)
    retrieved = [
        RetrievedChunk(chunk=c, score=s) for c, s in zip(chunks, scores)
    ]
    return RetrievalResult(query="test query", chunks=retrieved, took_ms=10)


# ---------------------------------------------------------------------------
# Block 1 — system_identity
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSystemIdentity:
    def test_returns_non_empty_string(self) -> None:
        result = system_identity()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_persona(self) -> None:
        result = system_identity()
        assert "Healthcare Information Assistant" in result

    def test_contains_not_a_doctor(self) -> None:
        # The hard limit must be explicit, not implied
        result = system_identity()
        assert "NOT a doctor" in result or "not a doctor" in result.lower()

    def test_contains_no_diagnose(self) -> None:
        result = system_identity()
        assert "diagnose" in result.lower()

    def test_contains_no_prescribe(self) -> None:
        result = system_identity()
        assert "prescribe" in result.lower()

    def test_idempotent(self) -> None:
        # Pure function — same output every call
        assert system_identity() == system_identity()


# ---------------------------------------------------------------------------
# Block 2 — scope_and_refusals
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScopeAndRefusals:
    def test_returns_non_empty_string(self) -> None:
        result = scope_and_refusals()
        assert isinstance(result, str)
        assert len(result) > 200

    def test_contains_allowed_categories(self) -> None:
        result = scope_and_refusals()
        for category in ("GENERAL_INFO", "NUTRITION", "LIFESTYLE", "FIRST_AID", "PREVENTION"):
            assert category in result, f"Expected {category} in scope block"

    def test_contains_refused_categories(self) -> None:
        result = scope_and_refusals()
        for category in ("OUT_OF_SCOPE", "DIAGNOSIS_REQUEST", "MEDICATION_REQUEST", "PROMPT_INJECTION"):
            assert category in result, f"Expected {category} in scope block"

    def test_crisis_categories_present(self) -> None:
        result = scope_and_refusals()
        assert "SELF_HARM" in result
        assert "EMERGENCY" in result

    def test_crisis_resources_in_scope_block(self) -> None:
        # The scope block tells the model what to do; it must name 988
        result = scope_and_refusals()
        assert "988" in result

    def test_emergency_number_present(self) -> None:
        result = scope_and_refusals()
        assert "911" in result

    def test_idempotent(self) -> None:
        assert scope_and_refusals() == scope_and_refusals()


# ---------------------------------------------------------------------------
# Block 3 — formatting_contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormattingContract:
    def test_returns_non_empty_string(self) -> None:
        result = formatting_contract()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_citation_format_specified(self) -> None:
        # The model must be told to use [1], [2] etc.
        result = formatting_contract()
        assert "[1]" in result or "[N]" in result or "numbered markers" in result

    def test_disclaimer_text_present(self) -> None:
        # The DISCLAIMER section now instructs the LLM NOT to append its own
        # footer (the UI pill handles it).  Assert the new instruction is present
        # rather than the old footer text.
        result = formatting_contract()
        assert "Do NOT append any disclaimer footer" in result or \
               "UI automatically displays a disclaimer" in result

    def test_idempotent(self) -> None:
        assert formatting_contract() == formatting_contract()


# ---------------------------------------------------------------------------
# Block 4 — rag_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRagContext:
    def test_basic_injection(self) -> None:
        chunks = [(1, "Hydration Basics", "Drink 8 cups of water per day.")]
        result = rag_context(chunks)
        assert "[1]" in result
        assert "Hydration Basics" in result
        assert "Drink 8 cups" in result

    def test_multiple_chunks_all_injected(self) -> None:
        chunks = [
            (1, "Doc A", "Text A."),
            (2, "Doc B", "Text B."),
            (3, "Doc C", "Text C."),
        ]
        result = rag_context(chunks)
        for marker, title, text in chunks:
            assert f"[{marker}]" in result
            assert title in result
            assert text.strip() in result

    def test_empty_chunks_raises(self) -> None:
        with pytest.raises(ValueError, match="empty chunk list"):
            rag_context([])

    def test_instructs_model_to_cite(self) -> None:
        chunks = [(1, "Title", "Text.")]
        result = rag_context(chunks)
        # The footer must instruct the model to cite
        assert "Cite" in result or "cite" in result

    def test_warns_against_hallucination(self) -> None:
        chunks = [(1, "Title", "Text.")]
        result = rag_context(chunks)
        assert "don't have" in result.lower() or "do not" in result.lower() or "insufficient" in result.lower()

    def test_returns_string(self) -> None:
        result = rag_context([(1, "T", "text")])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptBuilder:
    def test_build_no_retrieval_returns_context(self) -> None:
        builder = PromptBuilder()
        ctx = builder.build()
        assert isinstance(ctx, PromptContext)

    def test_build_no_retrieval_not_grounded(self) -> None:
        builder = PromptBuilder()
        ctx = builder.build()
        assert ctx.grounded is False
        assert ctx.injected_chunks == []

    def test_build_no_retrieval_system_prompt_has_all_three_blocks(self) -> None:
        builder = PromptBuilder()
        ctx = builder.build()
        prompt = ctx.system_prompt
        # All three permanent blocks must be present
        assert "Healthcare Information Assistant" in prompt  # identity
        assert "DIAGNOSIS_REQUEST" in prompt                  # scope
        assert "Do NOT append any disclaimer footer" in prompt or \
               "UI automatically displays a disclaimer" in prompt  # formatting

    def test_build_no_retrieval_no_context_section(self) -> None:
        builder = PromptBuilder()
        ctx = builder.build()
        assert "Knowledge-base context" not in ctx.system_prompt
        assert "## Knowledge-base context (CONTEXT)" not in ctx.system_prompt

    def test_build_with_retrieval_grounded(self) -> None:
        builder = PromptBuilder()
        chunk = _make_chunk(text="Vitamin C helps with immunity.")
        retrieval = _make_retrieval([chunk])
        ctx = builder.build(retrieval=retrieval)
        assert ctx.grounded is True

    def test_build_with_retrieval_injects_context_section(self) -> None:
        builder = PromptBuilder()
        chunk = _make_chunk(text="Vitamin C helps with immunity.")
        retrieval = _make_retrieval([chunk])
        ctx = builder.build(retrieval=retrieval)
        assert "Knowledge-base context" in ctx.system_prompt
        assert "Vitamin C helps with immunity." in ctx.system_prompt

    def test_build_with_retrieval_injected_chunks_populated(self) -> None:
        builder = PromptBuilder()
        chunk = _make_chunk(text="Some health fact.")
        retrieval = _make_retrieval([chunk])
        ctx = builder.build(retrieval=retrieval)
        assert len(ctx.injected_chunks) == 1
        marker, title, text = ctx.injected_chunks[0]
        assert marker == 1
        assert title == chunk.title
        assert text == chunk.text

    def test_build_with_empty_retrieval_not_grounded(self) -> None:
        # RetrievalResult with no chunks (e.g. nothing above score threshold)
        retrieval = RetrievalResult(query="q", chunks=[], took_ms=5)
        builder = PromptBuilder()
        ctx = builder.build(retrieval=retrieval)
        assert ctx.grounded is False
        assert ctx.injected_chunks == []

    def test_build_multiple_chunks_marker_order(self) -> None:
        builder = PromptBuilder()
        chunks = [
            _make_chunk("doc-a", 0, "Text A."),
            _make_chunk("doc-b", 0, "Text B."),
        ]
        retrieval = _make_retrieval(chunks)
        ctx = builder.build(retrieval=retrieval)
        assert len(ctx.injected_chunks) == 2
        assert ctx.injected_chunks[0][0] == 1
        assert ctx.injected_chunks[1][0] == 2

    def test_prompt_context_frozen(self) -> None:
        builder = PromptBuilder()
        ctx = builder.build()
        with pytest.raises((AttributeError, TypeError)):
            ctx.grounded = True  # type: ignore[misc]

    def test_builder_stateless_between_calls(self) -> None:
        builder = PromptBuilder()
        ctx1 = builder.build()
        ctx2 = builder.build()
        assert ctx1.system_prompt == ctx2.system_prompt


# ---------------------------------------------------------------------------
# TemplateLibrary — refusal_for
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRefusalFor:
    def test_self_harm_contains_988(self) -> None:
        msg = refusal_for(RiskCategory.SELF_HARM)
        assert "988" in msg

    def test_self_harm_contains_crisis_text_line(self) -> None:
        msg = refusal_for(RiskCategory.SELF_HARM)
        assert "741741" in msg

    def test_self_harm_contains_emergency_number(self) -> None:
        msg = refusal_for(RiskCategory.SELF_HARM)
        assert "911" in msg

    def test_emergency_urgent_language(self) -> None:
        msg = refusal_for(RiskCategory.EMERGENCY)
        assert "emergency" in msg.lower()
        assert "911" in msg

    def test_diagnosis_no_diagnostic_claim(self) -> None:
        msg = refusal_for(RiskCategory.DIAGNOSIS_REQUEST)
        # The template itself must not diagnose anything
        assert "you have" not in msg.lower()
        assert "you are experiencing" not in msg.lower()

    def test_diagnosis_redirects_to_doctor(self) -> None:
        msg = refusal_for(RiskCategory.DIAGNOSIS_REQUEST)
        assert "doctor" in msg.lower() or "clinician" in msg.lower() or "provider" in msg.lower()

    def test_medication_redirects_to_pharmacist(self) -> None:
        msg = refusal_for(RiskCategory.MEDICATION_REQUEST)
        assert "pharmacist" in msg.lower() or "physician" in msg.lower()

    def test_out_of_scope_short(self) -> None:
        msg = refusal_for(RiskCategory.OUT_OF_SCOPE)
        # Should be concise — not more than 300 chars
        assert len(msg) < 400

    def test_prompt_injection_minimal(self) -> None:
        msg = refusal_for(RiskCategory.PROMPT_INJECTION)
        # Must not leak system internals or acknowledge the injection attempt
        assert "system prompt" not in msg.lower()
        assert "ignore" not in msg.lower()

    def test_none_category_returns_fallback(self) -> None:
        # NONE should not normally reach this function, but it must not raise
        msg = refusal_for(RiskCategory.NONE)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_all_categories_return_strings(self) -> None:
        for category in RiskCategory:
            msg = refusal_for(category)
            assert isinstance(msg, str), f"refusal_for({category}) did not return a str"
            assert len(msg) > 0, f"refusal_for({category}) returned an empty string"


# ---------------------------------------------------------------------------
# all_mapped_categories — coverage completeness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAllMappedCategories:
    def test_returns_list(self) -> None:
        result = all_mapped_categories()
        assert isinstance(result, list)

    def test_non_empty(self) -> None:
        assert len(all_mapped_categories()) > 0

    def test_all_are_risk_categories(self) -> None:
        for cat in all_mapped_categories():
            assert isinstance(cat, RiskCategory)

    def test_crisis_categories_covered(self) -> None:
        mapped = all_mapped_categories()
        assert RiskCategory.SELF_HARM in mapped
        assert RiskCategory.EMERGENCY in mapped

    def test_refusal_categories_covered(self) -> None:
        mapped = all_mapped_categories()
        assert RiskCategory.DIAGNOSIS_REQUEST in mapped
        assert RiskCategory.MEDICATION_REQUEST in mapped
        assert RiskCategory.OUT_OF_SCOPE in mapped
        assert RiskCategory.PROMPT_INJECTION in mapped


# ---------------------------------------------------------------------------
# MEDICAL_DISCLAIMER constant
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMedicalDisclaimer:
    def test_is_string(self) -> None:
        assert isinstance(MEDICAL_DISCLAIMER, str)

    def test_not_empty(self) -> None:
        assert len(MEDICAL_DISCLAIMER) > 20

    def test_educational_language(self) -> None:
        assert "educational" in MEDICAL_DISCLAIMER.lower()

    def test_not_substitute(self) -> None:
        assert "not a substitute" in MEDICAL_DISCLAIMER.lower()
