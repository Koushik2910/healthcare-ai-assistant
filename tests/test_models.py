"""Domain model tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.chat import (
    ChatResponse,
    Citation,
    Conversation,
    Message,
    ResponseSource,
    Role,
)
from src.models.rag import (
    Chunk,
    DocumentLicence,
    RetrievalResult,
    RetrievedChunk,
)
from src.models.safety import (
    OutputValidationResult,
    RiskCategory,
    SafetyAction,
    SafetyVerdict,
    ValidationIssue,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #


def test_provider_dict_carries_only_role_and_content() -> None:
    """Ids, citations and metadata must never reach the model as tokens."""
    message = Message(
        role=Role.USER,
        content="What is a balanced diet?",
        metadata={"latency_ms": 42},
    )

    assert message.to_provider_dict() == {
        "role": "user",
        "content": "What is a balanced diet?",
    }


def test_conversation_add_updates_timestamp() -> None:
    conversation = Conversation()
    before = conversation.updated_at

    conversation.add(Message(role=Role.USER, content="hello"))

    assert len(conversation.messages) == 1
    assert conversation.updated_at >= before


def test_recent_returns_trailing_window_of_pairs() -> None:
    conversation = Conversation()
    for index in range(10):
        conversation.add(Message(role=Role.USER, content=f"q{index}"))
        conversation.add(Message(role=Role.ASSISTANT, content=f"a{index}"))

    recent = conversation.recent(max_turns=2)

    assert len(recent) == 4
    assert recent[0].content == "q8"
    assert recent[-1].content == "a9"


def test_title_derived_from_first_user_message() -> None:
    conversation = Conversation()
    conversation.add(Message(role=Role.SYSTEM, content="system preamble"))
    conversation.add(Message(role=Role.USER, content="How much water should I drink?"))

    assert conversation.derive_title() == "How much water should I drink?"


def test_long_title_is_truncated_with_ellipsis() -> None:
    conversation = Conversation()
    conversation.add(Message(role=Role.USER, content="word " * 40))

    title = conversation.derive_title(max_length=20)

    assert len(title) == 20
    assert title.endswith("\u2026")


def test_title_falls_back_when_no_user_message() -> None:
    assert Conversation().derive_title() == "New chat"


def test_is_grounded_requires_both_source_and_citations() -> None:
    citation = Citation(marker=1, title="Hydration", source="MedlinePlus", score=0.8)
    message = Message(role=Role.ASSISTANT, content="Drink water.")

    grounded = ChatResponse(
        message=message, source=ResponseSource.GROUNDED, citations=[citation]
    )
    unsupported = ChatResponse(
        message=message, source=ResponseSource.GROUNDED, citations=[]
    )
    ungrounded = ChatResponse(message=message, source=ResponseSource.MODEL_ONLY)

    assert grounded.is_grounded is True
    assert unsupported.is_grounded is False
    assert ungrounded.is_grounded is False


def test_citation_is_immutable() -> None:
    citation = Citation(marker=1, title="Hydration", source="MedlinePlus")

    with pytest.raises(ValidationError):
        citation.title = "Tampered"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #


def test_default_verdict_allows_and_does_not_block() -> None:
    verdict = SafetyVerdict.allow()

    assert verdict.action is SafetyAction.ALLOW
    assert verdict.category is RiskCategory.NONE
    assert verdict.blocks_model_call is False


@pytest.mark.parametrize(
    ("action", "blocks"),
    [
        (SafetyAction.ALLOW, False),
        (SafetyAction.ALLOW_WITH_CONSTRAINTS, False),
        (SafetyAction.REFUSE, True),
        (SafetyAction.ESCALATE, True),
    ],
)
def test_only_refusal_and_escalation_block_the_model_call(
    action: SafetyAction, blocks: bool
) -> None:
    assert SafetyVerdict(action=action).blocks_model_call is blocks


def test_validation_result_reports_pass_and_block_thresholds() -> None:
    clean = OutputValidationResult()
    assert clean.passed is True
    assert clean.must_block is False
    assert clean.max_severity == 0

    warned = OutputValidationResult(
        issues=[
            ValidationIssue(
                rule_id="hedging_missing",
                category=RiskCategory.NONE,
                message="No disclaimer present",
                severity=1,
            )
        ]
    )
    assert warned.passed is False
    assert warned.must_block is False

    blocked = OutputValidationResult(
        issues=[
            ValidationIssue(
                rule_id="dosage_stated",
                category=RiskCategory.MEDICATION_REQUEST,
                message="Response contains a numeric dosage",
                severity=3,
            )
        ]
    )
    assert blocked.must_block is True
    assert blocked.max_severity == 3


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def _chunk(index: int = 0, topics: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"hydration::{index}",
        doc_id="hydration",
        index=index,
        text="Adults generally need regular fluid intake through the day.",
        title="Hydration basics",
        source="MedlinePlus",
        licence=DocumentLicence.US_GOV_PUBLIC_DOMAIN,
        topics=topics or ["nutrition", "hydration"],
    )


def test_chunk_metadata_is_chroma_compatible() -> None:
    """Chroma accepts scalars only, so lists must be flattened at the boundary."""
    metadata = _chunk().to_metadata()

    assert metadata["topics"] == "nutrition|hydration"
    assert all(isinstance(value, (str, int)) for value in metadata.values())


def test_meets_threshold_is_inclusive() -> None:
    result = RetrievedChunk(chunk=_chunk(), score=0.25)

    assert result.meets(0.25) is True
    assert result.meets(0.26) is False


def test_top_returns_highest_scoring_chunks_in_order() -> None:
    result = RetrievalResult(
        query="hydration",
        chunks=[
            RetrievedChunk(chunk=_chunk(0), score=0.4),
            RetrievedChunk(chunk=_chunk(1), score=0.9),
            RetrievedChunk(chunk=_chunk(2), score=0.6),
        ],
    )

    scores = [item.score for item in result.top(2)]

    assert scores == [0.9, 0.6]


def test_empty_retrieval_reports_no_context() -> None:
    assert RetrievalResult(query="anything").has_context is False


def test_score_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievedChunk(chunk=_chunk(), score=1.5)
