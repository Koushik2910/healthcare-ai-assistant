"""Safety domain models.

Design rationale
----------------
The central decision encoded here is that **a refusal is a normal outcome,
not an error**. Guardrails return a :class:`SafetyVerdict`; they do not raise.
Three consequences follow, and all three are the point:

1. Every decision the safety layer makes is a first-class value that can be
   asserted on in a test, which is what turns "we have guardrails" into a
   measurable pass rate in the evaluation harness.
2. Refusals carry a *reason* and a *category*, so the UI can respond
   proportionately -- a scope refusal is a gentle redirect, a crisis
   detection is a prominent escalation card.
3. Nothing in the pipeline can accidentally swallow a refusal in a broad
   ``except`` block, because it never travelled as an exception.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RiskCategory(str, Enum):
    """Taxonomy of risks the guardrail layer recognises.

    Ordered roughly from most to least severe. The taxonomy is explicit
    rather than free-text so that evaluation results can be reported per
    category and regressions localised.
    """

    #: Self-harm or suicidal ideation. Highest priority; bypasses the model.
    SELF_HARM = "self_harm"
    #: Medical emergency in progress (chest pain, stroke signs, anaphylaxis).
    EMERGENCY = "emergency"
    #: Request for a diagnosis of the user's specific condition.
    DIAGNOSIS_REQUEST = "diagnosis_request"
    #: Request for a prescription, drug choice or dosage.
    MEDICATION_REQUEST = "medication_request"
    #: Instructions that could cause harm if followed.
    DANGEROUS_INSTRUCTION = "dangerous_instruction"
    #: Attempt to override the system prompt or extract it.
    PROMPT_INJECTION = "prompt_injection"
    #: Legitimate question, but outside the healthcare remit.
    OUT_OF_SCOPE = "out_of_scope"
    #: Nothing of concern detected.
    NONE = "none"


class SafetyAction(str, Enum):
    """What the pipeline should do with a message after screening."""

    #: Proceed to the model unchanged.
    ALLOW = "allow"
    #: Proceed, but inject additional constraints into the prompt.
    ALLOW_WITH_CONSTRAINTS = "allow_with_constraints"
    #: Do not call the model; return a templated refusal.
    REFUSE = "refuse"
    #: Do not call the model; return crisis or emergency resources.
    ESCALATE = "escalate"


class SafetyVerdict(BaseModel):
    """The outcome of screening a single message.

    Attributes:
        action: What the pipeline should do next.
        category: Which risk, if any, triggered the decision.
        reason: Engineer-facing explanation, logged and asserted in tests.
        confidence: Detector confidence. Deterministic pattern matches report
            ``1.0``; model-assisted classification reports its own estimate.
        matched_rules: Identifiers of the rules that fired, so a false
            positive can be traced to a specific pattern instead of a vague
            "the classifier did it".
        constraints: Extra instructions injected into the system prompt when
            the action is :attr:`SafetyAction.ALLOW_WITH_CONSTRAINTS`.
    """

    model_config = ConfigDict(frozen=True)

    action: SafetyAction = SafetyAction.ALLOW
    category: RiskCategory = RiskCategory.NONE
    reason: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    matched_rules: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)

    @property
    def blocks_model_call(self) -> bool:
        """True when no request should be sent to the language model."""
        return self.action in (SafetyAction.REFUSE, SafetyAction.ESCALATE)

    @classmethod
    def allow(cls) -> "SafetyVerdict":
        """Return the default permissive verdict."""
        return cls()


class ValidationIssue(BaseModel):
    """A problem found in generated output by the post-generation validators.

    Output validation is separate from input screening because the two fail
    differently: input screening prevents a bad request reaching the model,
    while output validation catches the model producing something unsafe in
    response to a request that looked perfectly benign.
    """

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(description="Stable identifier of the validator rule.")
    category: RiskCategory
    message: str = Field(description="Engineer-facing description of the problem.")
    severity: int = Field(
        default=1,
        ge=1,
        le=3,
        description="1 warn, 2 rewrite required, 3 block the response entirely.",
    )
    excerpt: str = Field(
        default="", description="The offending fragment, for debugging."
    )


class OutputValidationResult(BaseModel):
    """Aggregate result of running every output validator over one response."""

    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no issue was found."""
        return not self.issues

    @property
    def must_block(self) -> bool:
        """True when at least one issue is severe enough to withhold output."""
        return any(issue.severity >= 3 for issue in self.issues)

    @property
    def max_severity(self) -> int:
        """Highest severity observed, or ``0`` when clean."""
        return max((issue.severity for issue in self.issues), default=0)
