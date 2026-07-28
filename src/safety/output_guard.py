"""Post-model output validator — Layer 4 of 4 safety layers.

Design rationale
----------------
The output guard catches the model producing something unsafe in response to
a question that looked benign going in. This is the failure mode the input
guard cannot prevent by design: a user asks "what is diabetes?" (allowed) and
the model answers "you probably have type 2 diabetes" (not allowed).

**Why a separate output guard rather than trusting the prompt?**

The prompt contract (``formatting_contract()``, ``scope_and_refusals()``)
tells the model what to do. The output guard enforces it. These two things
are both necessary because large language models sometimes fail to follow
instructions — especially when a question is ambiguous or the model is
running on a weaker backend during failover. Having a post-generation check
means that instruction drift in the model is never silently served to the
user.

**Severity levels (defined on ValidationIssue):**

- ``severity=3`` — block the response entirely regardless of strict/lenient
  mode. Used for explicit diagnosis statements and numeric dosage
  recommendations: these are the hardest regulatory violations.
- ``severity=2`` — block in strict mode (``settings.safety_strict_mode=True``,
  the default), log-only in lenient mode. Used for softer prescription-style
  recommendations and definitive prognosis language.

The ``must_block`` property on ``OutputValidationResult`` already encodes
the severity-3 check; the service layer adds the strict-mode severity-2
check with a single ``if settings.safety_strict_mode`` branch.

**What is deliberately NOT checked here:**

- Whether the answer is factually correct — that is the RAG grounding layer's
  job (Phase 5).
- Tone or length — those are style issues, not safety issues.
- Repetition of the user's health question — already redacted in logs by the
  logging layer (Phase 1).
"""

from __future__ import annotations

import re

from src.models.safety import (
    OutputValidationResult,
    RiskCategory,
    ValidationIssue,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _r(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Severity-3 rules — block always
# ---------------------------------------------------------------------------

_SEVERITY_3_RULES: list[tuple[str, RiskCategory, re.Pattern[str]]] = [
    # Explicit diagnosis aimed at the user
    (
        "output.diagnosis.you_have",
        RiskCategory.DIAGNOSIS_REQUEST,
        _r(r"\byou\s+(have|likely\s+have|probably\s+have|may\s+have|might\s+have)\s+"
           r"(type\s*[12]?\s*)?"
           r"(diabetes|cancer|hypertension|depression|anxiety\s+disorder|"
           r"lupus|ms|als|hiv|covid|infection|condition|disease|disorder|syndrome)\b"),
    ),
    (
        "output.diagnosis.sounds_like",
        RiskCategory.DIAGNOSIS_REQUEST,
        _r(r"\b(this\s+sounds?\s+like|this\s+looks?\s+like|that\s+sounds?\s+like)\s+"
           r"(it\s+could\s+be\s+)?"
           r"(diabetes|cancer|hypertension|depression|anxiety\s+disorder|"
           r"a\s+heart\s+attack|appendicitis|pneumonia|infection)\b"),
    ),
    (
        "output.diagnosis.you_are_suffering",
        RiskCategory.DIAGNOSIS_REQUEST,
        _r(r"\byou\s+are\s+(suffering|experiencing|showing\s+signs)\s+"
           r"(from\s+)?(a\s+)?(\w+\s+)?"
           r"(disease|disorder|condition|syndrome|illness)\b"),
    ),
    # Specific numeric dosage recommendations
    (
        "output.medication.numeric_dosage",
        RiskCategory.MEDICATION_REQUEST,
        _r(r"\b(take|use|apply|inject|administer)\s+\d+\s*(mg|mcg|ml|units?|tablets?|capsules?|pills?|drops?)\b"),
    ),
    (
        "output.medication.times_per_day",
        RiskCategory.MEDICATION_REQUEST,
        _r(r"\b\d+\s*(mg|mcg|ml|units?|tablets?|capsules?|pills?)\s+"
           r"(once|twice|\d+\s+times?)\s+(a\s+day|daily|per\s+day)\b"),
    ),
    (
        "output.medication.dosage_ceiling",
        RiskCategory.MEDICATION_REQUEST,
        _r(r"\b(maximum|max|safe)\s+dose\s+(is|of)\s+\d+\s*(mg|mcg)\b"),
    ),
]

# ---------------------------------------------------------------------------
# Severity-2 rules — block in strict mode, log in lenient mode
# ---------------------------------------------------------------------------

_SEVERITY_2_RULES: list[tuple[str, RiskCategory, re.Pattern[str]]] = [
    # Prescription-style recommendations without numeric dose
    (
        "output.medication.i_recommend_drug",
        RiskCategory.MEDICATION_REQUEST,
        _r(r"\b(i\s+recommend|you\s+should\s+take|try\s+taking)"
           r"(\s+you\s+(take|use|try))?"
           r"\s+(ibuprofen|paracetamol|acetaminophen|aspirin|antibiotic|metformin|"
           r"lisinopril|atorvastatin|omeprazole|antihistamine)\b"),
    ),
    (
        "output.medication.switch_to",
        RiskCategory.MEDICATION_REQUEST,
        _r(r"\byou\s+should\s+(switch|change)\s+(to|from)\s+"
           r"(ibuprofen|paracetamol|aspirin|metformin|lisinopril|atorvastatin)\b"),
    ),
    # Definitive prognosis
    (
        "output.diagnosis.you_will_recover",
        RiskCategory.DIAGNOSIS_REQUEST,
        _r(r"\byou\s+will\s+(recover|be\s+(fine|better|cured|healed))\s+"
           r"(in\s+\d+\s+(days?|weeks?|months?)|soon|quickly)\b"),
    ),
    (
        "output.diagnosis.your_condition_is",
        RiskCategory.DIAGNOSIS_REQUEST,
        _r(r"\byour\s+(condition|illness|disease|disorder)\s+is\s+"
           r"(serious|severe|critical|life[- ]threatening|benign|mild)\b"),
    ),
]


# ---------------------------------------------------------------------------
# Public guard class
# ---------------------------------------------------------------------------


class OutputGuard:
    """Validate generated text before it is returned to the user.

    Usage::

        guard = OutputGuard()
        result = guard.validate(generated_text)
        if result.must_block:
            # substitute a safe fallback message
            ...

    The guard is stateless — instantiate it once at service startup.
    """

    def validate(self, text: str) -> OutputValidationResult:
        """Run all validation rules against *text*.

        All rules are evaluated (not short-circuited) so the result carries
        the complete set of issues, which the eval harness uses to measure
        per-rule failure rates.

        Args:
            text: The raw text string returned by the language model.

        Returns:
            :class:`~src.models.safety.OutputValidationResult` with zero or
            more :class:`~src.models.safety.ValidationIssue` entries.
        """
        issues: list[ValidationIssue] = []

        for rule_id, category, pattern in _SEVERITY_3_RULES:
            match = pattern.search(text)
            if match:
                issues.append(
                    ValidationIssue(
                        rule_id=rule_id,
                        category=category,
                        message=(
                            f"Severity-3 output violation: rule '{rule_id}' "
                            f"matched at position {match.start()}."
                        ),
                        severity=3,
                        excerpt=text[max(0, match.start() - 20): match.end() + 20],
                    )
                )

        for rule_id, category, pattern in _SEVERITY_2_RULES:
            match = pattern.search(text)
            if match:
                issues.append(
                    ValidationIssue(
                        rule_id=rule_id,
                        category=category,
                        message=(
                            f"Severity-2 output violation: rule '{rule_id}' "
                            f"matched at position {match.start()}."
                        ),
                        severity=2,
                        excerpt=text[max(0, match.start() - 20): match.end() + 20],
                    )
                )

        return OutputValidationResult(issues=issues)
