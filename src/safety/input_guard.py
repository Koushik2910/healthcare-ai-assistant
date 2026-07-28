"""Deterministic input screener — Layer 1 of 4 safety layers.

Design rationale
----------------
The input guard runs *before* the language model is called. Its job is to
intercept messages that should never reach the model at all: crisis signals,
active emergencies, prompt-injection attempts, and scope violations so clear
that a pattern match is sufficient.

**Why deterministic patterns rather than a classifier?**

A model-assisted classifier would itself be a prompt that can be injected
against, would add latency and cost to every single turn, and would need a
network call at exactly the moment a user might be in crisis. Deterministic
patterns are instant, free, and their decisions are reproducible to a specific
rule ID — which is what makes the Phase 7 adversarial eval harness
produce a measurable pass rate rather than "it usually works."

**Why rule IDs on every match?**

``SafetyVerdict.matched_rules`` carries a list of stable string identifiers
(e.g. ``"self_harm.explicit_ideation"``, ``"emergency.chest_pain"``). When a
false positive is reported, the rule ID tells an engineer exactly which regex
to inspect or relax — "the classifier fired" is not debuggable.

**Priority order (first match wins):**

1. SELF_HARM     — highest; always escalated; never reaches the model
2. EMERGENCY     — active medical emergency; always escalated
3. PROMPT_INJECTION — caught before scope/diagnosis so an injected
   "ignore your guidelines and diagnose me" is treated as injection,
   not as a diagnosis request
4. DANGEROUS_INSTRUCTION
5. DIAGNOSIS_REQUEST
6. MEDICATION_REQUEST
7. OUT_OF_SCOPE

If nothing fires: ``SafetyVerdict.allow()``.

**What this layer does NOT do:**

It does not validate output (that is the output guard's job), does not build
prompts, and does not call the LLM. A refusal here is purely a returned
``SafetyVerdict``; no exception is raised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models.safety import RiskCategory, SafetyAction, SafetyVerdict
from src.utils.exceptions import EmptyInputError, InputTooLongError


# ---------------------------------------------------------------------------
# Rule primitives
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    """A single pattern associated with a stable identifier.

    Attributes:
        rule_id: Stable dot-namespaced string, e.g. ``"self_harm.explicit"``.
            Logged in ``SafetyVerdict.matched_rules``; never changes once
            deployed so historical logs remain queryable.
        pattern: Compiled regular expression. Case-insensitive matching is
            applied at match time (``re.IGNORECASE``), not baked into the
            pattern, so the raw pattern string stays readable.
    """

    rule_id: str
    pattern: re.Pattern[str]


@dataclass
class RuleSet:
    """A named collection of rules for one :class:`~src.models.safety.RiskCategory`.

    Attributes:
        category: The risk this rule set covers.
        action: What the pipeline should do when any rule in this set fires.
        rules: Ordered list of :class:`Rule` objects. All are evaluated; all
            matches are collected into ``SafetyVerdict.matched_rules``.
    """

    category: RiskCategory
    action: SafetyAction
    rules: list[Rule] = field(default_factory=list)

    def matches(self, text: str) -> list[str]:
        """Return rule IDs for every rule that matches *text*.

        Args:
            text: The normalised user input to test.

        Returns:
            List of matched rule IDs, or an empty list when nothing fires.
        """
        return [
            rule.rule_id
            for rule in self.rules
            if rule.pattern.search(text)
        ]


# ---------------------------------------------------------------------------
# Helper: compile a pattern and give it a namespaced rule ID
# ---------------------------------------------------------------------------

def _r(name: str, pattern: str) -> Rule:
    """Compile *pattern* into a :class:`Rule` with the given *name*."""
    return Rule(rule_id=name, pattern=re.compile(pattern, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Rule sets — one per actionable RiskCategory
# ---------------------------------------------------------------------------

_SELF_HARM_RULES = RuleSet(
    category=RiskCategory.SELF_HARM,
    action=SafetyAction.ESCALATE,
    rules=[
        _r("self_harm.want_to_die",
           r"\bwant\s+to\s+die\b"),
        _r("self_harm.kill_myself",
           r"\bkill\s+my\s*self\b"),
        _r("self_harm.end_my_life",
           r"\b(end|ending)\s+my\s+life\b"),
        _r("self_harm.thinking_about_ending",
           r"\b(thinking|thoughts?|thinking\s+about)\s+(about\s+)?(ending|taking)\s+my\s+(life|own\s+life)\b"),
        _r("self_harm.suicide",
           r"\bsuicid(e|al|ally)\b"),
        _r("self_harm.self_harm_act",
           r"\bself[- ]?harm\w*\b"),
        _r("self_harm.cutting",
           r"\bcutting\s+my\s*self\b|\bi\s+(cut|cuts|hurt)\s+my\s*self\b"),
        _r("self_harm.overdose_intent",
           r"\boverdos(e|ing)\s+(on|myself|pills)\b"),
        _r("self_harm.no_reason_to_live",
           r"\bno\s+reason\s+to\s+live\b"),
        _r("self_harm.hopeless",
           r"\b(nothing|no\s+one)\s+(can\s+help|cares|is\s+worth\s+it)\b"),
    ],
)

_EMERGENCY_RULES = RuleSet(
    category=RiskCategory.EMERGENCY,
    action=SafetyAction.ESCALATE,
    rules=[
        _r("emergency.chest_pain",
           r"\bchest\s+pain\b|\bchest\s+tightness\b|\bpain\s+in\s+my\s+chest\b"),
        _r("emergency.heart_attack",
           r"\bheart\s+attack\b|\bmyocardial\s+infarction\b"),
        _r("emergency.cant_breathe",
           r"\bcan'?t\s+breathe\b|\bcannot\s+breathe\b|\bno\s+air\b|\bnot\s+breathing\b"),
        _r("emergency.stroke_symptoms",
           r"\bstroke\b.*\b(symptoms?|signs?|having)\b"
           r"|\b(face\s+drooping|arm\s+weakness|speech\s+difficulty)\b"),
        _r("emergency.unconscious",
           r"\bunconscious\b|\bpassed\s+out\b|\bnot\s+responding\b|\bwon'?t\s+wake\b"),
        _r("emergency.severe_bleeding",
           r"\bsever(e|ely)\s+bleeding\b|\bblood\s+everywhere\b|\bcan'?t\s+stop\s+(the\s+)?bleeding\b"),
        _r("emergency.anaphylaxis",
           r"\banaphyla(xis|ctic)\b|\bthroat\s+closing\b|\bswelling\s+throat\b"),
        _r("emergency.seizure_active",
           r"\bhaving\s+a\s+seizure\b|\bseizing\s+right\s+now\b|\bcurrently\s+seizing\b"),
    ],
)

_INJECTION_RULES = RuleSet(
    category=RiskCategory.PROMPT_INJECTION,
    action=SafetyAction.REFUSE,
    rules=[
        _r("injection.ignore_instructions",
           r"\bignore\s+(your\s+)?(previous|prior|above|all|system)?\s*(instructions?|prompt|guidelines?|rules?)\b"),
        _r("injection.pretend_unrestricted",
           r"\bpretend\s+(you\s+are|you're|to\s+be)\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|dAN|evil)\b"),
        _r("injection.act_as_different_ai",
           r"\bact\s+as\s+(an?\s+)?(unrestricted|unfiltered|different|evil|jailbroken)\b"),
        _r("injection.reveal_system_prompt",
           r"\b(reveal|show|print|output|repeat|ignore\s+and\s+print)\s+(your\s+)?(system\s+prompt|instructions)\b"),
        _r("injection.override_guidelines",
           r"\b(override|bypass|disable|remove)\s+(your\s+)?(safety|guidelines?|restrictions?|filter)\b"),
        _r("injection.new_instructions",
           r"\byour\s+new\s+(instructions?|guidelines?|rules?)\s+(are|is)\b"),
        _r("injection.developer_mode",
           r"\b(developer|dev|god|admin)\s+mode\b"),
        _r("injection.dan",
           r"\bDAN\b|\bdo\s+anything\s+now\b"),
    ],
)

_DANGEROUS_RULES = RuleSet(
    category=RiskCategory.DANGEROUS_INSTRUCTION,
    action=SafetyAction.REFUSE,
    rules=[
        _r("dangerous.induce_vomiting",
           r"\bhow\s+to\s+(induce|make\s+(yourself|someone)\s+)(vomit|throw\s+up)\b"),
        _r("dangerous.starvation",
           r"\bhow\s+(long\s+can\s+i\s+(go\s+)?)?(survive\s+)?(without\s+eating|starv(e|ing))\b"),
        _r("dangerous.dangerous_substance_mix",
           r"\b(mix|combine|take)\s+.{0,30}\b(bleach|ammonia|lye)\b"),
        _r("dangerous.harmful_purging",
           r"\bhow\s+to\s+(purge|get\s+rid\s+of\s+(food|calories)\s+(after|without))\b"),
    ],
)

_DIAGNOSIS_RULES = RuleSet(
    category=RiskCategory.DIAGNOSIS_REQUEST,
    action=SafetyAction.REFUSE,
    rules=[
        _r("diagnosis.do_i_have",
           r"\bdo\s+i\s+have\b"),
        _r("diagnosis.what_disease",
           r"\bwhat\s+(disease|condition|illness|disorder)\s+(do\s+i\s+have|is\s+this|explains?)\b"),
        _r("diagnosis.is_this_cancer",
           r"\bis\s+this\s+(cancer|diabetes|covid|hiv|lupus|ms|als)\b"),
        _r("diagnosis.diagnose_me",
           r"\b(diagnose\s+me|give\s+me\s+a\s+diagnosis|tell\s+me\s+what\s+i\s+have)\b"),
        _r("diagnosis.my_symptoms_are",
           r"\bmy\s+symptoms\s+are\b.{0,80}\b(what\s+(is|do)\s+i|could\s+it\s+be|is\s+it)\b"),
    ],
)

_MEDICATION_RULES = RuleSet(
    category=RiskCategory.MEDICATION_REQUEST,
    action=SafetyAction.REFUSE,
    rules=[
        _r("medication.what_dose",
           r"\bwhat\s+(dose|dosage|amount)\s+(of|should)\b"),
        _r("medication.how_much_to_take",
           r"\bhow\s+much\s+.{0,20}\bshould\s+i\s+take\b"),
        _r("medication.can_i_take_together",
           r"\bcan\s+i\s+(take|combine|mix)\s+.{0,40}"
           r"(mg|mcg|aspirin|ibuprofen|paracetamol|acetaminophen|antibiotic|metformin)\b"),
        _r("medication.prescribe_me",
           r"\bprescribe\s+me\b|\bprescribe\s+(me\s+)?something\b"),
        _r("medication.switch_medication",
           r"\bshould\s+i\s+switch\s+(from|to)\b.{0,30}\b(mg|tablet|pill|drug|medication)\b"),
        _r("medication.specific_dosage_query",
           r"\b\d+\s*(mg|mcg|ml|units?)\b.{0,30}\b(too\s+much|enough|safe|right\s+dose)\b"),
    ],
)

_SCOPE_RULES = RuleSet(
    category=RiskCategory.OUT_OF_SCOPE,
    action=SafetyAction.REFUSE,
    rules=[
        _r("scope.stock_market",
           r"\b(stock|share\s+price|invest(ment|ing)?|crypto(currency)?|bitcoin|nft)\b"),
        _r("scope.sports_scores",
           r"\b(who\s+won|final\s+score|match\s+result|nba|nfl|ipl|cricket\s+score)\b"),
        _r("scope.coding_help",
           r"\b(write\s+(a\s+)?(function|code|script|program)|debug\s+my\s+code|python|javascript|sql\s+query)\b"),
        _r("scope.politics",
           r"\b(who\s+should\s+i\s+vote|political\s+party|election\s+result|trump|biden|modi)\b"),
        _r("scope.recipe_non_health",
           r"\b(recipe\s+for|how\s+to\s+(cook|bake|make)\s+(?!healthy|nutritious|diabetic|low[- ]fat))\b"),
        _r("scope.entertainment",
           r"\b(best\s+movie|watch\s+tonight|tv\s+show|netflix|spotify|gaming|video\s+game)\b"),
        _r("scope.weather",
           r"\b(weather\s+(today|tomorrow|forecast)|will\s+it\s+rain)\b"),
        _r("scope.legal_advice",
           r"\b(is\s+it\s+legal|can\s+i\s+sue|legal\s+advice|law\s+says)\b"),
    ],
)

# Priority order — the guard iterates this list and returns on the first hit.
_RULE_SETS: list[RuleSet] = [
    _SELF_HARM_RULES,
    _EMERGENCY_RULES,
    _INJECTION_RULES,
    _DANGEROUS_RULES,
    _DIAGNOSIS_RULES,
    _MEDICATION_RULES,
    _SCOPE_RULES,
]


# ---------------------------------------------------------------------------
# Public guard class
# ---------------------------------------------------------------------------


class InputGuard:
    """Screen user input before it reaches the language model.

    Usage::

        guard = InputGuard(max_chars=2000)
        verdict = guard.screen("Do I have diabetes?")
        if verdict.blocks_model_call:
            return refusal_for(verdict.category)

    The guard is stateless and cheap to construct — instantiate it once at
    service startup rather than per request.

    Args:
        max_chars: Maximum permitted input length in characters. Inputs
            exceeding this are rejected with
            :class:`~src.utils.exceptions.InputTooLongError` *before* pattern
            matching runs, limiting prompt-stuffing and denial-of-wallet
            attempts.
    """

    def __init__(self, max_chars: int = 2000) -> None:
        self._max_chars = max_chars

    def screen(self, text: str) -> SafetyVerdict:
        """Screen *text* and return a verdict.

        Pre-checks (raise, do not return a verdict):
        - Empty or whitespace-only input → :class:`~src.utils.exceptions.EmptyInputError`
        - Input longer than ``max_chars`` → :class:`~src.utils.exceptions.InputTooLongError`

        Pattern matching (returns a verdict, never raises):
        - First :class:`RuleSet` with any matches wins.
        - ``SafetyVerdict.matched_rules`` carries every rule ID that fired in
          that set so false positives are traceable without re-running the guard.

        Args:
            text: Raw user input, exactly as typed.

        Returns:
            :class:`~src.models.safety.SafetyVerdict` with action ALLOW,
            REFUSE, or ESCALATE.

        Raises:
            EmptyInputError: Input is blank.
            InputTooLongError: Input exceeds ``max_chars``.
        """
        # --- pre-checks (cheap, no regex) ---
        stripped = text.strip()
        if not stripped:
            raise EmptyInputError("User submitted empty input.")

        if len(stripped) > self._max_chars:
            raise InputTooLongError(
                f"Input length {len(stripped)} exceeds limit {self._max_chars}.",
                context={"input_length": len(stripped), "limit": self._max_chars},
            )

        # --- pattern matching ---
        for rule_set in _RULE_SETS:
            matched = rule_set.matches(stripped)
            if matched:
                return SafetyVerdict(
                    action=rule_set.action,
                    category=rule_set.category,
                    reason=(
                        f"Input matched {rule_set.category.value} rules: "
                        f"{', '.join(matched)}"
                    ),
                    confidence=1.0,
                    matched_rules=matched,
                )

        return SafetyVerdict.allow()
