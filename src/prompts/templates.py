"""Static response templates for guardrail-intercepted messages.

Design rationale
----------------
When the input guardrail (Phase 4) sets ``SafetyVerdict.action`` to
:attr:`~src.models.safety.SafetyAction.REFUSE` or
:attr:`~src.models.safety.SafetyAction.ESCALATE`, the model is never called.
The response that reaches the user comes entirely from this module — not from
the LLM.

Keeping these strings here rather than inline in the service layer has three
advantages:

1. **Testable in isolation.** The evaluation harness (Phase 7) can assert
   that every ``RiskCategory`` maps to a template and that every template
   contains the required elements (crisis numbers, no diagnostic language,
   correct redirect).

2. **Auditable.** A non-engineer (product manager, medical advisor) can read
   this file and confirm the wording is correct without understanding the
   rest of the codebase.

3. **No accidental coupling.** The service layer calls
   :func:`refusal_for` with a :class:`~src.models.safety.RiskCategory` and
   gets back a string.  It does not care how the string was chosen; this
   module does not care how the string is rendered.

Wording principles
------------------
- Refusals are **redirects**, not rejections. The goal is to get the user to
  the right resource, not to scold them.
- Crisis templates prioritise **speed and resources** over empathy prose.
  A user in crisis needs a phone number in the first sentence, not a
  paragraph of acknowledgement before it.
- Refusal templates for out-of-scope or overreach situations are **one to
  three sentences**: acknowledge, decline, redirect.
- No template fabricates medical information. Even the "I can share general
  information…" language is carefully qualified.
"""

from __future__ import annotations

from src.models.safety import RiskCategory

# ---------------------------------------------------------------------------
# Crisis and emergency templates (ESCALATE action)
# ---------------------------------------------------------------------------

_SELF_HARM_TEMPLATE = """\
You're not alone, and I'm glad you reached out. Please contact a crisis \
service right now — they are available 24/7 and are there to help:

• **988 Suicide and Crisis Lifeline** — call or text **988** (US)
• **Crisis Text Line** — text **HOME** to **741741** (US)
• **International Association for Suicide Prevention** — \
https://www.iasp.info/resources/Crisis_Centres/ (worldwide directory)

If you are in immediate danger, please call **911** (or your local \
emergency number) or go to your nearest emergency room.

I'm a health information assistant and I'm not equipped to provide the \
support you need right now — but the people at these services are. \
Please reach out to them.\
"""

_EMERGENCY_TEMPLATE = """\
**This sounds like a medical emergency.**

**Call 911 (or your local emergency number) immediately.** \
Do not wait or search for more information online.

If you are with someone who is unconscious and not breathing normally, \
begin CPR if you are trained to do so and keep the emergency dispatcher \
on the line — they can guide you.

I'm a health information assistant, not an emergency service. \
Please call emergency services now.\
"""

# ---------------------------------------------------------------------------
# Refusal templates (REFUSE action)
# ---------------------------------------------------------------------------

_DIAGNOSIS_TEMPLATE = """\
I can share general information about the symptoms or conditions you \
mentioned, but I'm not able to diagnose any specific condition — that \
requires a physical examination, your medical history, and the professional \
judgement of a qualified clinician.

If you'd like, I can explain what those symptoms generally indicate, \
what kinds of conditions are commonly associated with them, or what to \
expect from a doctor's visit. But please do consult a doctor or urgent \
care provider for an actual assessment.

*This information is for general educational purposes only and is not a \
substitute for professional medical advice.*\
"""

_MEDICATION_TEMPLATE = """\
Medication decisions — including which drug to take, what dose is \
appropriate, and whether two medications are safe to combine — must be \
made with a pharmacist or prescribing physician who knows your complete \
medical history, other medications, and current health status.

I'm not able to provide that guidance. If you have an urgent question \
about a medication you've already been prescribed, your pharmacist is \
the fastest and most reliable resource, and most pharmacies offer free \
phone consultations.

*This information is for general educational purposes only and is not a \
substitute for professional medical advice.*\
"""

_OUT_OF_SCOPE_TEMPLATE = """\
That's outside my scope as a health and wellness assistant. \
I'm here to help with health education, nutrition, lifestyle, first aid, \
and general medical information.

Is there a health-related question I can help you with?\
"""

_PROMPT_INJECTION_TEMPLATE = """\
I can only help with health and wellness questions.\
"""

_DANGEROUS_INSTRUCTION_TEMPLATE = """\
I'm not able to provide that information, as following it could cause harm. \
If you have a health concern that prompted this question, I'm happy to share \
safe, evidence-based information instead.

*If this is a medical emergency, please call 911 or your local emergency \
number immediately.*\
"""

# ---------------------------------------------------------------------------
# General disclaimer (appended by the service layer when needed)
# ---------------------------------------------------------------------------

MEDICAL_DISCLAIMER = (
    "*This information is for general educational purposes only and is not "
    "a substitute for professional medical advice. Please consult a qualified "
    "healthcare provider for guidance specific to your situation.*"
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Mapping from RiskCategory to the template string the service layer should
# return when that category blocks the model call.  Both ESCALATE and REFUSE
# categories are included; the service layer selects the right action from
# the SafetyVerdict and looks up the wording here.
_TEMPLATES: dict[RiskCategory, str] = {
    RiskCategory.SELF_HARM: _SELF_HARM_TEMPLATE,
    RiskCategory.EMERGENCY: _EMERGENCY_TEMPLATE,
    RiskCategory.DIAGNOSIS_REQUEST: _DIAGNOSIS_TEMPLATE,
    RiskCategory.MEDICATION_REQUEST: _MEDICATION_TEMPLATE,
    RiskCategory.OUT_OF_SCOPE: _OUT_OF_SCOPE_TEMPLATE,
    RiskCategory.PROMPT_INJECTION: _PROMPT_INJECTION_TEMPLATE,
    RiskCategory.DANGEROUS_INSTRUCTION: _DANGEROUS_INSTRUCTION_TEMPLATE,
}

# Fallback used when a category has no specific template.  This should never
# be reached in practice, but having a safe fallback is better than raising
# an unhandled KeyError in production.
_FALLBACK_TEMPLATE = (
    "I'm not able to help with that request. "
    "Please ask a health or wellness question and I'll do my best to assist."
)


def refusal_for(category: RiskCategory) -> str:
    """Return the user-facing refusal or escalation message for a category.

    This is the *only* function the service layer should call.  It maps
    a :class:`~src.models.safety.RiskCategory` to a ready-to-display string
    that has been reviewed for safety wording.

    Args:
        category: The risk category that triggered the guardrail.

    Returns:
        A user-safe string.  Never raises; returns the fallback template for
        any category not explicitly mapped (including ``RiskCategory.NONE``,
        which should not reach this function in normal operation).

    Examples:
        >>> msg = refusal_for(RiskCategory.OUT_OF_SCOPE)
        >>> "health" in msg.lower()
        True
        >>> msg = refusal_for(RiskCategory.SELF_HARM)
        >>> "988" in msg
        True
    """
    return _TEMPLATES.get(category, _FALLBACK_TEMPLATE)


def all_mapped_categories() -> list[RiskCategory]:
    """Return every :class:`~src.models.safety.RiskCategory` that has a template.

    Used by the evaluation harness (Phase 7) to assert complete coverage
    without hard-coding the list of categories in the test.

    Returns:
        List of categories, in the order they appear in ``_TEMPLATES``.
    """
    return list(_TEMPLATES.keys())
