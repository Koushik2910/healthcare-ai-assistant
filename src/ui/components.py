"""Reusable Streamlit widgets for the Healthcare AI Assistant UI.

Each function renders one self-contained UI element.  They accept domain
model objects directly so the rendering logic stays decoupled from session
state management in ``app.py``.

Design decisions
----------------
- Functions, not classes.  Streamlit's programming model is procedural;
  wrapping widgets in classes adds indirection without benefit.
- No ``st.container()`` nesting beyond what's needed for layout.  Every
  extra container is a potential source of whitespace glitches.
- HTML is emitted only where Streamlit's built-ins can't achieve the
  desired style (message bubbles, crisis card, score bars).  Everything
  else uses native widgets.
"""

from __future__ import annotations

import streamlit as st

from src.models.chat import ChatResponse, Citation, ResponseSource, Role
from src.models.rag import RetrievedChunk, RetrievalResult
from src.models.safety import RiskCategory, SafetyAction, SafetyVerdict

# ---------------------------------------------------------------------------
# Crisis / emergency redirect card
# ---------------------------------------------------------------------------

# Hotline copy — kept in one place so it's easy to update.
_CRISIS_HOTLINES = """
<div class="crisis-card">
  <div class="crisis-icon">&#9888;</div>
  <div class="crisis-body">
    <p class="crisis-title">Immediate Help Available</p>
    <p class="crisis-lines">
      <strong>Medical Emergency:</strong> Call <span class="hotline">112</span><br>
      <strong>Mental Health Crisis / Suicide Lifeline:</strong> Call or text <span class="hotline">988</span><br>
      <strong>iCall (India):</strong> <span class="hotline">9152987821</span>
    </p>
    <p class="crisis-note">
      Please reach out to one of these services right now.
      They are available 24&nbsp;hours a day, 7&nbsp;days a week.
    </p>
  </div>
</div>
"""

_EMERGENCY_HOTLINES = """
<div class="crisis-card">
  <div class="crisis-icon">&#128680;</div>
  <div class="crisis-body">
    <p class="crisis-title">This sounds like a medical emergency</p>
    <p class="crisis-lines">
      <strong>Call 112 immediately</strong> or have someone take you to the nearest
      emergency department.<br>
      Do not wait — seconds matter in an emergency.
    </p>
  </div>
</div>
"""


def render_crisis_card(verdict: SafetyVerdict) -> None:
    """Render a full-width crisis/emergency redirect card.

    Called only when ``verdict.action == SafetyAction.ESCALATE``.  The card
    replaces the normal response area rather than sitting beside it.

    Args:
        verdict: The :class:`~src.models.safety.SafetyVerdict` that triggered
            escalation.  Used to choose between the self-harm and emergency
            variants.
    """
    if verdict.category == RiskCategory.EMERGENCY:
        st.markdown(_EMERGENCY_HOTLINES, unsafe_allow_html=True)
    else:
        # SELF_HARM or any other ESCALATE category
        st.markdown(_CRISIS_HOTLINES, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Safety status badge
# ---------------------------------------------------------------------------

_BADGE_HTML = """<span class="safety-badge badge-{level}">{label}</span>"""

_BADGE_MAP: dict[RiskCategory, tuple[str, str]] = {
    RiskCategory.NONE: ("safe", "Verified Safe"),
    RiskCategory.OUT_OF_SCOPE: ("caution", "Out of Scope"),
    RiskCategory.DIAGNOSIS_REQUEST: ("caution", "No Diagnosis"),
    RiskCategory.MEDICATION_REQUEST: ("caution", "No Prescription"),
    RiskCategory.DANGEROUS_INSTRUCTION: ("warn", "Flagged"),
    RiskCategory.PROMPT_INJECTION: ("warn", "Blocked"),
    RiskCategory.SELF_HARM: ("crisis", "Crisis"),
    RiskCategory.EMERGENCY: ("crisis", "Emergency"),
}


def render_status_badge(verdict: SafetyVerdict) -> None:
    """Render a small coloured safety badge inline.

    Shown beneath assistant messages when the input guard fired
    (``verdict.action != SafetyAction.ALLOW``).  Not shown for clean passes
    to avoid visual noise on every message.

    Args:
        verdict: The screening result for this turn's input.
    """
    if verdict.action == SafetyAction.ALLOW:
        return  # nothing to show — no noise on clean messages

    level, label = _BADGE_MAP.get(verdict.category, ("caution", "Reviewed"))
    st.markdown(
        _BADGE_HTML.format(level=level, label=label),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Source card (inside the RAG expander)
# ---------------------------------------------------------------------------


def render_source_card(rc: RetrievedChunk, index: int) -> None:
    """Render one retrieved chunk as a styled source card.

    Args:
        rc: The :class:`~src.models.rag.RetrievedChunk` to display.
        index: 1-based display number.
    """
    score_pct = int(rc.score * 100)
    url_part = (
        f'<a class="source-url" href="{rc.chunk.url}" target="_blank">'
        f"{rc.chunk.url}</a>"
        if rc.chunk.url
        else ""
    )
    html = f"""
<div class="source-card">
  <span class="source-num">[{index}]</span>
  <span class="source-title">{rc.chunk.title}</span>
  <span class="source-pub">{rc.chunk.source}</span>
  <div class="source-snippet">{rc.chunk.text[:260].rstrip()}&#8230;</div>
  <div class="source-meta">
    Relevance&nbsp;{score_pct}%
    &nbsp;&middot;&nbsp;{rc.chunk.licence.value.replace("_", " ")}
    {("&nbsp;&middot;&nbsp;" + url_part) if url_part else ""}
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# RAG sources expander
# ---------------------------------------------------------------------------


def render_sources_expander(retrieval: RetrievalResult | None) -> None:
    """Render a collapsible 'Sources' expander for one assistant turn.

    Shows nothing when retrieval is ``None`` or produced no usable chunks.
    Shows a degraded-retrieval notice when ``retrieval.degraded`` is True.

    Args:
        retrieval: The retrieval result attached to this turn, or ``None``
            when the assistant answered from parametric knowledge only.
    """
    if retrieval is None:
        return

    if retrieval.degraded:
        with st.expander("⚠️ Knowledge base unavailable", expanded=False):
            st.caption(
                "The knowledge base could not be queried for this question. "
                "The answer is based on the model's general knowledge only."
            )
        return

    if not retrieval.has_context:
        return  # no relevant chunks — don't show an empty expander

    label = f"📚 Sources ({len(retrieval.chunks)})"
    with st.expander(label, expanded=False):
        for i, rc in enumerate(retrieval.top(5), start=1):
            render_source_card(rc, i)
            if i < len(retrieval.chunks):
                st.markdown("<hr class='source-divider'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Message bubble
# ---------------------------------------------------------------------------


def render_message_bubble(
    role: str,
    content: str,
    *,
    response: ChatResponse | None = None,
    retrieval: RetrievalResult | None = None,
) -> None:
    """Render one chat turn as a styled message bubble.

    Uses Streamlit's ``st.chat_message`` context manager so the avatar and
    role label are handled natively.  HTML enhancements (disclaimer pill,
    source badges) are injected inside the native container.

    Args:
        role: ``"user"`` or ``"assistant"``.
        content: The message text.
        response: Full :class:`~src.models.chat.ChatResponse` for assistant
            turns.  ``None`` for user messages.
        retrieval: The retrieval result for this turn.  ``None`` for user
            messages or when RAG was not active.
    """
    with st.chat_message(role):
        st.markdown(content)

        if role == Role.ASSISTANT.value and response is not None:
            # Disclaimer pill
            if response.disclaimer:
                st.markdown(
                    '<p class="disclaimer-pill">&#x2139;&#xFE0F;&nbsp;'
                    "This information is for educational purposes only. "
                    "Always consult a qualified healthcare professional.</p>",
                    unsafe_allow_html=True,
                )

            # Source badge for non-clean safety actions
            # (crisis cards are rendered separately in app.py before this)
            if response.refused and response.source == ResponseSource.GUARDRAIL:
                st.markdown(
                    '<span class="safety-badge badge-caution">&#x1F6AB;&nbsp;Guardrail</span>',
                    unsafe_allow_html=True,
                )

            # Grounded badge
            if response.source == ResponseSource.GROUNDED:
                st.markdown(
                    '<span class="safety-badge badge-safe">&#x1F4DA;&nbsp;RAG Grounded</span>',
                    unsafe_allow_html=True,
                )

            # RAG sources
            render_sources_expander(retrieval)
