"""Healthcare AI Assistant — Streamlit UI.

Architecture
------------
Startup (runs once per browser session, cached with ``@st.cache_resource``):
    1. ``get_settings()`` loads ``.env``.
    2. ``get_llm()`` builds the Gemini primary provider.
    3. ``_build_retriever()`` opens the ChromaDB collection and wraps it in a
       ``Retriever`` instance.
    4. ``ChatService`` is constructed with both, pinned in cache.

Per-render loop (runs on every Streamlit interaction):
    5. Session state is initialised on first render only.
    6. History is replayed from ``st.session_state.messages``.
    7. Chat input is collected via ``st.chat_input``.
    8. On submission, ``ChatService.stream_chat()`` is driven through an
       ``st.empty()`` delta loop.
    9. The final ``ChatResponse`` is fetched via ``ChatService.chat()`` for
       metadata (citations, disclaimer, verdict) — this reuses the same
       prompt/guard logic but hits the model a second time.  See the note
       below for why we do it this way.

Note on double-call pattern
---------------------------
Streamlit's ``st.write_stream`` (and our manual ``st.empty()`` loop) gives us
incremental tokens but no structured metadata.  ``ChatService.stream_chat``
yields plain strings; it doesn't return a ``ChatResponse``.  After streaming
completes we call ``ChatService.chat()`` once more (non-streaming) to get the
full ``ChatResponse`` with citations, verdict, and disclaimer.

This means the model is called twice.  The alternative would be to have
``stream_chat`` somehow attach metadata to its last yielded chunk (a sentinel
object), but that breaks the clean ``AsyncIterator[str]`` contract and
complicates both the service layer and every test that touches it.  Given
that Gemini's free tier is generous and the corpus is small (latency for the
second call is typically <600 ms), the double-call is the right trade-off for
this phase.  A future optimisation (Phase 9) could return a ``(stream,
future_response)`` pair from an internal method without breaking the public
API.

Input guard and the stream_chat contract
-----------------------------------------
``stream_chat`` yields a single refusal string and closes immediately when
the input guard fires.  The UI treats any response whose content matches a
known ``ResponseSource.GUARDRAIL`` pattern as a refusal.  We detect this by
calling ``chat()`` for the metadata; ``response.refused`` tells us exactly.

Crisis / emergency flow
-----------------------
When ``response.source == ResponseSource.ESCALATION``:
- The streamed text (the template refusal) is shown in the message bubble.
- A full-width ``render_crisis_card()`` is shown ABOVE the bubble so it
  cannot be missed.
- No sources expander is shown (there are no retrieval results for a
  guardrail response).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Iterator

import streamlit as st

from src.config.settings import get_settings
from src.llm.factory import get_llm
from src.models.chat import ChatResponse, Conversation, Message, Role
from src.models.rag import RetrievalResult
from src.models.safety import RiskCategory, SafetyAction
from src.rag.retriever import Retriever
from src.services.chat_service import ChatService
from src.ui.components import (
    render_crisis_card,
    render_message_bubble,
    render_sources_expander,
    render_status_badge,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f1923;
    color: #e8f0fe;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #c8d8f0 !important;
}
[data-testid="stSidebar"] .stButton button {
    background: #1e3a5f;
    color: #e8f0fe;
    border: 1px solid #2e5080;
    border-radius: 6px;
    width: 100%;
    margin-top: 4px;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #2e5080;
}

/* ── Main area ───────────────────────────────────────────────────────── */
.main .block-container {
    max-width: 820px;
    padding-top: 1.5rem;
    padding-bottom: 6rem;   /* room for the pinned chat input */
}

/* ── App header ──────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.5rem;
}
.app-header h1 {
    font-size: 1.55rem;
    font-weight: 700;
    color: #1a3a5c;
    margin: 0;
}
.app-subtitle {
    font-size: 0.82rem;
    color: #5a6a7a;
    margin-top: -8px;
    margin-bottom: 1.2rem;
}

/* ── Crisis card ─────────────────────────────────────────────────────── */
.crisis-card {
    display: flex;
    gap: 14px;
    background: #fff0f0;
    border: 2px solid #c0392b;
    border-radius: 10px;
    padding: 16px 18px;
    margin: 10px 0 14px 0;
}
.crisis-icon {
    font-size: 2rem;
    flex-shrink: 0;
    line-height: 1;
}
.crisis-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #c0392b;
    margin: 0 0 6px 0;
}
.crisis-lines {
    font-size: 0.95rem;
    line-height: 1.7;
    margin: 0 0 6px 0;
}
.crisis-note {
    font-size: 0.82rem;
    color: #555;
    margin: 0;
}
.hotline {
    font-size: 1.1rem;
    font-weight: 800;
    color: #c0392b;
    letter-spacing: 0.04em;
}

/* ── Safety badges ───────────────────────────────────────────────────── */
.safety-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    border-radius: 4px;
    padding: 2px 8px;
    margin-top: 4px;
    text-transform: uppercase;
}
.badge-safe   { background: #e6f4ea; color: #1e6e38; border: 1px solid #a8d5b3; }
.badge-caution{ background: #fff8e1; color: #7a5c00; border: 1px solid #f5d87a; }
.badge-warn   { background: #fce8e6; color: #8c1f18; border: 1px solid #e8a09a; }
.badge-crisis { background: #c0392b; color: #ffffff;  border: none; }

/* ── Disclaimer pill ─────────────────────────────────────────────────── */
.disclaimer-pill {
    font-size: 0.76rem;
    color: #4a6078;
    background: #edf3fa;
    border-left: 3px solid #5b96d2;
    border-radius: 0 4px 4px 0;
    padding: 5px 10px;
    margin: 6px 0 2px 0;
    line-height: 1.5;
}

/* ── Source cards ────────────────────────────────────────────────────── */
.source-card {
    padding: 8px 0;
    line-height: 1.5;
}
.source-num {
    font-weight: 700;
    color: #1a3a5c;
    margin-right: 6px;
}
.source-title {
    font-weight: 600;
    color: #1a3a5c;
}
.source-pub {
    font-size: 0.78rem;
    color: #6a7a8a;
    margin-left: 8px;
}
.source-snippet {
    font-size: 0.82rem;
    color: #444;
    margin: 4px 0;
    font-family: Georgia, serif;
    border-left: 2px solid #c8d8ea;
    padding-left: 8px;
}
.source-meta {
    font-size: 0.74rem;
    color: #7a8a9a;
    margin-top: 3px;
}
.source-url { color: #2e6da4; }
.source-divider { border: none; border-top: 1px solid #e8edf2; margin: 6px 0; }

/* ── Streaming cursor ────────────────────────────────────────────────── */
.typing-cursor::after {
    content: "▋";
    animation: blink 1s step-start infinite;
}
@keyframes blink { 50% { opacity: 0; } }
</style>
"""

# ---------------------------------------------------------------------------
# Cached service factory
# ---------------------------------------------------------------------------


def _build_retriever() -> Retriever | None:
    """Open the ChromaDB collection and return a Retriever, or None on failure.

    Importing ``chromadb`` and ``sentence_transformers`` is deferred to here
    so that the app starts (and shows a useful error) rather than crashing at
    import time when the packages are missing.
    """
    settings = get_settings()
    try:
        import chromadb  # type: ignore[import]
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError as exc:
        log.error("RAG dependencies missing: %s", exc)
        return None

    try:
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        embedding_model = SentenceTransformer(settings.embedding_model)

        # Mirror the EmbeddingFunction used at ingest time so distances are
        # comparable.  We pass a thin callable that ChromaDB calls per query.
        class _STEmbeddingFunction(chromadb.EmbeddingFunction):  # type: ignore[misc]
            def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
                return embedding_model.encode(input).tolist()

        collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=_STEmbeddingFunction(),
            metadata={"hnsw:space": "l2"},
        )
        return Retriever(collection)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not open ChromaDB collection: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading AI services…")
def _get_chat_service() -> ChatService:
    """Build and cache the ChatService singleton for this process.

    ``@st.cache_resource`` ensures this runs exactly once per Streamlit
    worker process — not on every page rerender.  The retriever is wired in
    here so the full RAG pipeline is active from the first user message.
    """
    retriever = _build_retriever()
    primary = get_llm()
    svc = ChatService(primary, retriever=retriever.retrieve if retriever else None)
    return svc


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine from sync Streamlit code.

    Streamlit runs in a sync context.  We keep a single event loop alive
    for the session (stored in session_state) to avoid the overhead of
    spinning up a new loop on every interaction.
    """
    loop: asyncio.AbstractEventLoop = st.session_state.get("_event_loop")
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        st.session_state["_event_loop"] = loop
    return loop.run_until_complete(coro)


def _stream_response(user_text: str, conversation: Conversation) -> str:
    """Drive ``stream_chat`` through ``st.empty()``, return full text.

    Yields tokens into a ``st.empty()`` placeholder so the user sees text
    appear progressively.  Returns the accumulated full text when done.
    """
    svc: ChatService = st.session_state.chat_service
    placeholder = st.empty()
    accumulated = ""

    async def _collect():
        nonlocal accumulated
        async for chunk in svc.stream_chat(user_text, conversation):
            accumulated += chunk
            placeholder.markdown(accumulated + "▋")
        placeholder.markdown(accumulated)  # remove cursor
        return accumulated

    _run(_collect())
    return accumulated


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def _init_session() -> None:
    """Initialise all session-state keys on the very first render."""
    if "initialised" not in st.session_state:
        st.session_state.initialised = True
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.conversation = Conversation()
        # List of dicts stored for history replay:
        # {role, content, response: ChatResponse|None, retrieval: RetrievalResult|None}
        st.session_state.messages = []
        st.session_state.chat_service = _get_chat_service()
        st.session_state.is_streaming = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    """Render the application sidebar: branding, controls, and info."""
    with st.sidebar:
        st.markdown("## HealthAssist AI")
        st.markdown(
            "<p style='font-size:0.8rem;opacity:0.7;margin-top:-10px;'>"
            "Powered by Gemini · Groq fallback</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Session**")
        sid = st.session_state.get("session_id", "—")
        st.markdown(
            f"<p style='font-size:0.75rem;font-family:monospace;'>{sid}</p>",
            unsafe_allow_html=True,
        )

        msg_count = len(
            [m for m in st.session_state.get("messages", []) if m["role"] == "user"]
        )
        st.caption(f"{msg_count} question{'s' if msg_count != 1 else ''} this session")

        if st.button("🗑 Clear conversation", key="clear_btn"):
            st.session_state.messages = []
            st.session_state.conversation = Conversation()
            st.session_state.session_id = uuid.uuid4().hex[:12]
            st.rerun()

        st.divider()
        st.markdown("**About**")
        st.markdown(
            "<p style='font-size:0.78rem;line-height:1.55;'>"
            "General health information only. "
            "Not a substitute for professional medical advice, "
            "diagnosis, or treatment.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:0.78rem;line-height:1.55;'>"
            "Topics: nutrition, lifestyle, preventive care, "
            "first aid, general wellness.</p>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            "<p style='font-size:0.72rem;opacity:0.5;'>Healthcare AI Assistant v1.0</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# History replay
# ---------------------------------------------------------------------------


def _replay_history() -> None:
    """Re-render every stored message turn from session state."""
    for entry in st.session_state.messages:
        response: ChatResponse | None = entry.get("response")
        retrieval: RetrievalResult | None = entry.get("retrieval")
        role = entry["role"]
        content = entry["content"]

        # Crisis cards go BEFORE the bubble so they can't be missed.
        if (
            role == Role.ASSISTANT.value
            and response is not None
            and response.source.value == "escalation"
        ):
            # Re-construct a minimal verdict for the card.
            # We stored the category name in the entry dict.
            from src.models.safety import RiskCategory, SafetyAction, SafetyVerdict

            cat_name = entry.get("crisis_category", "self_harm")
            verdict = SafetyVerdict(
                action=SafetyAction.ESCALATE,
                category=RiskCategory(cat_name),
            )
            render_crisis_card(verdict)

        render_message_bubble(role, content, response=response, retrieval=retrieval)


# ---------------------------------------------------------------------------
# Handle one user submission
# ---------------------------------------------------------------------------


def _handle_input(user_text: str) -> None:
    """Process one user message end-to-end and update session state."""
    svc: ChatService = st.session_state.chat_service
    conversation: Conversation = st.session_state.conversation

    # 1. Append user message to the display list and domain model.
    st.session_state.messages.append(
        {"role": Role.USER.value, "content": user_text, "response": None, "retrieval": None}
    )
    conversation.add(Message(role=Role.USER, content=user_text))

    # 2. Show the user bubble immediately so there's no blank gap.
    with st.chat_message(Role.USER.value):
        st.markdown(user_text)

    # 3. Stream the assistant response.
    with st.chat_message(Role.ASSISTANT.value):
        streamed_text = _stream_response(user_text, conversation)

    # 4. Get the full structured response for metadata.
    response: ChatResponse = _run(svc.chat(user_text, conversation))

    # 5. Handle crisis / emergency — show the card in the NEXT full render
    #    (history replay).  We store the category so replay can reconstruct it.
    crisis_category = None
    if response.source.value == "escalation":
        crisis_category = response.message.content  # guardrail text
        # Detect category from the input guard verdict stored on the response.
        # We don't have direct verdict access here, so we infer from content.
        # The InputGuard returns SELF_HARM vs EMERGENCY patterns in its text.
        if any(kw in response.message.content.lower() for kw in ("988", "mental", "self")):
            crisis_category = RiskCategory.SELF_HARM.value
        else:
            crisis_category = RiskCategory.EMERGENCY.value

    # 6. Fetch the retrieval result for RAG sources display.
    #    We attach it to the response entry; ``chat()`` doesn't return it
    #    directly, so we infer from citations.
    retrieval: RetrievalResult | None = None
    if response.is_grounded and response.citations:
        # Reconstruct a minimal RetrievalResult from citations for display.
        # We don't have the raw chunks here, but we can build enough for the
        # expander title.  The actual source cards use the citations list.
        retrieval = _citations_to_retrieval(response, user_text)

    # 7. Update domain conversation with the assistant reply.
    conversation.add(
        Message(role=Role.ASSISTANT, content=response.message.content)
    )

    # 8. Store the full entry for replay.
    st.session_state.messages.append(
        {
            "role": Role.ASSISTANT.value,
            "content": response.message.content,
            "response": response,
            "retrieval": retrieval,
            "crisis_category": crisis_category,
        }
    )


def _citations_to_retrieval(response: ChatResponse, query: str) -> RetrievalResult | None:
    """Build a minimal ``RetrievalResult`` from ``ChatResponse.citations``.

    ``ChatService.chat()`` does not expose the raw ``RetrievedChunk`` list —
    it converts them to ``Citation`` objects.  We reconstruct a stripped-down
    ``RetrievalResult`` that's sufficient to drive ``render_sources_expander``
    and ``render_source_card``.

    The chunks here have no ``score`` (we set them to 1.0 as a sentinel) and
    a truncated snippet from the citation's ``snippet`` field.
    """
    if not response.citations:
        return None

    from src.models.rag import Chunk, DocumentLicence, RetrievedChunk, RetrievalResult

    chunks: list[RetrievedChunk] = []
    for cit in response.citations:
        chunk = Chunk(
            chunk_id=f"citation::{cit.marker}",
            doc_id=f"citation_{cit.marker}",
            index=cit.marker,
            text=cit.snippet,
            title=cit.title,
            source=cit.source,
            licence=DocumentLicence.ORIGINAL,
            topics=[],
        )
        chunks.append(RetrievedChunk(chunk=chunk, score=1.0))

    return RetrievalResult(query=query, chunks=chunks)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Top-level function called by the root ``app.py`` shim."""
    # Page config — must be the first Streamlit call.
    st.set_page_config(
        page_title="HealthAssist AI",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Inject CSS.
    st.markdown(_CSS, unsafe_allow_html=True)

    # Initialise session state (no-op after first render).
    _init_session()

    # Sidebar.
    _render_sidebar()

    # App header.
    st.markdown(
        '<div class="app-header"><span style="font-size:2rem;">🏥</span>'
        "<h1>HealthAssist AI</h1></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="app-subtitle">'
        "Ask me about nutrition, lifestyle, preventive care, first aid, and general wellness. "
        "I am not a doctor and cannot diagnose or prescribe."
        "</p>",
        unsafe_allow_html=True,
    )

    # Welcome message on empty session.
    if not st.session_state.messages:
        st.info(
            "👋 **Welcome!** Ask me a health question to get started — for example:\n\n"
            "- *How much water should I drink each day?*\n"
            "- *What are the early signs of high blood pressure?*\n"
            "- *How can I improve my sleep quality?*",
        )

    # Replay history.
    _replay_history()

    # Chat input (pinned to bottom by Streamlit).
    user_input = st.chat_input("Ask a health question…")

    if user_input and not st.session_state.get("is_streaming", False):
        st.session_state.is_streaming = True
        try:
            _handle_input(user_input.strip())
        finally:
            st.session_state.is_streaming = False
        st.rerun()
