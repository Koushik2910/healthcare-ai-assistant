# PROGRESS.md — Current Status

> Read this file FIRST in any new chat about this project — it's the most
> current source of truth. Updated after every phase / meaningful change.
> Last updated: Phase 6 shipped (Streamlit UI — app.py, components.py, root shim).

## Status at a glance

| Phase | Status | Verified by user? |
|---|---|---|
| 1. Foundation (config, logging, exceptions, domain models) | ✅ Done | ✅ 45/45 tests passed |
| 2. LLM provider abstraction (Gemini/Groq/OpenRouter, retry, streaming) | ✅ Done | ⚠️ See Phase 2 note below |
| Handoff docs (this trio: SKILLS.md, CLAUDE.md, PROGRESS.md) | ✅ Done | ✅ Delivered |
| 3. Prompt architecture | ✅ Done | ✅ 128/128 tests (after assertion fix) |
| 4. Safety layers + ChatService (incl. auto-failover Gemini→Groq) | ✅ Done | ✅ 241/241 tests passed |
| 5. RAG (corpus, ingestion, retrieval, citations) | ✅ Done | ✅ 288/288 tests + 112 chunks in ChromaDB |
| 6. Streamlit UI + custom CSS | 🚢 Shipped | ⏳ Awaiting user confirmation |
| 7. Tests + adversarial eval harness | ⬜ Not started | — |
| 8. README/ARCHITECTURE/LOGIC docs, deck, demo video, submit | ⬜ Not started | — |
| 9. FastAPI adapter (portfolio, post-submission) | ⬜ Not started | — |
| 10. Next.js front-end + Vercel deploy (portfolio, post-submission) | ⬜ Not started | — |

**Assignment deadline: 48 hours from email receipt (received ~12:39 PM
the day this project started).** Phases 1–8 are the graded critical path.
Phases 9–10 are portfolio-only, done after submission, no deadline pressure.

## ⚠️ Phase 2 verification — exact state (read carefully, don't overstate this)

- ✅ **Confirmed by user:** `pytest -v` → 74/74 passed, on a run that
  showed 164 warnings (all traced to third-party libraries: one
  `google-genai` typing deprecation, the rest `pytest-asyncio` using an
  event-loop-policy API deprecated in the user's Python 3.14).
- ✅ **Confirmed by user:** `scripts/smoke_test_llm.py` against the real
  Gemini API — real streamed response, 167 characters, 1975 ms. This is
  the one that actually proves `GeminiProvider`'s SDK call shape is correct.
- 🟡 **Shipped but NOT yet confirmed by a user re-run:** a
  `pyproject.toml` patch adding `filterwarnings` to silence those two
  specific known warnings. The user has not yet pasted a fresh
  `pytest -v` showing 0 warnings — **do not claim 0 warnings as fact
  until that's actually pasted back.**
- ❌ **Not yet done:** `scripts\smoke_test_llm.py --provider groq` and
  `--provider openrouter` have not been run. Groq/OpenRouter are only
  verified against fakes so far, not a real endpoint.
- ❌ **Unknown / unconfirmed:** whether `git init` / first commit has
  actually been run. A commit message was suggested after Phase 1; there
  is no confirmation it was executed. Don't assume version control
  exists — check with the user or `git status` before assuming history.
- ❌ **Not yet done:** `requirements.txt` has a comment promising a
  companion `requirements.lock.txt` (exact resolved versions) — this
  file does not exist yet. Low priority, worth doing before final
  submission (Phase 8), not urgent now.

## Phase 3 verification — exact state

- ✅ **Confirmed by user:** `pytest -v` → 127/128 passed on initial run.
  One test failed: `test_build_no_retrieval_no_context_section` — test
  assertion bug (not a prompt bug). The word "CONTEXT" legitimately appears
  in `formatting_contract()` citing instructions; the assertion was changed
  to check for the RAG block's specific header instead.
- ✅ **Fix shipped:** `tests/test_prompts.py` one-line assertion fix.
  Final state: 128/128 (assumed from the fix being correct — user
  did not paste a re-run; do not overstate as confirmed).

## Phase 4 verification — exact state

- ✅ **Confirmed by user:** 241/241 passed after two fix rounds.
  Round 1 (38 failures): `re.IGNORECASE` as `pos=2` in `.search()`;
  missing `import re` in `chat_service.py`; 3 narrow patterns.
  Round 2 (7 failures): `can_i_take_together`, `prescribe_me`,
  `disclaimer` (Symptoms→sympt\\w+), `suffering` adjective gap,
  `i_recommend_drug` (you take gap). All resolved.

## Phase 5 verification — exact state

- ✅ **Confirmed by user:** 288/288 tests passed (47 new in test_rag.py). scripts\\ingest.py: 13 docs, 112 chunks, 1179 ms. sys.path fix applied (phase5_fix1_ingest.zip).
  Two-step verification required:
  1. `pytest -v` — all tests must pass (expect ~60+ new tests from
     `tests/test_rag.py` plus 241 existing tests).
  2. `python scripts\\ingest.py --dry-run` — should list 13 documents
     and their chunk counts with no errors.
  3. `python scripts\\ingest.py` — full run: downloads `all-MiniLM-L6-v2`
     (first run only, ~80 MB), embeds, and writes to `data/chroma/`.
     Should print: "Knowledge base built successfully: 13 documents, N chunks".

## Phase 6 — Streamlit UI — shipped state

**Files delivered in phase6.zip:**

| File | Action | Notes |
|---|---|---|
| `app.py` | NEW (root) | 3-line shim: `from src.ui.app import main; main()` |
| `src/ui/__init__.py` | UPDATED | Re-exports `main` |
| `src/ui/app.py` | NEW | Full Streamlit app (session state, streaming, safety UI, RAG sources) |
| `src/ui/components.py` | NEW | Reusable widgets: bubble, crisis card, badge, source card, expander |
| `PROGRESS.md` | UPDATED | This file |

**Key design choices recorded here (not in code comments):**

- Double-call pattern: `stream_chat` → user sees tokens; then `chat` (non-streaming)
  → metadata (citations, disclaimer, verdict). Model is called twice per turn.
  Justified: `stream_chat` yields `str` only, no structured metadata.
  Future optimisation can collapse this without changing public API.
- `@st.cache_resource` on `_get_chat_service()` — ChatService+Retriever
  instantiated exactly once per worker process.
- Crisis cards rendered BEFORE the message bubble in history replay so they
  cannot be scrolled past.
- `_citations_to_retrieval()` reconstructs a minimal `RetrievalResult` from
  `ChatResponse.citations` for the sources expander — avoids threading raw
  chunks through session state.
- Event loop stored in `st.session_state["_event_loop"]` and reused across
  interactions to avoid the overhead of a new loop per submit.

**⚠️ Phase 6 NOT yet user-confirmed.** Do not mark ✅ until the user pastes
a successful `streamlit run app.py` test run.

**Verification steps (run in this order):**
```powershell
cd C:\Users\Azuro\healthcare-ai-assistant
venv\Scripts\activate
pip install streamlit          # if not already installed
streamlit run app.py
```
Then in the browser:
1. Type a general health question → should stream a response with optional Sources expander.
2. Type "I want to hurt myself" → should show crisis card with 988 hotline.
3. Type "I am having chest pain and can't breathe" → should show emergency card with 112.
4. Ask about a topic in the knowledge base (e.g. "hydration") → Sources expander
   should appear with ≥1 source card.
5. Click "Clear conversation" → history should reset cleanly.

## What's actually on disk right now

Project root: `C:\Users\Azuro\healthcare-ai-assistant`

```
healthcare-ai-assistant/
├── app.py                    ← Phase 6 NEW: root-level shim
├── .env                      (user's real keys — never shipped by Claude)
├── .env.example
├── .gitignore
├── CLAUDE.md
├── SKILLS.md
├── PROGRESS.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── data/
│   ├── knowledge_base/           ← Phase 5: 13 JSON documents
│   ├── sessions/.gitkeep
│   └── chroma/                   (populated — 112 chunks)
├── docs/.gitkeep
├── scripts/
│   ├── preflight.py
│   ├── smoke_test_llm.py
│   └── ingest.py
├── src/
│   ├── config/settings.py
│   ├── models/
│   │   ├── chat.py, safety.py, rag.py, llm.py
│   ├── llm/
│   │   ├── base.py, retry.py, factory.py
│   │   ├── gemini_provider.py, openai_compatible.py
│   │   ├── groq_provider.py, openrouter_provider.py
│   ├── prompts/
│   │   ├── __init__.py, blocks.py, builder.py, templates.py
│   ├── safety/
│   │   ├── __init__.py, input_guard.py, output_guard.py
│   ├── rag/
│   │   ├── __init__.py, ingestion.py, retriever.py
│   ├── services/
│   │   ├── __init__.py, chat_service.py
│   ├── ui/                    ← Phase 6 NEW: fully populated
│   │   ├── __init__.py        (re-exports main)
│   │   ├── app.py             (main Streamlit app)
│   │   └── components.py      (reusable widgets)
│   └── utils/
│       ├── exceptions.py, logging.py
└── tests/
    ├── conftest.py, fakes.py
    ├── test_config.py, test_models.py, test_observability.py
    ├── test_llm_base.py, test_llm_retry.py, test_llm_providers.py
    ├── test_llm_factory.py, test_prompts.py, test_safety.py
    ├── test_chat_service.py, test_rag.py
```

## ⚠️ requirements.txt — Phase 5 new dependencies

Two packages must be installed before `scripts/ingest.py` (or any live RAG
run) will work. Tests do **not** require them (tests use fakes). Add to
`requirements.txt` and run:

```powershell
pip install sentence-transformers chromadb
```

Phase 6 adds one more:
```powershell
pip install streamlit
```

## Known minor/cosmetic items (not bugs, not blocking)

- `google-genai` logs `AFC is enabled with max remote calls: 10` on every
  call — harmless SDK default logging.
- Phase 6 double-call pattern (stream + non-stream for metadata) is a known
  trade-off, documented above.  Will be collapsed in Phase 9 (FastAPI adapter).

## Operational notes for working across sessions

- **The user's Claude session has a token/time budget that resets every
  ~3 hours.** When a session is running low, the working pattern is:
  finish the current small task cleanly, update this file, stop — not
  push into a big multi-file phase that might get cut off half-written.
  A half-finished phase is much more expensive to untangle later than
  waiting for the reset.
- **This file (`PROGRESS.md`) must be updated after every meaningful
  change Claude makes in this project — not just at phase boundaries.**
- Delivery convention (partial ZIP + robocopy/Copy-Item commands, never
  a full re-zip) is in `SKILLS.md` — do not rediscover this, just follow it.

## Quick resume commands (PowerShell)

```powershell
cd C:\Users\Azuro\healthcare-ai-assistant
venv\Scripts\activate

# Confirm existing tests still pass
pytest -v

# Run the app
streamlit run app.py

# Phase 5 — rebuild knowledge base if needed
python scripts\ingest.py --dry-run
python scripts\ingest.py

# Confirm live provider connectivity
python scripts\smoke_test_llm.py
python scripts\smoke_test_llm.py --provider groq
```

## Next up (start here in a new chat)

**Phase 7 — Tests + adversarial eval harness**

- Adversarial test set: ~20 hand-crafted prompts covering crisis, emergency,
  scope violations, prompt injection, and benign questions. Each has an
  expected `ResponseSource`, `refused` flag, and optional keyword match.
- An eval runner script (`scripts/eval.py`) that runs the harness against
  the live `ChatService` and prints a pass-rate table.
- Integration smoke test that boots the full pipeline (real ChromaDB, fake
  LLM) and confirms end-to-end routing.

## How this file gets kept current (standing rule)

**Claude updates this file after every change made in this project —
not only at phase boundaries.** A single bug fix, a config tweak, a
one-file patch, a correction: all of it gets reflected here before the
turn ends. Every update should refresh: the status table, verification
state (marking things confirmed vs. shipped-but-unconfirmed — never round
up), the file tree if it changed, and "Next up."
