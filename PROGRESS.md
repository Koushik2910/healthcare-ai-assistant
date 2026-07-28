# PROGRESS.md — Current Status

> Read this file FIRST in any new chat about this project — it's the most
> current source of truth. Updated after every phase / meaningful change.
> Last updated: Phase 3 shipped (prompt architecture).

## Status at a glance

| Phase | Status | Verified by user? |
|---|---|---|
| 1. Foundation (config, logging, exceptions, domain models) | ✅ Done | ✅ 45/45 tests passed |
| 2. LLM provider abstraction (Gemini/Groq/OpenRouter, retry, streaming) | ✅ Done | ⚠️ See note below |
| Handoff docs (this trio: SKILLS.md, CLAUDE.md, PROGRESS.md) | ✅ Done | Not yet delivered as of writing this line |
| 3. Prompt architecture | ✅ Done | — |
| 4. Safety layers + ChatService (incl. auto-failover Gemini→Groq) | ✅ Done | ✅ 241/241 tests passed |
| 5. RAG (corpus, ingestion, retrieval, citations) | ⬜ Not started | — |
| 6. Streamlit UI + custom CSS | ⬜ Not started | — |
| 7. Tests + adversarial eval harness | ⬜ Not started | — |
| 8. README/ARCHITECTURE/LOGIC docs, deck, demo video, submit | ⬜ Not started | — |
| 9. FastAPI adapter (portfolio, post-submission) | ⬜ Not started | — |
| 10. Next.js front-end + Vercel deploy (portfolio, post-submission) | ⬜ Not started | — |

**Assignment deadline: 48 hours from email receipt (received ~12:39 PM
the day this project started).** Phases 1–8 are the graded critical path.
Phases 9–10 are portfolio-only, done after submission, no deadline
pressure.

## ⚠️ Phase 2 verification — exact state (read carefully, don't overstate this)

- ✅ **Confirmed by user:** `pytest -v` → 74/74 passed, on a run that
  showed 164 warnings (all traced to third-party libraries: one
  `google-genai` typing deprecation, the rest `pytest-asyncio` using an
  event-loop-policy API deprecated in the user's Python 3.14).
- ✅ **Confirmed by user:** `scripts/smoke_test_llm.py` against the real
  Gemini API — real streamed response, 167 characters, 1975 ms. This is
  the one that actually proves `GeminiProvider`'s SDK call shape is
  correct.
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
- 🚢 **Fix shipped:** `tests/test_prompts.py` one-line assertion fix. Not
  yet re-run by user — do not claim 128/128 until user pastes confirmation.

## What's actually on disk right now

Project root: `C:\Users\Azuro\healthcare-ai-assistant`

```
healthcare-ai-assistant/
├── .env                      (user's real keys — never shipped by Claude)
├── .env.example
├── .gitignore
├── CLAUDE.md                 (this trio)
├── SKILLS.md                 (this trio)
├── PROGRESS.md               (this trio — this file)
├── README.md
├── pyproject.toml            (pytest/ruff/mypy config, incl. warning filters)
├── requirements.txt
├── requirements-dev.txt
├── data/
│   ├── knowledge_base/.gitkeep   (empty — Phase 5 fills this)
│   ├── sessions/.gitkeep
│   └── chroma/                   (created by preflight.py, empty)
├── docs/.gitkeep
├── scripts/
│   ├── preflight.py           (startup diagnostic — user has run, passes)
│   └── smoke_test_llm.py      (live single-call SDK check — user has run
│                                against Gemini, succeeded: 167 chars, 1975ms)
├── src/
│   ├── config/settings.py     (typed settings incl. gemini/groq/openrouter)
│   ├── models/
│   │   ├── chat.py            (Message, Conversation, ChatResponse, etc.)
│   │   ├── safety.py          (SafetyVerdict, RiskCategory, etc.)
│   │   ├── rag.py             (KBDocument, Chunk, RetrievalResult, etc.)
│   │   └── llm.py             (ProviderName, GenerationResult)
│   ├── llm/
│   │   ├── base.py            (LLMProvider ABC)
│   │   ├── retry.py           (stream_with_retry — first-chunk-only retry)
│   │   ├── gemini_provider.py
│   │   ├── openai_compatible.py  (shared Groq/OpenRouter implementation)
│   │   ├── groq_provider.py
│   │   ├── openrouter_provider.py
│   │   └── factory.py         (get_llm() — the one switch point)
│   ├── prompts/
│   │   ├── __init__.py        (re-exports public API)
│   │   ├── blocks.py          (4 composable block functions)
│   │   ├── builder.py         (PromptBuilder + PromptContext)
│   │   └── templates.py       (TemplateLibrary — static refusal strings)
│   ├── safety/
│   │   ├── __init__.py        (re-exports InputGuard, OutputGuard)
│   │   ├── input_guard.py     (deterministic pre-model screener, 7 rule sets)
│   │   └── output_guard.py    (post-model validator, sev-2 + sev-3 rules)
│   ├── rag/         (empty package, Phase 5)
│   ├── services/
│   │   ├── __init__.py        (re-exports ChatService)
│   │   └── chat_service.py    (full turn orchestration + Gemini→Groq failover)
│   ├── ui/          (empty package, Phase 6)
│   └── utils/
│       ├── exceptions.py      (HealthAssistantError hierarchy)
│       └── logging.py         (redaction, correlation IDs, JSON formatter)
└── tests/
    ├── conftest.py             (hermetic env isolation, build_settings fixture)
    ├── fakes.py                (FakeLLMProvider, FakeGeminiClient, FakeOpenAIClient)
    ├── test_config.py
    ├── test_models.py
    ├── test_observability.py
    ├── test_llm_base.py
    ├── test_llm_retry.py
    ├── test_llm_providers.py
    ├── test_llm_factory.py
    ├── test_prompts.py         (Phase 3 — 54 unit tests)
    ├── test_safety.py          (Phase 4 — InputGuard + OutputGuard)
    └── test_chat_service.py    (Phase 4 — ChatService, no network)
```

**Test and live-verification status:** see the "Phase 2 verification —
exact state" section above. Don't restate a warning/test count here too —
one place to update, not two.

## Known minor/cosmetic items (not bugs, not blocking)

- `google-genai` logs `AFC is enabled with max remote calls: 10` on every
  call — harmless SDK default logging. Planned to quiet this logger
  (alongside the existing httpx/chromadb suppression list in
  `src/utils/logging.py`) when that file is next touched in Phase 3 —
  not urgent enough for its own patch.

## Operational notes for working across sessions

- **The user's Claude session has a token/time budget that resets every
  ~3 hours.** When a session is running low, the working pattern is:
  finish the current small task cleanly, update this file, stop — not
  push into a big multi-file phase that might get cut off half-written.
  A half-finished phase is much more expensive to untangle later than
  waiting for the reset.
- **This file (`PROGRESS.md`) must be updated after every meaningful
  change Claude makes in this project — not just at phase boundaries.**
  A one-file patch, a bug fix, a config tweak: all of it gets reflected
  here before the turn ends. See the closing section of this file for
  the exact rule.
- The file tree below is maintained by hand and can drift. If it and the
  real repo ever disagree, trust the repo — run
  `Get-ChildItem -Recurse -File` to verify before relying on the tree.
- Delivery convention (partial ZIP + robocopy/Copy-Item commands, never
  a full re-zip) is in `SKILLS.md` — do not rediscover this, just follow
  it.

## Quick resume commands (PowerShell)

```powershell
cd C:\Users\Azuro\healthcare-ai-assistant
venv\Scripts\activate

# Confirm current state before doing anything new
pytest -v
python scripts\preflight.py

# Confirm live provider connectivity (uses real .env keys, makes a real call)
python scripts\smoke_test_llm.py
python scripts\smoke_test_llm.py --provider groq
```

## Phase 4 verification — exact state

- ✅ **Confirmed by user:** 241/241 passed after two fix rounds.
  Round 1 (38 failures): `re.IGNORECASE` as `pos=2` in `.search()`;
  missing `import re` in `chat_service.py`; 3 narrow patterns.
  Round 2 (7 failures): `can_i_take_together`, `prescribe_me`,
  `disclaimer` (Symptoms→sympt\w+), `suffering` adjective gap,
  `i_recommend_drug` (you take gap). All resolved.

## Next up (start here in a new chat)

**Phase 5 — RAG (corpus, ingestion, retrieval, citations)**

1. Write 12–15 short public-domain health documents into
   `data/knowledge_base/` (MedlinePlus/CDC/NIH summaries, original content).
   Each document needs `DocumentLicence` + provenance, enforced by the
   `KBDocument` model already in `src/models/rag.py`.
2. `src/rag/ingestion.py` — chunk documents, embed with
   `all-MiniLM-L6-v2`, persist to ChromaDB.
3. `src/rag/retriever.py` — query ChromaDB, apply score threshold,
   return `RetrievalResult`. Plugs into `ChatService._retriever`.
4. `scripts/ingest.py` — one-command CLI to (re)build the knowledge base.
5. Tests against fake ChromaDB client.

Also still outstanding from Phase 2 (low priority, do before Phase 8):
`requirements.lock.txt` (exact resolved versions).

## How this file gets kept current (standing rule)

**Claude updates this file after every change made in this project —
not only at phase boundaries.** A single bug fix, a config tweak, a
one-file patch, a correction like the one made to this file just now:
all of it gets reflected here, before the turn ends, as part of
delivering the change — not as a separate follow-up the user has to ask
for. Specifically, every update should refresh: the status table, the
verification-state section (marking things confirmed vs. shipped-but-
unconfirmed — never round up), the file tree if it changed, and "Next
up." If this file and the actual repo ever disagree, the repo is the
truth and this file needs fixing — that mismatch is itself a bug.
