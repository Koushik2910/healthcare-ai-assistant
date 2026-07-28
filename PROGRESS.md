# PROGRESS.md — Current Status

> Read this file FIRST in any new chat about this project — it's the most
> current source of truth. Updated after every phase / meaningful change.
> Last updated: end of Phase 2 + warnings/smoke-test follow-up.

## Status at a glance

| Phase | Status | Verified by user? |
|---|---|---|
| 1. Foundation (config, logging, exceptions, domain models) | ✅ Done | ✅ 45/45 tests passed |
| 2. LLM provider abstraction (Gemini/Groq/OpenRouter, retry, streaming) | ✅ Done | ✅ 74/74 tests passed, 0 warnings, live Gemini smoke test succeeded |
| Handoff docs (this trio: SKILLS.md, CLAUDE.md, PROGRESS.md) | ✅ Done | Not yet delivered as of writing this line |
| 3. Prompt architecture | ⬜ Not started | — |
| 4. Safety layers + ChatService (incl. auto-failover Gemini→Groq) | ⬜ Not started | — |
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
│   ├── prompts/    (empty package, Phase 3)
│   ├── safety/      (empty package, Phase 4)
│   ├── rag/         (empty package, Phase 5)
│   ├── services/    (empty package, Phase 4 — ChatService goes here)
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
    └── test_llm_factory.py
```

**Test count: 74 passed, 0 warnings** (as of the last user-run `pytest -v`,
after the `filterwarnings` fix in `pyproject.toml`).

**Live verification:** `scripts/smoke_test_llm.py` run by the user against
the real Gemini API succeeded — confirms `GeminiProvider`'s SDK call shape
(`client.aio.models.generate_content_stream(...)`) is correct against the
installed `google-genai` version. Groq/OpenRouter live calls have NOT yet
been smoke-tested by the user (only tested against fakes) — worth doing
before relying on the fallback path for real.

## Known minor/cosmetic items (not bugs, not blocking)

- `google-genai` logs `AFC is enabled with max remote calls: 10` on every
  call — harmless SDK default logging. Planned to quiet this logger
  (alongside the existing httpx/chromadb suppression list in
  `src/utils/logging.py`) when that file is next touched in Phase 3 —
  not urgent enough for its own patch.

## Next up (start here in a new chat)

**Phase 3: Prompt architecture.** Not started. Scope per the original
plan: system prompt, safety/scope prompt, healthcare disclaimer,
formatting prompt, guardrail instructions, prompt templates — each
composable and independently testable, with WHY explained for each
before writing code, per SKILLS.md process rules.

Also outstanding from Phase 2's discussion: **automatic Gemini→Groq
failover** was agreed but not yet built — fold it into Phase 4
(ChatService), not Phase 3.

## How this file gets kept current

Claude updates this file (the status table, the file tree, the test
count, "Next up") every time a phase completes or a meaningful decision
is made — as its own small patch, same delivery workflow as any other
change (see SKILLS.md). If this file and the actual repo ever disagree,
the repo is the truth and this file needs fixing.
