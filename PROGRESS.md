# PROGRESS.md — Current Status

> Read this file FIRST in any new chat about this project — it's the most
> current source of truth. Updated after every phase / meaningful change.
> Last updated: Post-submission testing & fixing session (Phase 8 + all bug fixes pushed).

## Status at a glance

| Phase | Status | Verified by user? |
|---|---|---|
| 1. Foundation (config, logging, exceptions, domain models) | ✅ Done | ✅ 45/45 tests passed |
| 2. LLM provider abstraction (Gemini/Groq/OpenRouter, retry, streaming) | ✅ Done | ✅ Smoke test confirmed |
| Handoff docs (SKILLS.md, CLAUDE.md, PROGRESS.md) | ✅ Done | ✅ Delivered |
| 3. Prompt architecture | ✅ Done | ✅ 128/128 tests |
| 4. Safety layers + ChatService (Gemini→Groq auto-failover) | ✅ Done | ✅ 241/241 tests |
| 5. RAG (corpus, ingestion, retrieval, citations) | ✅ Done | ✅ 288/288 tests + 112 chunks |
| 6. Streamlit UI + custom CSS | ✅ Done | ✅ All browser scenarios confirmed |
| 7. Adversarial eval harness | ✅ Done | ✅ 363/363 tests + 20/20 eval.py PASS |
| 8. README, ARCHITECTURE, EVALUATION docs | ✅ Done | ✅ Pushed to GitHub |
| Post-submission: Bug fixes + cascade + UI improvements | ✅ Done | ✅ 363/363 tests + pushed |
| 9. FastAPI adapter (portfolio, post-submission) | ⬜ Not started | — |
| 10. Next.js front-end + Vercel deploy (portfolio, post-submission) | ⬜ Not started | — |

**Test state: 363/363 pytest tests passing. 20/20 live eval PASS.**
**Repo state: Clean. All fixes pushed to GitHub (commit ff633a5).**

---

## Post-submission fixes (all in commit ff633a5)

This session was a live testing + fixing run. All issues found and fixed:

### Fix 1 — Double disclaimer (UI bug)
**Root cause:** `_FORMATTING_BLOCK` instructed the LLM to append an italic
`---…---` disclaimer footer. The LLM obeyed, so it appeared in `result.text`
rendered by `st.markdown()`. Then the UI disclaimer pill rendered a second one.

**Fix:**
- `src/prompts/blocks.py` — DISCLAIMER section now tells the LLM explicitly
  NOT to append a footer ("the UI displays a disclaimer automatically").
- `src/services/chat_service.py` — `_strip_llm_disclaimer()` strips any
  trailing disclaimer block defensively as a belt-and-suspenders measure.
- `tests/test_prompts.py` — updated assertions to match new wording.

---

### Fix 2 — "Service busy" second message (double LLM call bug)
**Root cause:** After `stream_chat` finished, `_handle_input` made a second
`svc.chat()` call to get `ChatResponse` metadata. This hit Gemini again
immediately, triggering `LLMRateLimitError` on the free tier (10 req/min).
The second call's failure message appeared as a new chat bubble.

**Fix:**
- `src/ui/app.py` — Eliminated the second `svc.chat()` call entirely.
  New `_build_response_from_stream()` helper synthesises a `ChatResponse`
  from the already-streamed text using the same disclaimer/refusal-detection
  logic — zero extra API calls per turn.

---

### Fix 3 — Failover not triggering (async generator try/except bug)
**Root cause:** `_stream_with_failover` was an async generator (uses `yield`).
Python's `try/except` inside an async generator does NOT catch exceptions
raised by iterated sub-generators — the exception escapes the generator frame
before the `except` clause can fire. So `LLMRateLimitError` from Gemini was
never caught and failover never triggered.

**Fix:**
- `src/services/chat_service.py` — Converted `_stream_with_failover` from
  async generator to plain `async def` that RETURNS an `AsyncIterator`.
  In a plain coroutine, `try/except` works normally. Probes the primary
  stream for first chunk, catches errors, returns fallback's iterator instead.
  Call site changed to `async for chunk in await self._stream_with_failover(...)`.

---

### Fix 4 — Three-provider cascade (Gemini → Groq → OpenRouter)
**What was built:**
- `src/services/chat_service.py` — `_build_fallback_chain()` replaces
  `_build_fallback()`. Builds `[GroqProvider, OpenRouterProvider]` based on
  which keys exist in `.env`. Empty list = no failover.
- Both `_stream_with_failover` and `_call_with_failover` iterate the full
  chain, catching `LLMError` (base class, not just subclasses) so all failure
  modes (rate limit, timeout, response error) trigger failover.
- `OpenRouterProvider` added as tier-2 fallback using existing `$50` credit
  balance on `google/gemini-2.5-flash` via OpenRouter.
- **No `.env` changes ever needed** — `LLM_PROVIDER=gemini` stays. The app
  reads all three keys at startup and builds the chain automatically.

**Cascade order:** Gemini (primary, 20 req/day free) → Groq (tier-1, 6000
req/day free) → OpenRouter (tier-2, paid, ~₹0.08/message, $50 balance).

- `tests/test_chat_service.py` — Updated 5 tests from `svc._fallback = x`
  to `svc._fallbacks = [x]` (new list API).

---

### Fix 5 — Missing OpenRouterProvider import (NameError)
**Root cause:** `OpenRouterProvider` was used in `_build_fallback_chain` but
never imported in `chat_service.py`. Built fine in isolation (diagnostic
script imported it explicitly) but failed inside Streamlit's `@st.cache_resource`.

**Fix:** Added `from src.llm.openrouter_provider import OpenRouterProvider`
to imports in `chat_service.py`.

---

### Fix 6 — Missing LLMError import (NameError in cascade)
**Root cause:** Changed `except (LLMRateLimitError, LLMTimeoutError)` to
`except LLMError` but forgot to add `LLMError` to the imports block.

**Fix:** Added `LLMError` to the `from src.utils.exceptions import (...)` block.

---

### Fix 7 — Output guard false positives blocking good responses
Two separate rules were firing incorrectly:

**7a — `output.diagnosis.you_have` on empathy phrasing:**
Groq/Llama opens cancer-related responses with "thinking you might have cancer
can be frightening" — the phrase `"you might have cancer"` matched the severity-3
rule. Fix: added `(?<!thinking )` negative lookbehind so the rule only fires
when NOT preceded by `"thinking "`.

**7b — `output.diagnosis.you_are_suffering` on educational phrasing:**
Groq uses "you are experiencing symptoms of a condition" — legitimate educational
language that matched the rule. Fix: added `(?!\s+symptoms\b)` negative lookahead
on `experiencing` only, and added `of` as alternative to `from` for `"showing
signs of"`.

**Files:** `src/safety/output_guard.py`

---

### Fix 8 — STREAM_BLOCKED_SENTINEL for clean output guard UI
**Root cause:** When the output guard fired post-stream, the old code yielded
`f"\n\n⚠️ {_OUTPUT_BLOCKED_MESSAGE}"` appended onto the stream — users saw
the good response followed by a warning stitched to it.

**Fix:**
- `src/services/chat_service.py` — Output guard now yields
  `f"{STREAM_BLOCKED_SENTINEL}{_OUTPUT_BLOCKED_MESSAGE}"` as the ONLY final
  chunk (`STREAM_BLOCKED_SENTINEL = "\x00BLOCKED\x00"`).
- `src/ui/app.py` — `_stream_response` detects the sentinel, clears the
  placeholder, and shows only the blocked message cleanly.

---

### Fix 9 — Consistent heading font sizes across providers
**Root cause:** Groq/OpenRouter responses use `##` and `###` markdown headings.
Streamlit renders these as full H2/H3 HTML inside chat bubbles — large and
inconsistent vs Gemini responses that use prose.

**Fix:**
- `src/ui/app.py` — `_normalise_headings()` converts `## Heading` →
  `**Heading**` in streamed text before rendering (belt-and-suspenders).
- CSS block — Added heading size overrides capping all headings inside
  `[data-testid="stChatMessage"]` to `1.0rem` max.

---

### Fix 10 — OpenRouter provider deferred HTTP client construction
**Root cause:** `AsyncOpenAI` client was constructed eagerly during
`@st.cache_resource` init, sometimes attaching its aiohttp session to the wrong
asyncio event loop, causing intermittent build failures.

**Fix:** `src/llm/openrouter_provider.py` — Pass `http_client=None` to defer
httpx session creation until first actual API call.

---

### Fix 11 — Live provider indicator in sidebar + clear terminal logs
**What was added:**
- `src/ui/app.py` — Sidebar "Powered by Gemini · Groq fallback" replaced with
  dynamic colour-coded indicator that updates after each response:
  - 🟢 Gemini 2.5 Flash (primary)
  - 🟡 Groq · Llama 3.3 70B (tier-1 fallback)
  - 🔵 OpenRouter · Gemini Flash (tier-2 fallback)
- `src/services/chat_service.py` — Cascade startup logs:
  `[CASCADE] Provider chain ready: GEMINI → GROQ → OPENROUTER`
  Per-turn logs: `[CASCADE] GEMINI unavailable (LLMRateLimitError) — trying GROQ next.`
  Success log: `[CASCADE] ✓ Serving from GROQ (tier-1 fallback)`
- `_ACTIVE_PROVIDER` module-level var written immediately when provider is
  chosen; read by sidebar on next render via `st.session_state.active_provider`.

---

## Gemini free tier reality (learned during testing)

- Free tier limit: **20 requests per day** (not 10/min as originally thought)
- The double-call bug burned through this at 2x speed during testing
- **Groq free tier:** 6,000 req/day — effectively unlimited for demo
- **OpenRouter:** $50 balance, ~₹0.08/message on Gemini Flash — safety net
- Daily Gemini reset: midnight Pacific (~5:30 AM IST)
- **For demo:** reset to `LLM_PROVIDER=gemini` after Gemini quota resets
  so the grader sees the intended primary provider

---

## Current .env configuration

```
LLM_PROVIDER=gemini          ← never change this
GEMINI_API_KEY=...           ✓ confirmed working
GROQ_API_KEY=...             ✓ confirmed working (smoke test: 308 chars, 600ms)
OPENROUTER_API_KEY=...       ✓ key set, builds OK
OPENROUTER_MODEL=google/gemini-2.5-flash
```

---

## Application scope (settled — do not reopen)

The assistant answers: nutrition, lifestyle, preventive care, first aid,
general wellness, health education.

It does NOT answer: specific medication queries ("dolo 650 tablet uses" →
blocked), diagnoses, prescriptions, or off-topic questions (stocks, MongoDB).

This is correct and intentional. Interview answer: *"I deliberately scoped
this to general wellness because a healthcare AI that gives medication advice
without guardrails is dangerous. The guardrails demonstrate responsible AI."*

---

## What's on disk right now

```
healthcare-ai-assistant/
├── app.py                        ← root shim (delegates to src/ui/app.py)
├── .streamlit/config.toml        ← headless=true, fileWatcherType=none
├── CLAUDE.md                     ← updated this session
├── SKILLS.md
├── PROGRESS.md                   ← this file, updated this session
├── README.md                     ← Phase 8 rewrite
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
│
├── data/
│   ├── knowledge_base/           ← 13 JSON documents (Phase 5)
│   └── chroma/                   ← 112 chunks populated (Phase 5)
│
├── docs/
│   ├── ARCHITECTURE.md           ← Phase 8
│   └── EVALUATION.md             ← Phase 8
│
├── scripts/
│   ├── preflight.py
│   ├── smoke_test_llm.py
│   ├── ingest.py
│   └── eval.py
│
├── src/
│   ├── config/settings.py
│   ├── models/ (chat, safety, rag, llm)
│   ├── llm/ (base, retry, factory, gemini, groq, openrouter, openai_compat)
│   ├── prompts/ (blocks, builder, templates)   ← blocks.py updated
│   ├── safety/ (input_guard, output_guard)     ← output_guard.py updated
│   ├── rag/ (ingestion, retriever)
│   ├── services/chat_service.py                ← major update this session
│   ├── eval/ (cases.py)
│   └── ui/ (app.py, components.py)             ← app.py major update
│
└── tests/
    ├── conftest.py, fakes.py
    ├── test_config.py, test_models.py, test_observability.py
    ├── test_llm_*.py (base, retry, providers, factory)
    ├── test_prompts.py                          ← updated this session
    ├── test_safety.py, test_rag.py
    ├── test_chat_service.py                     ← updated this session
    └── test_eval_harness.py
```

---

## Quick resume commands

```powershell
cd C:\Users\Azuro\healthcare-ai-assistant
venv\Scripts\activate

pytest                          # 363/363 expected
streamlit run app.py            # UI at http://localhost:8501
python scripts\eval.py          # 20/20 expected (needs GEMINI_API_KEY with quota)
```

---

## Still outstanding (nice-to-have, not blocking)

1. **Demo video** — 2-3 min screen recording: boot → ask question → crisis card
   → Sources expander → show cascade logs → run eval.py.
2. **Submission ZIP** — exclude `venv/`, `data/chroma/`, `.env`.
3. **Phase 9** — FastAPI adapter (post-submission portfolio).
4. **Phase 10** — Next.js front-end + Vercel deploy (post-submission portfolio).

---

## How this file gets kept current

Claude updates this file after every meaningful change — not only at phase
boundaries. If this file and the actual repo disagree, the repo is the truth.
