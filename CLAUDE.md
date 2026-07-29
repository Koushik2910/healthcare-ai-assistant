# CLAUDE.md — Project Context: Healthcare AI Assistant

> Read SKILLS.md (how to work), this file (what's true), and PROGRESS.md
> (what's done / what's next) — in that order — before taking any action
> in a new conversation about this project.

## What this is

A Healthcare AI Assistant built for two purposes at once:
1. A take-home assignment from **First500days** (AI Engineer, Contractual)
   — received via email from `hr@9872549.brevosend.com` (Juhi Agnihotri),
   48-hour deadline from receipt.
2. A **flagship portfolio project** for Koushik, who is transitioning from
   Senior QA Automation Engineer to AI Engineering roles. The portfolio
   aim outweighs the assignment once the assignment is safely submitted
   — see "UI vs. portfolio" decision below.

Scope: answers general healthcare questions (nutrition, lifestyle,
preventive care, general symptoms, first aid, medical education). Must
never diagnose, prescribe, or replace a doctor.

## ⚠️ Two source documents exist — do not conflate them

1. **A long "elite engineering team" meta-prompt** the user pasted first
   (ROLE / PROJECT OBJECTIVE / UI REQUIREMENTS / etc.). This sets
   **aspirational tone and quality bar** — "build like a Staff Engineer,"
   premium UI language, exhaustive feature wishlist. It is aspirational
   framing, not the grading instrument.
2. **The actual `AI_Engineer_Assignment.docx`** (attached to the hiring
   email, parsed programmatically by Claude early in this project). This
   contains the **real grading rubric** (the weight table in SKILLS.md),
   the actual deliverables list, the 48-hour deadline, the "Streamlit
   (Preferred)" tech stack note, and the "no copyrighted/proprietary
   datasets" restriction.

**When the two conflict — e.g., the meta-prompt implies UI is the top
priority, the docx's rubric weights it at only 10% — the docx wins.**

## Assignment source — verification flags raised (informational, not urgent)

Two things were flagged to the user once, not repeated since, and not
blocking work:
- The hiring email's sender domain (`hr@9872549.brevosend.com`, hosted on
  Brevo, a bulk-mail platform, with an unsubscribe footer) looks more like
  a mass-mail domain than a corporate one. The user was advised to
  independently verify First500days and Juhi Agnihotri (e.g. on LinkedIn)
  before investing further time.
- Contractual AI Engineer roles that open with a 48-hour take-home are
  worth confirming the comp band on before proceeding much further.

Neither has been revisited since being raised. A new chat should not
re-raise these unless new information surfaces — the user is aware and
has chosen to proceed.

## Person / environment

- Name: Koushik Gattu. Windows 11, local username `Azuro`.
- Project lives at: `C:\Users\Azuro\healthcare-ai-assistant`
- Python 3.14.4, venv at `venv\` inside the project folder.
- PowerShell only — see SKILLS.md workflow section.
- Other context (career background, other projects like CareerApex AI,
  LuxeLane) lives in Claude's memory system, not repeated here — this
  file is specific to this project only.

## Key architecture decisions (settled — do not reopen without explicit user request)

### 1. Streamlit for graded submission; React/Next.js post-submission only
The assignment docx says "Streamlit (Preferred)"; UI is only 10% of the
rubric. Business logic lives in a UI-agnostic `ChatService`, so a
FastAPI + Next.js client can be added later without touching logic.
→ **Phases 9–10** (FastAPI + Next.js) are post-submission portfolio work.

### 2. Three-provider cascade: Gemini → Groq → OpenRouter (AUTO-FAILOVER)
**This is fully implemented and live as of the post-submission fix session.**

- **Gemini** (primary): `google-genai` SDK, free tier, 20 req/day.
- **Groq** (tier-1 fallback): free tier, 6000 req/day, `llama-3.3-70b-versatile`.
  Uses `openai` SDK pointed at Groq's OpenAI-compatible endpoint.
- **OpenRouter** (tier-2 fallback): paid, `google/gemini-2.5-flash`,
  ~₹0.08/message, $50 balance. Same `OpenAICompatibleProvider` base class.

**Cascade is fully automatic** — `LLM_PROVIDER=gemini` in `.env` NEVER
changes. The app reads all three keys at startup, builds
`[GroqProvider, OpenRouterProvider]` as `_fallbacks`, and tries each in
order on `LLMError`. No `.env` editing needed for quota exhaustion.

**How failover works:**
- `_stream_with_failover` is a plain `async def` (NOT async generator)
  returning `AsyncIterator[str]`. This is critical — `try/except` inside
  an async generator does NOT catch sub-generator exceptions. Plain
  coroutines have normal try/except semantics.
- Probes each provider for first chunk; on `LLMError` tries next.
- `_call_with_failover` does the same for non-streaming path.
- `_ACTIVE_PROVIDER` module-level var is set immediately when a provider
  wins; read by UI sidebar to show live indicator.

**Terminal logs per turn:**
```
[CASCADE] GEMINI unavailable (LLMRateLimitError) — trying GROQ next.
[CASCADE] ✓ Serving from GROQ (tier-1 fallback)
```

**Startup log:**
```
[CASCADE] Provider chain ready: GEMINI → GROQ → OPENROUTER
```

**Sidebar indicator:** 🟢 Gemini / 🟡 Groq / 🔵 OpenRouter — updates after
each response via `st.session_state.active_provider`.

### 3. Retry stops at first streamed token — never mid-stream
Retrying after the model has already started talking would duplicate or
interleave output. This is a design choice, not a limitation. Do NOT fix.
See `src/llm/retry.py` docstring.

### 4. Error translation is duck-typed, not isinstance-based
Third-party SDK exception constructors change shape across versions.
Duck typing (status codes, message substrings) is more resilient and
testable. See `src/llm/openai_compatible.py` and `src/llm/gemini_provider.py`.

### 5. RAG corpus: small, self-written/public-domain only
13 documents, 112 chunks in ChromaDB. Assignment forbids copyrighted
datasets. `SOURCES.md` records provenance.

### 6. Guardrails return values (SafetyVerdict), never raise
See `src/models/safety.py`. This is what makes the Phase 7 adversarial
eval harness produce a measurable pass rate rather than untestable `except` blocks.

### 7. Single LLM call per turn (no double-call)
Originally, `stream_chat` streamed tokens and then `chat()` was called again
for metadata (citations, disclaimer, refused flag). This doubled API quota
usage and caused rate-limit errors on Gemini free tier.

**Current design:** `_build_response_from_stream()` in `app.py` synthesises
a `ChatResponse` from the already-streamed text using the same
disclaimer/refusal-detection logic. Zero extra API calls per turn.

### 8. Output guard sentinel for clean blocked responses
When the output guard fires post-stream, `stream_chat` yields
`STREAM_BLOCKED_SENTINEL + message` as the ONLY final chunk.
`_stream_response` in `app.py` detects the sentinel, clears the placeholder,
and shows only the blocked message — no partial streamed content visible.

`STREAM_BLOCKED_SENTINEL = "\x00BLOCKED\x00"`

### 9. Heading normalisation for consistent font sizes
`_normalise_headings()` in `app.py` converts `## Heading` → `**Heading**`
in streamed text before `st.markdown()` renders it. CSS also caps headings
inside chat messages to `1.0rem`. Prevents Groq/OpenRouter's `##` markdown
from rendering as large H2/H3 elements inside chat bubbles.

### 10. Output guard false-positive rules (tightened)
- `output.diagnosis.you_have`: Added `(?<!thinking )` negative lookbehind.
  Prevents "thinking you might have cancer can be frightening" from blocking.
- `output.diagnosis.you_are_suffering`: Added `(?!\s+symptoms\b)` lookahead
  on `experiencing` only. Prevents "experiencing symptoms of a condition"
  (legitimate educational phrasing) from blocking. Also added `of` as
  alternative to `from` in preposition group.

### 11. Application scope (do not expand without discussion)
**In scope:** nutrition, lifestyle, preventive care, first aid, general
wellness, health education, general health questions.

**Out of scope (input guard blocks these):** specific medication queries
("dolo 650 tablet uses", "eldoper tablet uses"), diagnoses, prescriptions,
financial advice, off-topic questions (stocks, tech, MongoDB etc.).

This is intentional. The guards demonstrate responsible AI deployment.
Interview answer: *"A healthcare AI that gives medication advice without
guardrails is dangerous. The input guard enforces the scope boundary."*

## Gemini free tier reality

- **20 requests per day** (not 10/min) on `gemini-2.5-flash` free tier
- Quota is per Google Cloud project/account, NOT per API key
- Generating a new key under the same account does NOT reset quota
- Daily reset: midnight Pacific (~5:30 AM IST)
- **During testing:** switch to `LLM_PROVIDER=groq` temporarily; revert
  to `gemini` before demo/submission so grader sees intended primary provider
- The cascade handles quota exhaustion automatically — no manual `.env` edits

## Constraints Claude must respect in this project

- Claude's sandbox has **no network access** — cannot `pip install` or make
  real API calls. All SDK behavior verification happens on the user's machine.
- Every ZIP shipped is a **partial patch**, not a full re-zip — the user's
  `.env`, `venv/`, `data/chroma/` must survive every update untouched.
- **Safety files** (`src/safety/input_guard.py`, `src/safety/output_guard.py`)
  must not be modified unless the issue is specifically in those files.
- **363 tests must pass after every fix.** Never claim something works until
  the user confirms `pytest -v` output.
- Deliver files as downloadable ZIPs — never inline code blocks for file
  content (copy-paste causes U+00A0 encoding errors on Windows).

## Known cosmetic warnings (do not fix unless asked)

These appear in the terminal but do not affect functionality:
- `Warning: You are sending unauthenticated requests to the HF Hub` —
  harmless, no HF_TOKEN needed for `all-MiniLM-L6-v2`.
- `ChromaDB _EF DeprecationWarning` — harmless, future chromadb version fixes.
- `ConnectionResetError: [WinError 10054]` — Windows closing browser
  connection on navigate/refresh. Not a bug.
- `LF will be replaced by CRLF` git warnings — Windows line ending
  conversion. Harmless.

## How to resume this project in a brand-new chat

1. Upload the project ZIP or provide GitHub repo link.
2. Ask Claude to read `SKILLS.md`, `CLAUDE.md`, and `PROGRESS.md` in that
   order, confirm orientation, then stop and wait for instructions.
3. The new session should NOT re-litigate decisions logged above unless
   the user explicitly reopens them.

**GitHub repo:** https://github.com/Koushik2910/healthcare-ai-assistant
