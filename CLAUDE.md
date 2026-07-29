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
This exact conflict already came up once (see decision 1 below); it does
not need to be re-litigated.

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

## Key architecture decisions (with rationale)

**Quick-scan version — these are settled, do not reopen without the user
explicitly asking:**
1. Streamlit for the graded submission; React/Next.js only later, for
   the portfolio, after submission.
2. Three providers: Gemini (primary, free, graded path) / Groq (free
   fallback) / OpenRouter (paid, portfolio-only, never graded path).
3. Retry stops at the first streamed token — never retry mid-stream.
4. Error translation is duck-typed, not `isinstance`-based.
5. RAG corpus will be small, self-written/public-domain only.
6. Guardrails return values (`SafetyVerdict`), never raise.

Full rationale for each:

1. **Streamlit for the graded submission, not React.** The assignment
   docx says "Streamlit (Preferred)"; UI is only 10% of the rubric.
   Mitigation for "it's just Streamlit": business logic lives in a
   UI-agnostic `ChatService`, so Streamlit is one thin client and a
   FastAPI + Next.js client can be added later without touching logic.
   → **Next.js front-end is planned as a POST-SUBMISSION portfolio
   addition** (Phases 9–10 in the original 10-phase plan), reusing
   CareerApex's chat shell/streaming/theme tokens. Not on the graded
   critical path.

2. **Three interchangeable LLM providers: Gemini (primary), Groq
   (fallback), OpenRouter (optional third, portfolio-only).**
   - Gemini: direct `google-genai` SDK, free tier, reviewer needs only a
     free key from https://aistudio.google.com/apikey — zero-cost
     reproducibility for grading.
   - Groq: genuinely free, independent quota/infrastructure from Google —
     the real safety net if Gemini's daily quota is exhausted during
     grading. Uses the `openai` SDK pointed at Groq's OpenAI-compatible
     endpoint (not the native `groq` package) so Groq and OpenRouter share
     one implementation (`OpenAICompatibleProvider`).
   - OpenRouter: **NOT free for Gemini specifically** — draws down a
     paid credit balance (~$0.30/$2.50 per M tokens for
     `google/gemini-2.5-flash`, $5 minimum balance). Included only to
     demonstrate the abstraction and reuse Koushik's existing key from
     CareerApex; never the primary/graded path.
   - **Gemini free-tier quota is per Google Cloud project/account, NOT
     per API key** — generating a new key under the same account does
     NOT reset quota. If exhausted: switch `LLM_PROVIDER=groq` (free,
     first choice), or `=openrouter` (cheap, uses existing credits), or
     wait for daily reset, or enable billing on the same project.
   - **Not yet built:** automatic in-app failover (Gemini fails →
     silently retry on Groq, log which provider actually answered).
     Planned for Phase 4 (chat service) — agreed but not implemented as
     of Phase 2 completion.

3. **Retry policy stops at the first streamed token, on purpose.**
   Retrying after the model has already started talking would duplicate
   or interleave output shown to the user — worse than a clean error. See
   `src/llm/retry.py` docstring. This is a considered design choice, not
   a limitation — do not "fix" it by making retries resume mid-stream.

4. **Error translation uses duck-typing, not `isinstance` against SDK
   exception classes.** Third-party SDK exception constructors change
   shape across versions and couldn't be verified against a live
   installed version in Claude's sandbox (no network access there). Duck
   typing (status codes, message substrings) is more resilient and more
   testable. See `src/llm/openai_compatible.py` and
   `src/llm/gemini_provider.py`.

5. **RAG corpus will be small and self-written/public-domain only.**
   Assignment forbids copyrighted datasets. Plan: 12–18 short documents
   synthesized from MedlinePlus/CDC/NIH, with a `SOURCES.md` recording
   provenance, enforced by a required `DocumentLicence` enum on every
   `Chunk` (already modeled in `src/models/rag.py`; ingestion not yet
   built — that's Phase 5).

6. **A refusal is a returned value, never an exception.** See
   `src/models/safety.py` (`SafetyVerdict`). This is what will make the
   Phase 7 adversarial eval harness produce a measurable pass rate rather
   than an untestable pile of `except` blocks.

## Constraints Claude must respect in this project

- Claude's own code-execution sandbox has **no network access** — cannot
  `pip install` or make real API calls. All "verification" of real SDK
  behavior happens on the user's machine via scripts Claude provides
  (e.g. `scripts/smoke_test_llm.py`), never assumed.
- Every ZIP shipped is a **partial patch**, not a full re-zip — the
  user's `.env`, `venv/`, `data/chroma`, `data/sessions` must survive
  every update untouched.

## How to resume this project in a brand-new chat

1. Upload/paste `SKILLS.md`, `CLAUDE.md`, and `PROGRESS.md` (this trio)
   into the new conversation, plus the current project ZIP or repo if
   available.
2. Say: *"Read these three files and PROGRESS.md's 'Next up' section,
   then continue the project from there, following SKILLS.md's process
   rules."*
3. The new session should NOT re-litigate decisions already logged above
   unless the user explicitly reopens them.
