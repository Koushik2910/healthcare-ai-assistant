# SKILLS.md — Engineering Standards for This Project

> Rarely changes. This is the "how to work" contract. For "what's built,"
> see PROGRESS.md. For "what this project is," see CLAUDE.md.

## Role and goal

Build an enterprise-grade Healthcare AI Assistant good enough to be a
flagship portfolio project AND satisfy a 48-hour take-home assignment.
Treat it as production software, not a tutorial or hackathon demo. No
placeholder implementations, no dummy UI, no rushed code.

## Grading reality (from the actual assignment docx)

| Criterion | Weight |
|---|---|
| Code Quality & Structure | 25% |
| Chatbot Functionality | 25% |
| Prompt Engineering & Logic | 20% |
| UI/UX | 10% |
| Architecture & Documentation | 10% |
| Innovation & Additional Features | 10% |

UI is 10%, not the top priority. 70% of the grade is backend/prompt/code
quality. Build accordingly — see CLAUDE.md for how the UI-vs-portfolio
tension was resolved (two clients over one service layer).

## Process rules (non-negotiable)

1. **One phase at a time.** Finish a phase, summarize it, stop. Do not
   jump ahead to future phases unless explicitly asked.
2. **Before writing code:** analyze, explain the WHY behind a design
   decision, compare alternatives, recommend one, then build. When there
   are multiple viable implementations, compare them explicitly rather
   than silently picking one.
3. **Every phase ends with:** a summary, a checklist, the list of files
   created/changed, PowerShell testing steps, a git commit message
   suggestion, and interview talking points.
4. **State assumptions, don't ask unless truly blocking.** Pick the most
   reasonable interpretation and proceed; only ask a clarifying question
   when proceeding would clearly go in the wrong direction.

## Delivery workflow (how files move)

- Claude never assumes a code-execution sandbox has network access. It
  cannot `pip install` third-party SDKs to execute against them — expect
  syntax-checking and fake/mocked tests from Claude, with **real
  verification happening on the user's machine.**
- Every change ships as a **small, targeted ZIP** — only new/changed
  files, preserving relative paths — never a full project re-zip, because
  the user's `.env`, `venv/`, and `data/` (with a live Chroma store and
  session files) must never be overwritten or deleted.
- Every ZIP is followed by exact PowerShell commands: `Expand-Archive` to
  a temp folder, then `robocopy <temp> <project> /E` (merge, don't mirror
  — `/E` copies and overwrites without deleting anything not in the
  source) or `Copy-Item` for single files, then delete the temp
  folder/zip.
- **PowerShell only.** Windows backslash paths. Never bash or CMD syntax.
- The user runs every test themselves and pastes output back. Claude does
  not claim something works until the user's own `pytest` run confirms
  it — Claude's own sandbox tests against fakes are necessary but not
  sufficient proof for anything touching a real third-party SDK.

## Code quality bar

- PEP8, full type hints, Google-style docstrings explaining WHY not just
  WHAT (design rationale in module/class docstrings, not just parameter
  lists).
- SOLID principles, no duplicated logic, reusable classes/functions,
  readable names.
- Root cause stated before the fix — no motivational preamble, no
  hedging apology spirals.
- Fail fast: invalid configuration must be rejected at startup with an
  actionable message, never allowed to misbehave later downstream.
- Every exception carries an engineer-facing `detail` and a separate
  user-safe `user_message` — raw exceptions/stack traces must never reach
  the UI.
- User health-question content is redacted in logs by default; verbatim
  logging is local-dev-only and rejected by config validation outside
  `APP_ENV=local`.

## Testing standards

- Tests must be hermetic: no real network calls, no real API keys
  required, environment variables stripped and `.env` loading disabled
  per test (see `tests/conftest.py`).
- Third-party SDK calls are tested against hand-built fakes shaped like
  the real SDK's response objects — not against the real network.
- Every new phase gets its own test file(s); existing tests must keep
  passing (regression-free).
- Mark tests (`unit`, `safety`, `eval`) per `pyproject.toml` markers.

## Safety architecture (must hold across every phase)

Four independent layers — this is the flagship differentiator, do not
water it down for convenience:
1. Input screening (deterministic, pre-model) — crisis/emergency/
   diagnosis/prescription/injection detection.
2. Prompt architecture — role, scope, refusal taxonomy, formatting
   contract.
3. Retrieval grounding — cite real sources, say plainly when there are
   none.
4. Output validation (post-model) — block diagnosis language and numeric
   dosages even from an innocuous-looking question.

A refusal is a **value returned** (`SafetyVerdict`), never an exception.
The chatbot must never diagnose, prescribe, or replace a doctor — always
recommend professional consultation where appropriate.

## Documentation obligations (final deliverables)

- README with setup instructions (PowerShell, two-command install)
- ARCHITECTURE.md
- LOGIC documentation (PDF)
- 4–5 slide architecture presentation
- Clean GitHub repo, interview-quality code throughout
