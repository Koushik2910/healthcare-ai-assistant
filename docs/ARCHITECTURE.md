# Architecture — Healthcare AI Assistant

## System overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (Browser)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP (Streamlit WebSocket)
┌───────────────────────────▼─────────────────────────────────────┐
│                    src/ui/app.py  (Streamlit)                   │
│  Session state · st.empty() streaming · Crisis card · Sources  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  ChatService.chat() / stream_chat()
┌───────────────────────────▼─────────────────────────────────────┐
│              src/services/chat_service.py  (ChatService)        │
│                                                                 │
│  Stage 1 ──► InputGuard.screen()          ──► ESCALATE / REFUSE│
│  Stage 2 ──► Retriever.retrieve()         ──► RetrievalResult  │
│  Stage 3 ──► PromptBuilder.build()        ──► PromptContext     │
│  Stage 4 ──► LLMProvider.stream()         ──► text chunks       │
│              └─► Gemini (primary)                               │
│              └─► Groq   (auto-failover on rate-limit/timeout)   │
│  Stage 5 ──► OutputGuard.validate()       ──► block / pass      │
└───────────────────────────┬─────────────────────────────────────┘
          │                 │                    │
  ┌───────▼──────┐  ┌───────▼──────┐  ┌────────▼───────┐
  │  InputGuard  │  │  ChromaDB    │  │  Gemini API    │
  │  (regex,     │  │  + MiniLM    │  │  (Groq API     │
  │  deterministic│  │  embeddings) │  │   fallback)    │
  └──────────────┘  └──────────────┘  └────────────────┘
```

---

## Five-stage pipeline

Every chat turn passes through exactly five stages in a fixed order inside
`ChatService`. The order is not configurable — it is the safety contract.

### Stage 1 — Input screening (InputGuard)

Runs **before** any model call. Uses deterministic regex patterns grouped by
risk category:

| Category | Action | Example trigger |
|---|---|---|
| `SELF_HARM` | `ESCALATE` | "I want to kill myself" |
| `EMERGENCY` | `ESCALATE` | "I have chest pain" |
| `DIAGNOSIS_REQUEST` | `REFUSE` | "Do I have diabetes?" |
| `MEDICATION_REQUEST` | `REFUSE` | "What dose of ibuprofen?" |
| `DANGEROUS_INSTRUCTION` | `REFUSE` | "How to purge after eating" |
| `PROMPT_INJECTION` | `REFUSE` | "Ignore your instructions" |
| `OUT_OF_SCOPE` | `REFUSE` | "What stocks should I buy?" |
| `NONE` | `ALLOW` | "How much water per day?" |

If the action is `ESCALATE` or `REFUSE`, the pipeline returns immediately.
The model is never called. No system prompt, no user content, no jailbreak
can circumvent a code path that does not execute.

### Stage 2 — Retrieval (Retriever)

Queries ChromaDB for the top-5 nearest neighbours to the user's message.
Converts L2 distances to similarity scores via `1 / (1 + distance)` — a
monotone, corpus-independent mapping. Drops results below threshold (default
0.45). Returns a `RetrievalResult` with scored `RetrievedChunk` objects.

If ChromaDB is unavailable, returns `degraded=True` and the pipeline
continues ungrounded — chat is not interrupted.

### Stage 3 — Prompt assembly (PromptBuilder)

Assembles the system prompt from composable blocks:

- **Role block** — defines the assistant's identity and authority limits
- **Scope block** — enumerates permitted and forbidden topics
- **RAG block** — injects retrieved chunks with citation markers `[1]`, `[2]`
- **Refusal taxonomy block** — gives the model a vocabulary for declining
- **Formatting contract block** — output structure, disclaimer trigger rules

Blocks are independently testable units. The builder is tested to produce the
correct block combination for every retrieval/no-retrieval scenario.

### Stage 4 — LLM call with auto-failover

Primary: **Gemini 2.5 Flash** (low latency, generous free tier).  
Fallback: **Groq Llama-3.3-70B** (constructed lazily, only when GROQ_API_KEY is set).

Failover fires only on `LLMRateLimitError` and `LLMTimeoutError` — transient
conditions where the primary is overloaded. It does NOT fire on
`LLMResponseError` (content failure) because retrying on a different provider
would serve a different answer to the same question, which is surprising.

Retry stops at the first streamed token. Mid-stream retries are not supported:
the user is already reading the response, and a silent restart would produce
duplicate or inconsistent output.

### Stage 5 — Output validation (OutputGuard)

Runs after the full response is assembled from the stream. Checks for:
- Diagnosis language ("you have", "this is likely", condition names in
  a self-diagnosis context)
- Numeric dosage specifications
- Prescription recommendations

In strict mode (default), any violation replaces the response with a safe
fallback. In non-strict mode, violations are logged only.

---

## Key design decisions

### Why one class owns all five stages

The alternative — spreading the pipeline across the UI layer, a middleware,
and utility functions — means the ordering and error handling live in several
places and can drift. A single `ChatService.chat()` means the pipeline is
always the same pipeline, whether called by Streamlit, a FastAPI endpoint, or
a test.

### Why guardrails return values, not exceptions

`SafetyVerdict` is a value. `InputGuard.screen()` never raises. This means:

1. Every decision is a first-class value that can be asserted on in a test.
2. Refusals carry a `reason` and a `category`, so the UI responds
   proportionately — a scope refusal is a gentle redirect, a crisis detection
   is a prominent card with hotline numbers.
3. Nothing in the pipeline can accidentally swallow a refusal in a broad
   `except` block, because it never travelled as an exception.

### Why error translation is duck-typed, not isinstance-based

Each provider SDK raises its own exception types. Translating them at the
provider boundary (in `_translate_error`) using attribute inspection rather
than `isinstance` means adding a fourth provider requires no changes to the
retry or failover logic — only a new `_translate_error` implementation.

### Why the RAG corpus is small and self-authored

The assignment forbids copyrighted datasets. Small + self-authored means every
document can be reviewed, every chunk can be inspected, and the retrieval
quality can be reasoned about. A large third-party corpus would be harder to
audit and could introduce licensing risk.

### Why Streamlit rather than React for the graded submission

The assignment rubric weights UI at 10%. The graded deliverable needs to run
with `streamlit run app.py` — no build step, no Node.js. The service layer
(`ChatService`) is client-agnostic; a FastAPI + React frontend is a post-
submission portfolio addition that reuses the same service without changes.

---

## Domain model hierarchy

```
ChatResponse
├── message: Message
│   ├── role: Role (USER | ASSISTANT | SYSTEM)
│   ├── content: str
│   └── source: ResponseSource
├── source: ResponseSource
│   (GROUNDED | MODEL_ONLY | GUARDRAIL | ESCALATION | FALLBACK)
├── citations: list[Citation]
│   └── Citation(marker, title, source, snippet)
├── disclaimer: str | None
└── refused: bool

RetrievalResult
├── query: str
├── chunks: list[RetrievedChunk]
│   └── RetrievedChunk(chunk: Chunk, score: float)
│       └── Chunk(chunk_id, doc_id, text, title, source, licence, topics)
├── took_ms: int
└── degraded: bool

SafetyVerdict
├── action: SafetyAction (ALLOW | ALLOW_WITH_CONSTRAINTS | REFUSE | ESCALATE)
├── category: RiskCategory
├── reason: str
├── confidence: float
└── matched_rules: list[str]
```

---

## LLM provider abstraction

```
LLMProvider (ABC)
├── stream(messages, system_prompt) → AsyncIterator[str]   ← abstract
├── generate(messages, system_prompt) → GenerationResult   ← built on stream
└── name: ProviderName

GeminiProvider(LLMProvider)       ← google-genai SDK
GroqProvider(LLMProvider)         ← via OpenAICompatibleProvider
OpenRouterProvider(LLMProvider)   ← via OpenAICompatibleProvider

get_llm(settings) → LLMProvider   ← single switch point
```

`generate()` is built on `stream()` in the base class, so streaming and
non-streaming call paths cannot diverge in behaviour.

---

## Test architecture

```
363 total tests, zero network calls, zero API quota consumed.

test_config.py           13   Settings validation, env var isolation
test_models.py           19   Pydantic model contracts
test_observability.py    13   Logging redaction, exception hierarchy
test_llm_base.py          5   LLMProvider ABC contract
test_llm_retry.py         6   stream_with_retry: backoff, mid-stream stop
test_llm_providers.py    14   Gemini/Groq/OpenRouter against fakes
test_llm_factory.py       4   get_llm() switching
test_prompts.py          50   Block composition, builder, templates
test_safety.py           82   InputGuard patterns, OutputGuard validators
test_chat_service.py     31   Five-stage pipeline with FakeLLMProvider
test_rag.py              47   Ingestion, chunking, retrieval (fake ChromaDB)
test_eval_harness.py     75   Adversarial routing: guardrails + no false pos.
```

Key properties:
- **Hermetic:** `conftest.py` strips all env vars and clears the settings
  cache before every test. Results are identical on any machine.
- **No implicit async:** `asyncio_mode = strict` — every async test carries
  `@pytest.mark.asyncio` explicitly.
- **call_count proofs:** Guardrail tests assert `primary.call_count == 0`,
  proving the LLM was never called — not just that the response was refused.

---

## Data flow: one complete turn

```
User types: "How much water should I drink daily?"

1. InputGuard.screen("How much water...")
   → SafetyVerdict(action=ALLOW, category=NONE)
   → pipeline continues

2. Retriever.retrieve("How much water...")
   → ChromaDB query: top-5 L2 neighbours
   → 3 chunks pass threshold (score ≥ 0.45)
   → RetrievalResult(chunks=[...], has_context=True)

3. PromptBuilder.build(retrieval=result)
   → Injects 3 chunks as [1] [2] [3] citations
   → Returns PromptContext(system_prompt="...[1]...[2]...[3]...", grounded=True)

4. GeminiProvider.stream(history, system_prompt)
   → Yields text tokens: "Staying well-hydrated..."
   → UI renders tokens progressively via st.empty()

5. OutputGuard.validate(full_text)
   → No diagnosis language, no dosage specs
   → ValidationResult(passed=True)

6. ChatResponse assembled:
   → source=GROUNDED, citations=[...], disclaimer=None, refused=False

7. UI renders:
   → Message bubble with full response
   → 📚 Sources (3) expander with source cards
   → No disclaimer pill (no clinical content detected)
```

---

## File count and line count summary

| Area | Files | Purpose |
|---|---|---|
| `src/config/` | 1 | Typed settings |
| `src/models/` | 4 | Domain contracts |
| `src/llm/` | 6 | Provider abstraction |
| `src/prompts/` | 4 | Prompt architecture |
| `src/safety/` | 3 | Guardrail layers |
| `src/rag/` | 3 | Retrieval pipeline |
| `src/services/` | 2 | Orchestration |
| `src/ui/` | 3 | Streamlit client |
| `src/eval/` | 2 | Eval case definitions |
| `src/utils/` | 3 | Logging, exceptions |
| `scripts/` | 4 | CLI tools |
| `tests/` | 13 | 363 tests |
| `data/knowledge_base/` | 13 | Health documents |
