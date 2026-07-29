# Healthcare AI Assistant

A production-grade healthcare information chatbot built with safety-first
engineering: layered deterministic guardrails, retrieval-augmented generation
grounded in public-domain sources, multi-provider LLM failover, and a measured
adversarial evaluation harness.

**Built for:** First500days AI Engineer take-home assignment + portfolio.  
**Stack:** Python 3.11+ · Gemini 2.5 Flash (primary) · Groq Llama-3.3-70B (fallback) · ChromaDB · Streamlit  
**Test suite:** 363 tests, zero network calls, hermetic.  
**Eval harness:** 20 adversarial cases, 100% pass rate.

---

## What makes this different from a system-prompt chatbot

Most "safe" LLM assistants put their safety in a system prompt and hope the
model obeys. This one treats safety as an engineered subsystem with four
independent, measurable layers:

| Layer | When it runs | What it does |
|---|---|---|
| **1. Input screening** | Before any model call | Deterministic regex patterns catch crisis, emergency, diagnosis/prescription requests, and prompt injection. The model is never consulted — no jailbreak can bypass a branch that never executes. |
| **2. Prompt architecture** | At composition | Role declaration, scope boundaries, refusal taxonomy, RAG context injection, and formatting contract are assembled from independently testable modules. |
| **3. Retrieval grounding** | During generation | Answers are grounded in 13 curated public-domain documents (112 chunks, `all-MiniLM-L6-v2` embeddings, ChromaDB L2 similarity). Citations are carried from source through to the UI. |
| **4. Output validation** | After generation | Blocks diagnosis language and numeric dosages that the model produced despite an innocuous-looking input. |

Layer 1 fires **before** the model. A crisis message never depends on the model
behaving correctly, because the model is never called.

---

## Quick start

Requires **Python 3.11+**. All commands are PowerShell.

```powershell
# 1. Clone
git clone https://github.com/Koushik2910/healthcare-ai-assistant
cd healthcare-ai-assistant

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # needed for tests only

# 4. Configure
Copy-Item .env.example .env
# Edit .env — paste your Gemini API key into GEMINI_API_KEY
# Free key: https://aistudio.google.com/apikey

# 5. Preflight check
python scripts\preflight.py           # should print: All checks passed.

# 6. Build the knowledge base (one-time, ~2 min, downloads ~80 MB model)
python scripts\ingest.py

# 7. Run the app
streamlit run app.py
# Open http://localhost:8501 in your browser
```

---

## Running tests

```powershell
# Full suite — 363 tests, no network, no API key needed
pytest

# Adversarial eval — 20 cases against live Gemini API (~50 s)
python scripts\eval.py

# Guardrail-only eval (fast, ~20 s)
python scripts\eval.py --category CRISIS EMERGENCY OUT_OF_SCOPE PROMPT_INJECTION
```

---

## Project structure

```
healthcare-ai-assistant/
├── app.py                        Root Streamlit entrypoint
├── .streamlit/config.toml        Streamlit server config
├── requirements.txt              Runtime dependencies
├── requirements-dev.txt          Test/lint dependencies
├── pyproject.toml                pytest / ruff / mypy config
│
├── scripts/
│   ├── preflight.py              Startup diagnostic
│   ├── ingest.py                 Build / rebuild the ChromaDB knowledge base
│   ├── smoke_test_llm.py         Verify live provider connectivity
│   └── eval.py                   Adversarial evaluation runner
│
├── src/
│   ├── config/settings.py        Typed settings, single source of truth
│   ├── models/                   Pydantic domain models
│   │   ├── chat.py               Message, Conversation, ChatResponse
│   │   ├── safety.py             SafetyVerdict, RiskCategory, SafetyAction
│   │   ├── rag.py                Chunk, RetrievedChunk, RetrievalResult
│   │   └── llm.py                GenerationResult, ProviderName
│   ├── llm/                      LLM provider abstraction
│   │   ├── base.py               LLMProvider ABC + generate()
│   │   ├── retry.py              stream_with_retry — stops at first token
│   │   ├── factory.py            get_llm() — single switch point
│   │   ├── gemini_provider.py    Gemini 2.5 Flash (primary)
│   │   ├── groq_provider.py      Groq Llama-3.3-70B (free fallback)
│   │   └── openrouter_provider.py OpenRouter (portfolio, post-submission)
│   ├── prompts/                  Composable prompt architecture
│   │   ├── blocks.py             Role, scope, RAG, refusal, format blocks
│   │   ├── builder.py            PromptBuilder — assembles blocks per turn
│   │   └── templates.py          Crisis / refusal / disclaimer templates
│   ├── safety/
│   │   ├── input_guard.py        Layer 1 — deterministic input screening
│   │   └── output_guard.py       Layer 4 — post-generation validation
│   ├── rag/
│   │   ├── ingestion.py          Document loading, chunking, embedding
│   │   └── retriever.py          ChromaDB query → RetrievalResult
│   ├── services/
│   │   └── chat_service.py       Five-stage pipeline orchestrator
│   ├── eval/
│   │   └── cases.py              20 shared adversarial eval cases
│   └── ui/
│       ├── app.py                Streamlit single-page app
│       └── components.py         Message bubble, crisis card, source cards
│
├── data/
│   ├── knowledge_base/           13 public-domain health documents (JSON)
│   └── chroma/                   ChromaDB vector store (after ingest)
│
├── tests/
│   ├── conftest.py               Hermetic environment isolation fixture
│   ├── fakes.py                  FakeLLMProvider, FakeGeminiClient, etc.
│   ├── test_config.py            Settings validation
│   ├── test_models.py            Domain model contracts
│   ├── test_observability.py     Logging and exception hierarchy
│   ├── test_llm_*.py             Provider abstraction and retry logic
│   ├── test_prompts.py           Prompt block composition
│   ├── test_safety.py            Guardrail pattern coverage
│   ├── test_chat_service.py      Full pipeline integration (fake LLM)
│   ├── test_rag.py               Ingestion, chunking, retrieval (fake DB)
│   └── test_eval_harness.py      75 adversarial routing tests (fake LLM)
│
└── docs/
    ├── ARCHITECTURE.md           System design and decision rationale
    └── EVALUATION.md             Adversarial eval results and methodology
```

---

## Configuration reference

All settings live in `src/config/settings.py` and are loaded from `.env`.
No module reads `os.environ` directly.

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `groq`, or `openrouter` |
| `GEMINI_API_KEY` | — | Required when `LLM_PROVIDER=gemini` |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Any Gemini model string |
| `GROQ_API_KEY` | — | Optional; enables Gemini→Groq auto-failover |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq model string |
| `LLM_TEMPERATURE` | `0.3` | Low for consistency over creativity |
| `LLM_MAX_OUTPUT_TOKENS` | `1024` | Ceiling on generated tokens |
| `LLM_TIMEOUT_SECONDS` | `30.0` | Per-request timeout |
| `LLM_MAX_RETRIES` | `2` | Retries before first token only |
| `SAFETY_STRICT_MODE` | `true` | `false` logs output violations, not blocks |
| `MAX_INPUT_CHARS` | `2000` | Enforced before any billable API call |
| `LOG_USER_CONTENT` | `false` | Only permitted when `APP_ENV=local` |
| `CHROMA_COLLECTION` | `healthcare_kb` | ChromaDB collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |

---

## Knowledge base

13 public-domain health documents, self-authored for this project under an
original licence. Topics: hydration, nutrition, sleep hygiene, heart health,
vaccination, preventive screenings, mental health basics, stress management,
physical activity, healthy weight, back pain, cold/flu, first aid.

Each document is chunked (~300 chars, 50-char overlap), embedded with
`all-MiniLM-L6-v2`, and stored in ChromaDB with full provenance metadata
(title, source, licence, topics). Provenance travels from ingestion through
retrieval to citation in the UI — no chunk can appear without attribution.

---

## Safety scope

This assistant provides **general health information only**. It does not
diagnose conditions, recommend or dose medication, or substitute for a
qualified clinician. In a medical emergency, call **112**. For a mental
health crisis, call or text **988**.

---

## Build status

- [x] Phase 1 — Foundation: config, logging, exceptions, domain models
- [x] Phase 2 — LLM provider abstraction (Gemini / Groq / OpenRouter, retry, streaming)
- [x] Phase 3 — Prompt architecture (blocks, builder, templates)
- [x] Phase 4 — Safety layers + ChatService with Gemini→Groq auto-failover
- [x] Phase 5 — RAG: 13 documents, 112 chunks, ChromaDB, citations
- [x] Phase 6 — Streamlit UI: streaming, crisis cards, RAG sources, custom CSS
- [x] Phase 7 — Adversarial eval harness: 363 tests, 20/20 eval cases
- [x] Phase 8 — Documentation, architecture, evaluation report
