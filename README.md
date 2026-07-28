# Healthcare AI Assistant

A safety-first healthcare information assistant: an LLM chatbot that answers
general health questions, grounds its answers in a curated public-domain
knowledge base, and refuses -- deterministically -- to diagnose conditions or
recommend medication.

> **This project is under active construction.** Phase 1 (foundation) is
> complete. See [Build status](#build-status) below.

---

## What makes this different from a system-prompt chatbot

Most "safe" LLM assistants put their safety in a system prompt and hope. This
one treats safety as an engineered subsystem with four independent layers, and
measures it:

| Layer | Runs | Purpose |
|---|---|---|
| 1. Input screening | Before any model call | Detects crisis, emergency, diagnosis and prescription requests, and prompt injection using deterministic rules that cannot be argued out of |
| 2. Prompt architecture | At composition | Role, scope boundaries, refusal taxonomy and formatting contract, assembled from independently testable modules |
| 3. Retrieval grounding | During generation | Answers from cited public-domain sources; states plainly when it has none |
| 4. Output validation | After generation | Blocks diagnosis language and numeric dosages the model produced despite an innocuous-looking question |

Layer 1 running *before* the model is the key property: a crisis or emergency
message never depends on the model behaving correctly, because the model is
never consulted.

---

## Setup

Requires **Python 3.11 or later**. Commands below are PowerShell.

```powershell
# 1. Clone and enter the project
git clone <repository-url>
cd healthcare-ai-assistant

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed to run the tests

# 4. Configure
Copy-Item .env.example .env
# Open .env and paste your key into GEMINI_API_KEY.
# A free key is available at https://aistudio.google.com/apikey

# 5. Verify the setup
python scripts\preflight.py
```

`preflight.py` validates configuration, creates the data directories and
confirms the logging pipeline redacts user content. It should end with
`All checks passed.`

### Running the tests

```powershell
pytest
```

The suite makes no network calls and consumes no API quota, so it runs in
under a second and works offline.

---

## Configuration

Every tunable value is declared in `src/config/settings.py` with a type, a
default and a docstring; no module reads `os.environ` directly. `.env.example`
lists all of them. Notable entries:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `groq`. Only the active provider's key is required. |
| `LLM_TEMPERATURE` | `0.3` | Low by design: consistency matters more than creativity here. |
| `RAG_ENABLED` | `true` | Set `false` to compare grounded and ungrounded behaviour. |
| `SAFETY_STRICT_MODE` | `true` | When `false`, output-validation failures are logged rather than blocked. |
| `MAX_INPUT_CHARS` | `2000` | Enforced before any billable call. |
| `LOG_USER_CONTENT` | `false` | Verbatim logging of health questions. Rejected unless `APP_ENV=local`. |

Invalid configuration is rejected at startup with an actionable message rather
than failing later inside the retriever.

---

## Project structure

```
healthcare-ai-assistant/
|-- scripts/preflight.py      Startup diagnostic
|-- src/
|   |-- config/               Typed settings, single source of truth
|   |-- models/               Pydantic domain models (chat, safety, rag)
|   |-- llm/                  Provider abstraction: Gemini, Groq        [phase 2]
|   |-- prompts/              Composable prompt modules                 [phase 3]
|   |-- safety/               Guardrail layers 1 and 4                  [phase 4]
|   |-- rag/                  Ingestion, retrieval, citations           [phase 5]
|   |-- services/             ChatService -- UI-agnostic orchestrator   [phase 4]
|   |-- ui/                   Streamlit components                      [phase 6]
|   `-- utils/                Logging and exception hierarchy
|-- data/knowledge_base/      Source documents plus SOURCES.md          [phase 5]
|-- tests/                    Unit tests and adversarial eval suite
`-- docs/                     Architecture and logic documentation      [phase 8]
```

The domain models in `src/models/` are the contract between the service layer
and any client. `src/ui/` is one client; a FastAPI adapter is a second. Neither
contains business logic, which is what makes the interface replaceable rather
than merely separated in a diagram.

---

## Build status

- [x] **Phase 1** -- Foundation: configuration, logging, exceptions, domain models
- [ ] Phase 2 -- LLM provider abstraction with streaming
- [ ] Phase 3 -- Prompt architecture
- [ ] Phase 4 -- Safety layers and chat service
- [ ] Phase 5 -- Retrieval and citations
- [ ] Phase 6 -- Streamlit interface
- [ ] Phase 7 -- Test suite and adversarial evaluation harness
- [ ] Phase 8 -- Documentation, architecture deck, demo

---

## Scope and limitations

This assistant provides **general health information only**. It does not
diagnose conditions, recommend or dose medication, or substitute for a
clinician, and it is not a medical device. In an emergency, contact local
emergency services.

Knowledge-base documents are drawn only from US federal public-domain sources
or written for this project; provenance and licence are recorded per document
in `data/knowledge_base/SOURCES.md` and carried on every chunk through to the
citation shown in the interface.
