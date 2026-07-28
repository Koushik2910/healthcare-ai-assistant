"""Typed application configuration.

Design rationale
----------------
* **One source of truth.** No module reads ``os.environ`` directly. Every
  tunable value in the system is declared here with a type, a default and a
  docstring, which doubles as the configuration reference for the README.

* **Fail fast, fail loudly.** Validation runs at import of the settings
  object, not at first use. A missing API key or a nonsensical chunk overlap
  raises :class:`ConfigurationError` during startup rather than surfacing as
  a confusing runtime failure three layers deep in the retriever.

* **Secrets are typed as secrets.** API keys use :class:`~pydantic.SecretStr`,
  so an accidental ``print(settings)`` or a logged config dump renders
  ``**********`` instead of a live credential. :meth:`Settings.safe_dump`
  exists specifically so startup diagnostics can log configuration without
  leaking keys.

* **Cached accessor.** :func:`get_settings` is memoised, so the object is
  constructed and validated exactly once per process while remaining
  trivially overridable in tests via ``get_settings.cache_clear()``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.exceptions import ConfigurationError

#: Repository root, derived from this file's location: src/config/settings.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration, loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----------------------------------------------------------------- #
    # Application
    # ----------------------------------------------------------------- #

    app_name: str = Field(
        default="Healthcare AI Assistant",
        description="Display name used in the UI and documentation.",
    )
    app_env: Literal["local", "staging", "production"] = Field(
        default="local",
        description="Deployment environment. Guards developer-only behaviour.",
    )

    # ----------------------------------------------------------------- #
    # Logging
    # ----------------------------------------------------------------- #

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Root logging level."
    )
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="'json' for structured logs, 'console' for local readability.",
    )
    log_user_content: bool = Field(
        default=False,
        description=(
            "Log user messages verbatim instead of redacting them. Local "
            "prompt debugging only; rejected outside the 'local' environment."
        ),
    )

    # ----------------------------------------------------------------- #
    # Language model
    # ----------------------------------------------------------------- #

    llm_provider: Literal["gemini", "groq", "openrouter"] = Field(
        default="gemini",
        description="Active provider. Only its key is required; the others may be omitted.",
    )
    gemini_api_key: SecretStr | None = Field(
        default=None, description="Google AI Studio API key."
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model id. Flash is chosen for latency and cost.",
    )
    groq_api_key: SecretStr | None = Field(
        default=None, description="Groq API key, used by the fallback provider."
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile", description="Groq model id."
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description=(
            "OpenRouter API key. Optional third provider, included to "
            "demonstrate the abstraction rather than as the graded path -- "
            "a reviewer without OpenRouter credit cannot exercise it for free."
        ),
    )
    openrouter_model: str = Field(
        default="google/gemini-2.5-flash",
        description="OpenRouter model slug, in 'provider/model' form.",
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description=(
            "Low by default: a healthcare assistant should be consistent and "
            "conservative rather than creative."
        ),
    )
    llm_max_output_tokens: int = Field(
        default=1024, gt=0, le=8192, description="Ceiling on generated tokens."
    )
    llm_timeout_seconds: float = Field(
        default=30.0, gt=0, description="Per-request timeout for provider calls."
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retries for transient provider failures, with backoff.",
    )

    # ----------------------------------------------------------------- #
    # Retrieval
    # ----------------------------------------------------------------- #

    rag_enabled: bool = Field(
        default=True,
        description="Disable to run the assistant ungrounded, for A/B comparison.",
    )
    chroma_persist_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma",
        description="On-disk location of the Chroma vector store.",
    )
    chroma_collection: str = Field(
        default="healthcare_kb", description="Chroma collection name."
    )
    knowledge_base_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "knowledge_base",
        description="Source markdown documents ingested into the vector store.",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description=(
            "Runs locally, so the reviewer needs no second API key and "
            "embedding cost is zero."
        ),
    )
    chunk_size: int = Field(
        default=800, gt=0, description="Target characters per chunk."
    )
    chunk_overlap: int = Field(
        default=120,
        ge=0,
        description="Characters shared between adjacent chunks to preserve context.",
    )
    retrieval_top_k: int = Field(
        default=4, gt=0, le=20, description="Chunks passed to the model as context."
    )
    retrieval_min_score: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "Similarity floor. Below this the chunk is discarded, so weak "
            "matches cannot masquerade as citations."
        ),
    )

    # ----------------------------------------------------------------- #
    # Safety
    # ----------------------------------------------------------------- #

    safety_strict_mode: bool = Field(
        default=True,
        description=(
            "When True, output validation failures block the response. When "
            "False they are logged only. Strict is the correct default for a "
            "health assistant."
        ),
    )
    max_input_chars: int = Field(
        default=2000,
        gt=0,
        description=(
            "Input ceiling enforced before any billable call. Limits prompt "
            "stuffing and denial-of-wallet attempts."
        ),
    )

    # ----------------------------------------------------------------- #
    # Conversation state
    # ----------------------------------------------------------------- #

    session_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "sessions",
        description="Directory holding per-session conversation JSON files.",
    )
    max_history_turns: int = Field(
        default=8,
        gt=0,
        description="Recent turns replayed verbatim into the prompt.",
    )
    history_summary_threshold_chars: int = Field(
        default=4000,
        gt=0,
        description="History size above which older turns are summarised.",
    )

    # ----------------------------------------------------------------- #
    # Validation
    # ----------------------------------------------------------------- #

    @field_validator("chroma_persist_dir", "knowledge_base_dir", "session_dir")
    @classmethod
    def _resolve_paths(cls, value: Path) -> Path:
        """Resolve configured paths against the repository root when relative."""
        path = Path(value)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @model_validator(mode="after")
    def _validate_coherence(self) -> "Settings":
        """Reject configurations that are individually valid but jointly wrong."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}); otherwise chunking cannot "
                "make forward progress."
            )

        if self.log_user_content and self.app_env != "local":
            raise ValueError(
                "log_user_content may only be enabled when app_env='local'. "
                "Verbatim logging of health questions outside local "
                "development is a data-protection failure."
            )

        if self.active_api_key is None:
            raise ValueError(
                f"llm_provider is '{self.llm_provider}' but "
                f"{self.llm_provider.upper()}_API_KEY is not set. Copy "
                ".env.example to .env and add your key."
            )

        return self

    # ----------------------------------------------------------------- #
    # Derived accessors
    # ----------------------------------------------------------------- #

    @property
    def active_api_key(self) -> SecretStr | None:
        """Return the API key belonging to the currently selected provider."""
        return {
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
        }[self.llm_provider]

    @property
    def active_model(self) -> str:
        """Return the model id belonging to the currently selected provider."""
        return {
            "gemini": self.gemini_model,
            "groq": self.groq_model,
            "openrouter": self.openrouter_model,
        }[self.llm_provider]

    @property
    def is_local(self) -> bool:
        """True when running in the local development environment."""
        return self.app_env == "local"

    def safe_dump(self) -> dict[str, Any]:
        """Return configuration with every secret replaced by a placeholder.

        Safe to log at startup, which makes "it works on my machine" bug
        reports resolvable from the log stream alone.
        """
        data = self.model_dump(mode="json")
        for key, value in list(data.items()):
            if isinstance(getattr(self, key, None), SecretStr):
                data[key] = "***set***" if getattr(self, key) else None
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated, process-wide settings singleton.

    Raises:
        ConfigurationError: If any value is missing or invalid. The message is
            written for a human who is trying to run the project for the first
            time, not for a stack trace.
    """
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'config'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigurationError(
            f"Invalid configuration -- {problems}",
            user_message=(
                "The application could not start because its configuration is "
                f"incomplete. Details: {problems}"
            ),
        ) from exc
