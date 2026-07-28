"""Language-model domain models.

Kept separate from ``src.models.chat`` because these types describe a
provider's raw output (which model, how long it took, why it stopped),
whereas ``chat.py`` describes the conversation-level view the UI renders.
The chat service is the seam that turns a :class:`GenerationResult` into a
:class:`~src.models.chat.Message`.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProviderName(str, Enum):
    """Identifies which backend produced a given generation.

    Recorded on every result so the evaluation harness and logs can report
    outcomes per provider -- useful the moment Gemini and Groq disagree on
    an adversarial prompt.
    """

    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"


class GenerationResult(BaseModel):
    """The complete, non-streaming outcome of one model call.

    Produced by :meth:`~src.llm.base.LLMProvider.generate`, which itself is
    built by concatenating :meth:`~src.llm.base.LLMProvider.stream` -- so
    this type is what a full stream collapses into once it finishes.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    provider: ProviderName
    model: str
    latency_ms: int = Field(ge=0)

    @property
    def is_empty(self) -> bool:
        """True when the provider returned no usable text.

        A provider-side safety filter can return a 200 response with empty
        content; this makes that condition explicit rather than letting an
        empty string silently flow downstream as if it were a real answer.
        """
        return not self.text.strip()
