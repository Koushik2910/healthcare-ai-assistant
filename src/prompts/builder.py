"""Prompt assembly: turns building blocks into a single system prompt string.

Design rationale
----------------
The :class:`PromptBuilder` is the *only* place in the codebase that decides
which blocks appear in a given turn's system prompt and in what order.
Keeping that logic here — rather than spreading it across the service layer,
the UI, or the safety layer — means changing the composition strategy is a
one-file change.

:class:`PromptContext` is what the builder hands to the service layer. It
carries both the fully assembled system prompt and the (possibly empty) list
of citation-aware chunks so that the service layer can construct
:class:`~src.models.chat.Citation` objects from the same data that was
injected into the prompt, without having to re-derive it.

Usage pattern (Phase 4 — ChatService)::

    from src.prompts.builder import PromptBuilder
    from src.models.rag import RetrievalResult

    builder = PromptBuilder()                          # cheap, no I/O
    context = builder.build(retrieval=retrieval_result)
    result  = await llm.generate(messages, system_prompt=context.system_prompt)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.prompts.blocks import (
    formatting_contract,
    rag_context,
    scope_and_refusals,
    system_identity,
)
from src.models.rag import RetrievalResult


@dataclass(frozen=True)
class PromptContext:
    """The fully assembled prompt context for one chat turn.

    Attributes:
        system_prompt: The complete system prompt string to pass to the
            provider's ``stream()`` / ``generate()`` call.
        grounded: ``True`` when at least one RAG chunk was injected into the
            prompt.  The service layer uses this to set
            :attr:`~src.models.chat.ResponseSource` on the assistant message.
        injected_chunks: The ``(marker, title, text)`` tuples that were
            formatted into the RAG context block, in citation order.  Empty
            when ``grounded`` is ``False``.  The service layer uses this to
            build :class:`~src.models.chat.Citation` objects without a second
            lookup.
    """

    system_prompt: str
    grounded: bool = False
    injected_chunks: list[tuple[int, str, str]] = field(default_factory=list)


class PromptBuilder:
    """Assembles a per-turn system prompt from composable blocks.

    The builder is stateless between calls and cheap to construct, so the
    service layer may instantiate it once at startup (as an attribute) or per
    request — either pattern is correct.

    Example::

        builder = PromptBuilder()

        # Ungrounded turn (RAG disabled or no retrieval hits)
        ctx = builder.build()
        assert not ctx.grounded

        # Grounded turn (retrieval returned hits above the score floor)
        ctx = builder.build(retrieval=result_with_chunks)
        assert ctx.grounded
        assert len(ctx.injected_chunks) == len(result_with_chunks.chunks)
    """

    # Separator inserted between blocks for readability in the raw prompt
    _SEPARATOR: str = "\n\n"

    def build(
        self,
        *,
        retrieval: RetrievalResult | None = None,
    ) -> PromptContext:
        """Assemble the system prompt for one chat turn.

        Block order:
          1. System identity (always present)
          2. Scope + refusal taxonomy (always present)
          3. Formatting contract (always present)
          4. RAG context (only when ``retrieval`` has hits)

        Args:
            retrieval: The outcome of the knowledge-base query for this turn,
                or ``None`` when RAG is disabled or produced no results.

        Returns:
            A frozen :class:`PromptContext` ready to hand to the LLM provider.
        """
        blocks: list[str] = [
            system_identity(),
            scope_and_refusals(),
            formatting_contract(),
        ]

        injected_chunks: list[tuple[int, str, str]] = []
        grounded = False

        if retrieval is not None and retrieval.has_context:
            # Convert RetrievedChunk objects into (marker, title, text) tuples.
            # Markers are 1-based so they map directly to the [1], [2] tokens
            # the formatting_contract() tells the model to produce.
            for i, retrieved in enumerate(retrieval.top(limit=len(retrieval.chunks)), start=1):
                injected_chunks.append((i, retrieved.chunk.title, retrieved.chunk.text))

            blocks.append(rag_context(injected_chunks))
            grounded = True

        return PromptContext(
            system_prompt=self._SEPARATOR.join(blocks),
            grounded=grounded,
            injected_chunks=injected_chunks,
        )
