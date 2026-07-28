"""Prompt engineering layer.

Public re-exports for the ``src.prompts`` package.  Import from here rather
than from sub-modules so that internal reorganisation never breaks call sites.

Typical usage in Phase 4 (ChatService)::

    from src.prompts import PromptBuilder, PromptContext, refusal_for
    from src.prompts import MEDICAL_DISCLAIMER

    builder = PromptBuilder()
    ctx     = builder.build(retrieval=retrieval_result)
    # ...
    user_msg = refusal_for(verdict.category)
"""

from src.prompts.blocks import (
    formatting_contract,
    rag_context,
    scope_and_refusals,
    system_identity,
)
from src.prompts.builder import PromptBuilder, PromptContext
from src.prompts.templates import (
    MEDICAL_DISCLAIMER,
    all_mapped_categories,
    refusal_for,
)

__all__ = [
    # Blocks (exported for testing and introspection)
    "system_identity",
    "scope_and_refusals",
    "formatting_contract",
    "rag_context",
    # Builder
    "PromptBuilder",
    "PromptContext",
    # Templates
    "MEDICAL_DISCLAIMER",
    "refusal_for",
    "all_mapped_categories",
]
