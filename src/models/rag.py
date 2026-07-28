"""Retrieval domain models.

Provenance is carried on every chunk from ingestion through to citation.
This is a deliberate constraint rather than bookkeeping: the assignment
forbids copyrighted datasets, and a chunk that cannot name its licence and
publisher is a chunk that should never have entered the corpus. Making the
fields required means that rule is enforced by the type system instead of by
memory.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DocumentLicence(str, Enum):
    """Licence under which a source document may be redistributed.

    Only permissive values are allowed into the corpus. US federal health
    agency material is public domain; anything else must be text written for
    this project.
    """

    #: Work of the US federal government, public domain.
    US_GOV_PUBLIC_DOMAIN = "us_gov_public_domain"
    #: Written specifically for this project.
    ORIGINAL = "original"
    #: Creative Commons attribution, redistributable with credit.
    CC_BY = "cc_by"


class KBDocument(BaseModel):
    """A source document before chunking."""

    doc_id: str = Field(description="Stable slug, e.g. 'hydration-basics'.")
    title: str
    content: str
    source: str = Field(description="Publisher, e.g. 'MedlinePlus' or 'Original'.")
    licence: DocumentLicence
    url: str | None = None
    topics: list[str] = Field(
        default_factory=list,
        description="Coarse tags used for metadata filtering at query time.",
    )

    @property
    def char_count(self) -> int:
        """Length of the document body in characters."""
        return len(self.content)


class Chunk(BaseModel):
    """A retrievable fragment of a document.

    Provenance fields are duplicated from the parent document rather than
    referenced by id. Chroma returns metadata dictionaries, not object
    graphs, so denormalising here keeps citation construction a pure
    transformation with no second lookup and no chance of a dangling
    reference.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="'{doc_id}::{index}'.")
    doc_id: str
    index: int = Field(ge=0, description="Position within the parent document.")
    text: str
    title: str
    source: str
    licence: DocumentLicence
    url: str | None = None
    topics: list[str] = Field(default_factory=list)

    def to_metadata(self) -> dict[str, str | int]:
        """Return a Chroma-compatible metadata mapping.

        Chroma accepts only scalar metadata values, so the topic list is
        flattened to a delimited string and re-split on read.
        """
        return {
            "doc_id": self.doc_id,
            "index": self.index,
            "title": self.title,
            "source": self.source,
            "licence": self.licence.value,
            "url": self.url or "",
            "topics": "|".join(self.topics),
        }


class RetrievedChunk(BaseModel):
    """A chunk returned by a similarity search, with its score."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalised similarity, where 1.0 is an exact match.",
    )

    def meets(self, threshold: float) -> bool:
        """True when this result clears the configured relevance floor."""
        return self.score >= threshold


class RetrievalResult(BaseModel):
    """The full outcome of one knowledge-base query."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    took_ms: int = Field(default=0, ge=0)
    degraded: bool = Field(
        default=False,
        description=(
            "True when retrieval failed and the turn proceeded ungrounded. "
            "Surfaced to the user as a reduced-confidence notice rather than "
            "silently pretending the knowledge base was consulted."
        ),
    )

    @property
    def has_context(self) -> bool:
        """True when at least one chunk survived filtering."""
        return bool(self.chunks)

    def top(self, limit: int) -> list[RetrievedChunk]:
        """Return the ``limit`` highest-scoring chunks."""
        return sorted(self.chunks, key=lambda item: item.score, reverse=True)[:limit]
