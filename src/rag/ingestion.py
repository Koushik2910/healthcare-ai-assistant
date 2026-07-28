"""Knowledge-base ingestion pipeline.

Design rationale
----------------
Chunking strategy: paragraph-boundary splitting with a character-length cap.
Health documents have natural paragraph breaks that align with semantic units.
A paragraph about "how much water to drink" should not be split mid-sentence
just because it hit a character limit. The pipeline therefore splits on double-
newline (``\\n\\n``) first, then — only if an individual paragraph still exceeds
``MAX_CHUNK_CHARS`` — splits further at the nearest sentence boundary (``". "``
or ``"\\n"``). This produces semantically coherent chunks without requiring
``nltk`` or ``spacy``.

Embedding: ``all-MiniLM-L6-v2`` via ``sentence-transformers``. 384-dimensional
dense vectors, ~80 MB on first download, runs fully offline thereafter.

Persistence: ChromaDB ``upsert`` (not ``add``) so this pipeline is idempotent
— running ``scripts/ingest.py`` twice does not duplicate chunks.

Provenance: every ``Chunk`` inherits ``doc_id``, ``licence``, ``source``, and
``url`` from its parent ``KBDocument``. ChromaDB stores these as metadata so
the retriever can reconstruct fully-cited ``RetrievedChunk`` objects without a
second lookup.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.models.rag import Chunk, DocumentLicence, KBDocument

if TYPE_CHECKING:
    # These imports are only resolved at runtime (real or fake); the type
    # annotations keep mypy happy without hard-importing the library at
    # module load time, which would break tests that inject a fake client.
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------------

#: Maximum characters per chunk before the paragraph is further split.
MAX_CHUNK_CHARS: int = 600

#: Sentence-level split markers tried in order when a paragraph is too long.
_SENTENCE_SPLITS = (". ", ".\n", "! ", "? ")


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_documents(kb_dir: Path) -> list[KBDocument]:
    """Load every ``*.json`` file in *kb_dir* as a :class:`~src.models.rag.KBDocument`.

    Pydantic validates the schema on load, so a missing ``licence`` field is a
    load-time ``ValidationError``, not a silent omission.

    Args:
        kb_dir: Directory containing JSON knowledge-base files.

    Returns:
        List of validated :class:`~src.models.rag.KBDocument` objects.

    Raises:
        FileNotFoundError: If *kb_dir* does not exist.
        pydantic.ValidationError: If any file fails schema validation.
    """
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    docs: list[KBDocument] = []
    json_files = sorted(kb_dir.glob("*.json"))
    logger.info("Loading %d document(s) from %s", len(json_files), kb_dir)

    for path in json_files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        doc = KBDocument.model_validate(raw)
        docs.append(doc)
        logger.debug("Loaded document '%s' (%d chars)", doc.doc_id, doc.char_count)

    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_document(doc: KBDocument) -> list[Chunk]:
    """Split a :class:`~src.models.rag.KBDocument` into retrievable chunks.

    Algorithm:
    1. Split the document body on ``\\n\\n`` (paragraph boundaries).
    2. Drop empty paragraphs.
    3. If a paragraph exceeds :data:`MAX_CHUNK_CHARS`, split it further at
       the nearest sentence-ending punctuation marker.
    4. Wrap each text fragment as a :class:`~src.models.rag.Chunk` with full
       provenance from the parent document.

    Args:
        doc: Source document to chunk.

    Returns:
        Ordered list of :class:`~src.models.rag.Chunk` objects.
    """
    raw_paragraphs = [p.strip() for p in doc.content.split("\n\n")]
    paragraphs = [p for p in raw_paragraphs if p]

    fragments: list[str] = []
    for para in paragraphs:
        if len(para) <= MAX_CHUNK_CHARS:
            fragments.append(para)
        else:
            fragments.extend(_split_long_paragraph(para))

    chunks: list[Chunk] = []
    for idx, text in enumerate(fragments):
        chunk = Chunk(
            chunk_id=f"{doc.doc_id}::{idx}",
            doc_id=doc.doc_id,
            index=idx,
            text=text,
            title=doc.title,
            source=doc.source,
            licence=doc.licence,
            url=doc.url,
            topics=doc.topics,
        )
        chunks.append(chunk)

    logger.debug(
        "Document '%s' → %d chunk(s) from %d paragraph(s)",
        doc.doc_id,
        len(chunks),
        len(paragraphs),
    )
    return chunks


def _split_long_paragraph(para: str) -> list[str]:
    """Break *para* at sentence boundaries until all fragments are within limit.

    Uses a greedy left-to-right scan: accumulate sentences until the next
    addition would exceed :data:`MAX_CHUNK_CHARS`, then start a new fragment.
    Falls back to a hard character split only if no sentence boundary is found.

    Args:
        para: A paragraph string that exceeds :data:`MAX_CHUNK_CHARS`.

    Returns:
        List of shorter text fragments, all ``<= MAX_CHUNK_CHARS`` where
        possible (a single very long sentence may still exceed the limit).
    """
    # Find a sentence split marker that exists in this paragraph.
    splitter: str | None = None
    for marker in _SENTENCE_SPLITS:
        if marker in para:
            splitter = marker
            break

    if splitter is None:
        # No sentence boundary found — hard split.
        return [
            para[i : i + MAX_CHUNK_CHARS]
            for i in range(0, len(para), MAX_CHUNK_CHARS)
        ]

    sentences = para.split(splitter)
    # Re-attach the splitter (except after the final sentence).
    sentences = [
        s + splitter if i < len(sentences) - 1 else s
        for i, s in enumerate(sentences)
    ]

    fragments: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        if current and len(current) + len(sentence) > MAX_CHUNK_CHARS:
            fragments.append(current.strip())
            current = sentence
        else:
            current += sentence

    if current.strip():
        fragments.append(current.strip())

    return fragments or [para]


# ---------------------------------------------------------------------------
# Ingestion into ChromaDB
# ---------------------------------------------------------------------------


class Ingester:
    """Embeds chunks and upserts them into a ChromaDB collection.

    Keeping the ChromaDB client and embedding model as constructor arguments
    (rather than instantiating them here) makes this class testable with fakes
    without any monkey-patching.

    Args:
        chroma_collection: A ChromaDB collection object (real or fake) with
            ``.upsert(ids, embeddings, metadatas, documents)`` support.
        embedding_model: An object with ``.encode(texts) -> list[list[float]]``
            — typically a ``SentenceTransformer`` instance.
    """

    def __init__(self, chroma_collection, embedding_model) -> None:
        self._collection = chroma_collection
        self._model = embedding_model

    def ingest(self, chunks: list[Chunk]) -> int:
        """Embed and upsert *chunks* into the collection.

        ``upsert`` is used instead of ``add`` so this method is idempotent:
        re-ingesting the same chunk IDs overwrites rather than duplicates.

        Args:
            chunks: Chunks to embed and store.

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            logger.warning("ingest() called with empty chunk list — nothing to do")
            return 0

        t0 = time.monotonic()
        texts = [c.text for c in chunks]

        logger.info("Embedding %d chunk(s)…", len(chunks))
        embeddings: list[list[float]] = self._model.encode(texts).tolist()

        ids = [c.chunk_id for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Upserted %d chunk(s) in %d ms", len(chunks), elapsed_ms
        )
        return len(chunks)


# ---------------------------------------------------------------------------
# High-level helper used by scripts/ingest.py
# ---------------------------------------------------------------------------


def build_knowledge_base(
    kb_dir: Path,
    chroma_collection,
    embedding_model,
) -> dict[str, int]:
    """Load, chunk, and ingest all documents in *kb_dir*.

    This is the single function that ``scripts/ingest.py`` calls. It returns
    a summary dict so the CLI can print meaningful progress without knowing
    about internal objects.

    Args:
        kb_dir: Path to the directory containing ``*.json`` knowledge-base files.
        chroma_collection: Initialised ChromaDB collection (persistent or fake).
        embedding_model: ``SentenceTransformer`` (or compatible fake) instance.

    Returns:
        ``{"documents": int, "chunks": int, "elapsed_ms": int}``
    """
    t0 = time.monotonic()

    docs = load_documents(kb_dir)
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    ingester = Ingester(chroma_collection, embedding_model)
    ingester.ingest(all_chunks)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    summary = {
        "documents": len(docs),
        "chunks": len(all_chunks),
        "elapsed_ms": elapsed_ms,
    }
    logger.info(
        "Knowledge base built: %d docs → %d chunks in %d ms",
        summary["documents"],
        summary["chunks"],
        summary["elapsed_ms"],
    )
    return summary
