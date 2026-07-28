"""Knowledge-base retriever.

Design rationale
----------------
Score normalisation: ChromaDB returns L2 (Euclidean) distances — lower means
more similar. The :class:`~src.models.rag.RetrievedChunk` model stores a
*similarity* score in ``[0, 1]`` (higher = better). The mapping used is::

    score = 1 / (1 + distance)

This is a standard monotone transformation: distance=0 → score=1.0 (exact
match); distance=1.0 → score=0.5; distance→∞ → score→0. It is smooth,
always-positive, and requires no knowledge of the corpus's distance
distribution — unlike, say, min-max normalisation, which would shift if the
corpus changes.

Default threshold: 0.45 (corresponding to roughly L2 distance=1.2 for
``all-MiniLM-L6-v2`` on general text). This is intentionally conservative —
a result that barely passes feels like a weak retrieval and dilutes context.
The threshold is configurable at construction time for tuning.

Metadata reconstruction: ChromaDB stores chunk metadata as flat scalars.
``topics`` is stored as a ``"|"``-joined string (see
:meth:`~src.models.rag.Chunk.to_metadata`) and is re-split here. All other
fields map 1:1 from metadata keys to ``Chunk`` fields.

Graceful degradation: if ChromaDB raises, :meth:`Retriever.retrieve` catches,
logs, and returns ``RetrievalResult(query=..., degraded=True)``. This keeps the
chat turn alive — the model answers from its own knowledge, and the UI shows a
reduced-confidence notice rather than a hard error.
"""

from __future__ import annotations

import logging
import time

from src.models.rag import Chunk, DocumentLicence, RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)

#: Default minimum similarity score (0–1). Below this, results are discarded.
DEFAULT_SCORE_THRESHOLD: float = 0.45

#: Maximum number of candidates fetched from ChromaDB before threshold filtering.
_N_RESULTS: int = 5


def _distance_to_score(distance: float) -> float:
    """Convert an L2 distance to a [0, 1] similarity score.

    Uses ``1 / (1 + distance)`` — a monotone, always-positive mapping that
    requires no corpus-specific calibration.

    Args:
        distance: L2 distance returned by ChromaDB (non-negative).

    Returns:
        Similarity in ``[0, 1]``, where 1.0 is an exact match.
    """
    return 1.0 / (1.0 + distance)


def _chunk_from_metadata(metadata: dict, document: str) -> Chunk:
    """Reconstruct a :class:`~src.models.rag.Chunk` from ChromaDB metadata.

    ChromaDB stores only scalar metadata values, so ``topics`` was stored as
    a ``"|"``-joined string by :meth:`~src.models.rag.Chunk.to_metadata` and
    is re-split here.  An empty string produces an empty list.

    Args:
        metadata: The metadata dict returned by ChromaDB for one result.
        document: The original chunk text stored alongside the metadata.

    Returns:
        A fully-populated :class:`~src.models.rag.Chunk`.
    """
    raw_topics: str = metadata.get("topics", "")
    topics = [t for t in raw_topics.split("|") if t]

    return Chunk(
        chunk_id=f"{metadata['doc_id']}::{metadata['index']}",
        doc_id=metadata["doc_id"],
        index=int(metadata["index"]),
        text=document,
        title=metadata["title"],
        source=metadata["source"],
        licence=DocumentLicence(metadata["licence"]),
        url=metadata.get("url") or None,
        topics=topics,
    )


class Retriever:
    """Queries a ChromaDB collection and returns scored, filtered results.

    The retriever is intentionally thin: it converts between ChromaDB's
    distance-centric API and the domain models, applies the score threshold,
    and hands the structured result back to :class:`~src.services.chat_service.ChatService`.
    No ranking, re-scoring, or cross-encoding happens here — the 384-dim
    ``all-MiniLM-L6-v2`` cosine similarity is good enough for a small,
    tightly-scoped health corpus.

    Args:
        chroma_collection: A ChromaDB collection with a ``.query()`` method.
        score_threshold: Minimum similarity score for a chunk to be included
            in the result. Defaults to :data:`DEFAULT_SCORE_THRESHOLD`.
        n_results: How many nearest neighbours to fetch from ChromaDB before
            threshold filtering. Defaults to :data:`_N_RESULTS`.
    """

    def __init__(
        self,
        chroma_collection,
        *,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        n_results: int = _N_RESULTS,
    ) -> None:
        self._collection = chroma_collection
        self._score_threshold = score_threshold
        self._n_results = n_results

    def retrieve(self, query: str) -> RetrievalResult:
        """Query the knowledge base for context relevant to *query*.

        Fetches up to :attr:`n_results` nearest neighbours from ChromaDB,
        converts L2 distances to similarity scores, drops results below
        :attr:`score_threshold`, and returns the survivors as a
        :class:`~src.models.rag.RetrievalResult`.

        If the collection is empty or ChromaDB raises, the method logs the
        problem and returns a degraded result so the chat turn can proceed.

        Args:
            query: The user's message, used as the embedding query.

        Returns:
            :class:`~src.models.rag.RetrievalResult` with ``has_context=True``
            when at least one chunk passed the threshold, or
            ``degraded=True`` when ChromaDB raised an exception.
        """
        t0 = time.monotonic()

        try:
            raw = self._collection.query(
                query_texts=[query],
                n_results=self._n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ChromaDB query failed: %s", exc, exc_info=True)
            return RetrievalResult(query=query, degraded=True)

        # ChromaDB returns lists-of-lists (one sub-list per query).
        # We always pass exactly one query, so index [0] everywhere.
        documents: list[str] = raw.get("documents", [[]])[0] or []
        metadatas: list[dict] = raw.get("metadatas", [[]])[0] or []
        distances: list[float] = raw.get("distances", [[]])[0] or []

        retrieved: list[RetrievedChunk] = []
        for doc_text, meta, dist in zip(documents, metadatas, distances):
            score = _distance_to_score(dist)
            if score < self._score_threshold:
                logger.debug(
                    "Chunk '%s::%s' scored %.3f — below threshold %.3f, skipped",
                    meta.get("doc_id", "?"),
                    meta.get("index", "?"),
                    score,
                    self._score_threshold,
                )
                continue

            try:
                chunk = _chunk_from_metadata(meta, doc_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not reconstruct chunk from metadata %s: %s", meta, exc
                )
                continue

            retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Retrieval: query=%r → %d/%d chunks passed threshold=%.2f in %d ms",
            query[:60],
            len(retrieved),
            len(documents),
            self._score_threshold,
            elapsed_ms,
        )

        return RetrievalResult(
            query=query,
            chunks=retrieved,
            took_ms=elapsed_ms,
        )
