"""Tests for the Phase 5 RAG layer.

All tests are hermetic: no real ChromaDB, no real sentence-transformers,
no network. Fakes are defined here (not in fakes.py, which is LLM-only)
because they are RAG-specific and not shared with other test modules.

Coverage:
    ingestion
        load_documents          — happy path, schema validation, missing dir
        chunk_document          — short para, long para splitting, empty body
        Ingester.ingest         — upsert called with correct IDs/embeddings
        build_knowledge_base    — end-to-end with fakes

    retriever
        _distance_to_score      — correctness at key distances
        _chunk_from_metadata    — round-trip of to_metadata() + reconstruction
        Retriever.retrieve      — results above/below threshold, empty corpus,
                                   ChromaDB exception → degraded result
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.models.rag import (
    Chunk,
    DocumentLicence,
    KBDocument,
    RetrievalResult,
    RetrievedChunk,
)
from src.rag.ingestion import (
    MAX_CHUNK_CHARS,
    Ingester,
    build_knowledge_base,
    chunk_document,
    load_documents,
)
from src.rag.retriever import (
    DEFAULT_SCORE_THRESHOLD,
    Retriever,
    _chunk_from_metadata,
    _distance_to_score,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbeddingModel:
    """Returns deterministic 3-dimensional unit embeddings.

    The vector is ``[hash_mod, 0.0, 0.0]`` where ``hash_mod`` is a float
    derived from the text hash. Real semantics are irrelevant; we just need
    something shaped like ``numpy``-like output with ``.tolist()``.
    """

    class _ArrayLike:
        """Mimics the behaviour of ``numpy.ndarray.tolist()``."""

        def __init__(self, data: list[list[float]]) -> None:
            self._data = data

        def tolist(self) -> list[list[float]]:
            return self._data

    def encode(self, texts: list[str]) -> "_ArrayLike":
        vectors = [
            [float(abs(hash(t)) % 100) / 100.0, 0.0, 0.0] for t in texts
        ]
        return self._ArrayLike(vectors)


class FakeChromaCollection:
    """In-memory ChromaDB collection fake.

    Stores documents in a plain list; ``query()`` returns all stored items
    with a configurable distance list so tests can control threshold behaviour.

    Args:
        distances: If supplied, ``query()`` returns these distances for the
            first ``len(distances)`` results. Defaults to ``[0.0]`` (exact
            match, score=1.0) for every stored document.
        raise_on_query: If set, ``query()`` raises this exception.
    """

    def __init__(
        self,
        distances: list[float] | None = None,
        raise_on_query: Exception | None = None,
    ) -> None:
        self._ids: list[str] = []
        self._embeddings: list[list[float]] = []
        self._metadatas: list[dict] = []
        self._documents: list[str] = []
        self._distances = distances
        self._raise_on_query = raise_on_query

        # Track calls for assertions in tests.
        self.upsert_calls: list[dict] = []
        self.query_calls: list[dict] = []

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        self.upsert_calls.append(
            {"ids": ids, "embeddings": embeddings, "metadatas": metadatas, "documents": documents}
        )
        # Upsert semantics: overwrite existing, add new.
        for i, id_ in enumerate(ids):
            if id_ in self._ids:
                idx = self._ids.index(id_)
                self._embeddings[idx] = embeddings[i]
                self._metadatas[idx] = metadatas[i]
                self._documents[idx] = documents[i]
            else:
                self._ids.append(id_)
                self._embeddings.append(embeddings[i])
                self._metadatas.append(metadatas[i])
                self._documents.append(documents[i])

    def query(
        self,
        query_texts: list[str],
        n_results: int,
        include: list[str],
    ) -> dict[str, list]:
        self.query_calls.append({"query_texts": query_texts, "n_results": n_results})

        if self._raise_on_query is not None:
            raise self._raise_on_query

        count = min(n_results, len(self._ids))
        docs = self._documents[:count]
        metas = self._metadatas[:count]

        if self._distances is not None:
            dists = self._distances[:count]
        else:
            dists = [0.0] * count

        return {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_doc(**overrides) -> KBDocument:
    """Return a minimal valid KBDocument, with optional field overrides."""
    defaults = dict(
        doc_id="test-doc",
        title="Test Document",
        content="First paragraph with some text.\n\nSecond paragraph here.",
        source="Original",
        licence=DocumentLicence.ORIGINAL,
        url=None,
        topics=["test"],
    )
    defaults.update(overrides)
    return KBDocument(**defaults)


def _make_chunk(**overrides) -> Chunk:
    """Return a minimal valid Chunk."""
    defaults = dict(
        chunk_id="test-doc::0",
        doc_id="test-doc",
        index=0,
        text="Some text here.",
        title="Test Document",
        source="Original",
        licence=DocumentLicence.ORIGINAL,
        url=None,
        topics=["test"],
    )
    defaults.update(overrides)
    return Chunk(**defaults)


@pytest.fixture()
def kb_dir(tmp_path: Path) -> Path:
    """A temporary knowledge-base directory with one valid JSON document."""
    doc = _make_doc()
    doc_path = tmp_path / "test-doc.json"
    doc_path.write_text(doc.model_dump_json(), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# ingestion.load_documents
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    def test_loads_valid_document(self, kb_dir: Path) -> None:
        docs = load_documents(kb_dir)
        assert len(docs) == 1
        assert docs[0].doc_id == "test-doc"

    def test_raises_on_missing_directory(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            load_documents(missing)

    def test_loads_multiple_documents(self, tmp_path: Path) -> None:
        for i in range(3):
            doc = _make_doc(doc_id=f"doc-{i}", title=f"Document {i}")
            (tmp_path / f"doc-{i}.json").write_text(
                doc.model_dump_json(), encoding="utf-8"
            )
        docs = load_documents(tmp_path)
        assert len(docs) == 3

    def test_raises_on_schema_violation(self, tmp_path: Path) -> None:
        bad = {"doc_id": "bad", "title": "Bad"}  # missing required fields
        (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(Exception):  # pydantic ValidationError
            load_documents(tmp_path)

    def test_skips_non_json_files(self, tmp_path: Path) -> None:
        doc = _make_doc()
        (tmp_path / "doc.json").write_text(doc.model_dump_json(), encoding="utf-8")
        (tmp_path / "README.md").write_text("# Notes", encoding="utf-8")
        (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
        docs = load_documents(tmp_path)
        assert len(docs) == 1

    def test_licence_enum_validated(self, tmp_path: Path) -> None:
        raw = _make_doc().model_dump()
        raw["licence"] = "us_gov_public_domain"
        (tmp_path / "gov.json").write_text(json.dumps(raw), encoding="utf-8")
        docs = load_documents(tmp_path)
        assert docs[0].licence == DocumentLicence.US_GOV_PUBLIC_DOMAIN

    def test_invalid_licence_raises(self, tmp_path: Path) -> None:
        raw = _make_doc().model_dump()
        raw["licence"] = "not_a_real_licence"
        (tmp_path / "bad.json").write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(Exception):
            load_documents(tmp_path)


# ---------------------------------------------------------------------------
# ingestion.chunk_document
# ---------------------------------------------------------------------------


class TestChunkDocument:
    def test_two_paragraphs_two_chunks(self) -> None:
        doc = _make_doc(content="Para one.\n\nPara two.")
        chunks = chunk_document(doc)
        assert len(chunks) == 2
        assert chunks[0].text == "Para one."
        assert chunks[1].text == "Para two."

    def test_chunk_ids_are_sequential(self) -> None:
        doc = _make_doc(content="A.\n\nB.\n\nC.")
        chunks = chunk_document(doc)
        assert [c.chunk_id for c in chunks] == ["test-doc::0", "test-doc::1", "test-doc::2"]

    def test_chunk_inherits_provenance(self) -> None:
        doc = _make_doc(
            doc_id="hydration",
            source="MedlinePlus",
            licence=DocumentLicence.US_GOV_PUBLIC_DOMAIN,
            url="https://medlineplus.gov/hydration",
            topics=["hydration", "nutrition"],
        )
        chunks = chunk_document(doc)
        for c in chunks:
            assert c.doc_id == "hydration"
            assert c.source == "MedlinePlus"
            assert c.licence == DocumentLicence.US_GOV_PUBLIC_DOMAIN
            assert c.url == "https://medlineplus.gov/hydration"
            assert c.topics == ["hydration", "nutrition"]

    def test_long_paragraph_split(self) -> None:
        # Build a paragraph that exceeds MAX_CHUNK_CHARS using sentence-like text.
        sentence = "This is a sentence with some meaningful content. "
        long_para = sentence * (MAX_CHUNK_CHARS // len(sentence) + 5)
        doc = _make_doc(content=long_para)
        chunks = chunk_document(doc)
        # Must produce more than one chunk.
        assert len(chunks) > 1
        # Every chunk must be within the limit (except potentially the last one
        # in a degenerate all-one-sentence case, which is acceptable).
        for c in chunks[:-1]:
            assert len(c.text) <= MAX_CHUNK_CHARS + len(sentence)

    def test_empty_paragraphs_skipped(self) -> None:
        doc = _make_doc(content="Para one.\n\n\n\nPara two.")
        chunks = chunk_document(doc)
        assert len(chunks) == 2

    def test_single_paragraph_single_chunk(self) -> None:
        doc = _make_doc(content="Just one paragraph.")
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].index == 0

    def test_chunk_indices_are_contiguous(self) -> None:
        doc = _make_doc(content="A.\n\nB.\n\nC.\n\nD.")
        chunks = chunk_document(doc)
        assert [c.index for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# ingestion.Ingester
# ---------------------------------------------------------------------------


class TestIngester:
    def _make_chunks(self, n: int = 3) -> list[Chunk]:
        return [
            _make_chunk(
                chunk_id=f"doc::{i}",
                doc_id="doc",
                index=i,
                text=f"Chunk text number {i}.",
            )
            for i in range(n)
        ]

    def test_upsert_called_with_correct_ids(self) -> None:
        collection = FakeChromaCollection()
        model = FakeEmbeddingModel()
        ingester = Ingester(collection, model)
        chunks = self._make_chunks(3)

        ingester.ingest(chunks)

        assert len(collection.upsert_calls) == 1
        call = collection.upsert_calls[0]
        assert call["ids"] == ["doc::0", "doc::1", "doc::2"]

    def test_upsert_receives_correct_document_text(self) -> None:
        collection = FakeChromaCollection()
        ingester = Ingester(collection, FakeEmbeddingModel())
        chunks = self._make_chunks(2)

        ingester.ingest(chunks)

        call = collection.upsert_calls[0]
        assert call["documents"] == ["Chunk text number 0.", "Chunk text number 1."]

    def test_upsert_receives_embeddings_matching_chunk_count(self) -> None:
        collection = FakeChromaCollection()
        ingester = Ingester(collection, FakeEmbeddingModel())
        chunks = self._make_chunks(4)

        ingester.ingest(chunks)

        call = collection.upsert_calls[0]
        assert len(call["embeddings"]) == 4
        # Each embedding is 3-dimensional (FakeEmbeddingModel).
        assert all(len(e) == 3 for e in call["embeddings"])

    def test_metadata_contains_provenance_fields(self) -> None:
        collection = FakeChromaCollection()
        ingester = Ingester(collection, FakeEmbeddingModel())
        chunk = _make_chunk(
            source="MedlinePlus",
            licence=DocumentLicence.US_GOV_PUBLIC_DOMAIN,
            topics=["hydration", "nutrition"],
        )

        ingester.ingest([chunk])

        meta = collection.upsert_calls[0]["metadatas"][0]
        assert meta["source"] == "MedlinePlus"
        assert meta["licence"] == "us_gov_public_domain"
        assert meta["topics"] == "hydration|nutrition"

    def test_empty_chunk_list_returns_zero(self) -> None:
        collection = FakeChromaCollection()
        ingester = Ingester(collection, FakeEmbeddingModel())
        result = ingester.ingest([])
        assert result == 0
        assert len(collection.upsert_calls) == 0

    def test_ingest_returns_chunk_count(self) -> None:
        collection = FakeChromaCollection()
        ingester = Ingester(collection, FakeEmbeddingModel())
        chunks = self._make_chunks(5)
        result = ingester.ingest(chunks)
        assert result == 5

    def test_upsert_is_idempotent(self) -> None:
        """Upserting the same chunks twice should not duplicate them."""
        collection = FakeChromaCollection()
        ingester = Ingester(collection, FakeEmbeddingModel())
        chunks = self._make_chunks(2)

        ingester.ingest(chunks)
        ingester.ingest(chunks)  # second run — same IDs

        # Both upsert calls happened, but internal store deduplicates by ID.
        assert len(collection.upsert_calls) == 2
        assert len(collection._ids) == 2  # not 4


# ---------------------------------------------------------------------------
# ingestion.build_knowledge_base
# ---------------------------------------------------------------------------


class TestBuildKnowledgeBase:
    def test_returns_correct_summary(self, kb_dir: Path) -> None:
        collection = FakeChromaCollection()
        model = FakeEmbeddingModel()
        summary = build_knowledge_base(kb_dir, collection, model)

        assert summary["documents"] == 1
        assert summary["chunks"] >= 1
        assert summary["elapsed_ms"] >= 0

    def test_all_chunks_upserted(self, kb_dir: Path) -> None:
        collection = FakeChromaCollection()
        model = FakeEmbeddingModel()
        build_knowledge_base(kb_dir, collection, model)

        assert len(collection.upsert_calls) == 1
        ids = collection.upsert_calls[0]["ids"]
        # All IDs should start with the document's doc_id.
        assert all(id_.startswith("test-doc::") for id_ in ids)

    def test_raises_on_missing_kb_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        with pytest.raises(FileNotFoundError):
            build_knowledge_base(missing, FakeChromaCollection(), FakeEmbeddingModel())


# ---------------------------------------------------------------------------
# retriever._distance_to_score
# ---------------------------------------------------------------------------


class TestDistanceToScore:
    def test_zero_distance_gives_score_one(self) -> None:
        assert _distance_to_score(0.0) == pytest.approx(1.0)

    def test_positive_distance_gives_score_less_than_one(self) -> None:
        assert _distance_to_score(1.0) < 1.0

    def test_score_decreases_as_distance_increases(self) -> None:
        scores = [_distance_to_score(d) for d in [0.0, 0.5, 1.0, 2.0, 5.0]]
        assert scores == sorted(scores, reverse=True)

    def test_score_always_positive(self) -> None:
        for d in [0.0, 0.1, 1.0, 10.0, 100.0]:
            assert _distance_to_score(d) > 0.0

    def test_score_always_at_most_one(self) -> None:
        for d in [0.0, 0.01, 1.0, 100.0]:
            assert _distance_to_score(d) <= 1.0

    def test_threshold_correspondence(self) -> None:
        # score=0.45 corresponds to distance ≈ 1.22 — 1/(1+1.22) = 0.449…
        score = _distance_to_score(1.22)
        assert score == pytest.approx(0.45, abs=0.01)


# ---------------------------------------------------------------------------
# retriever._chunk_from_metadata
# ---------------------------------------------------------------------------


class TestChunkFromMetadata:
    def test_round_trip_via_to_metadata(self) -> None:
        original = _make_chunk(
            url="https://example.com",
            topics=["a", "b", "c"],
        )
        meta = original.to_metadata()
        reconstructed = _chunk_from_metadata(meta, original.text)

        assert reconstructed.doc_id == original.doc_id
        assert reconstructed.index == original.index
        assert reconstructed.text == original.text
        assert reconstructed.source == original.source
        assert reconstructed.licence == original.licence
        assert reconstructed.url == original.url
        assert reconstructed.topics == original.topics

    def test_empty_topics_string_gives_empty_list(self) -> None:
        chunk = _make_chunk(topics=[])
        meta = chunk.to_metadata()
        assert meta["topics"] == ""
        reconstructed = _chunk_from_metadata(meta, chunk.text)
        assert reconstructed.topics == []

    def test_empty_url_string_gives_none(self) -> None:
        chunk = _make_chunk(url=None)
        meta = chunk.to_metadata()
        assert meta["url"] == ""
        reconstructed = _chunk_from_metadata(meta, chunk.text)
        assert reconstructed.url is None

    def test_licence_enum_reconstructed(self) -> None:
        chunk = _make_chunk(licence=DocumentLicence.US_GOV_PUBLIC_DOMAIN)
        meta = chunk.to_metadata()
        reconstructed = _chunk_from_metadata(meta, chunk.text)
        assert reconstructed.licence == DocumentLicence.US_GOV_PUBLIC_DOMAIN


# ---------------------------------------------------------------------------
# Retriever.retrieve
# ---------------------------------------------------------------------------


class TestRetriever:
    def _make_stored_chunk(self, doc_id: str = "test-doc", idx: int = 0) -> Chunk:
        return _make_chunk(
            chunk_id=f"{doc_id}::{idx}",
            doc_id=doc_id,
            index=idx,
            text=f"Relevant text for chunk {idx}.",
        )

    def _collection_with_chunks(
        self, chunks: list[Chunk], distances: list[float]
    ) -> FakeChromaCollection:
        collection = FakeChromaCollection(distances=distances)
        model = FakeEmbeddingModel()
        ingester = Ingester(collection, model)
        ingester.ingest(chunks)
        return collection

    def test_returns_retrieval_result_type(self) -> None:
        chunk = self._make_stored_chunk()
        collection = self._collection_with_chunks([chunk], distances=[0.0])
        retriever = Retriever(collection)
        result = retriever.retrieve("some query")
        assert isinstance(result, RetrievalResult)

    def test_result_above_threshold_included(self) -> None:
        chunk = self._make_stored_chunk()
        # distance=0.0 → score=1.0, well above any threshold.
        collection = self._collection_with_chunks([chunk], distances=[0.0])
        retriever = Retriever(collection, score_threshold=0.45)
        result = retriever.retrieve("hydration question")
        assert result.has_context
        assert len(result.chunks) == 1

    def test_result_below_threshold_excluded(self) -> None:
        chunk = self._make_stored_chunk()
        # distance=10.0 → score ≈ 0.09, below any reasonable threshold.
        collection = self._collection_with_chunks([chunk], distances=[10.0])
        retriever = Retriever(collection, score_threshold=0.45)
        result = retriever.retrieve("unrelated query")
        assert not result.has_context
        assert result.chunks == []

    def test_mixed_threshold_only_passes_good_ones(self) -> None:
        chunks = [self._make_stored_chunk(idx=i) for i in range(3)]
        # Two close, one far.
        distances = [0.1, 0.2, 5.0]
        collection = self._collection_with_chunks(chunks, distances=distances)
        retriever = Retriever(collection, score_threshold=0.45)
        result = retriever.retrieve("health question")
        assert len(result.chunks) == 2

    def test_chromadb_exception_returns_degraded(self) -> None:
        collection = FakeChromaCollection(
            raise_on_query=RuntimeError("ChromaDB down")
        )
        retriever = Retriever(collection)
        result = retriever.retrieve("any query")
        assert result.degraded
        assert not result.has_context

    def test_empty_collection_returns_no_context(self) -> None:
        collection = FakeChromaCollection(distances=[])
        retriever = Retriever(collection)
        result = retriever.retrieve("any query")
        assert not result.has_context
        assert result.degraded is False

    def test_query_text_passed_to_collection(self) -> None:
        collection = FakeChromaCollection(distances=[])
        retriever = Retriever(collection)
        retriever.retrieve("my specific query")
        assert collection.query_calls[0]["query_texts"] == ["my specific query"]

    def test_retrieved_chunk_score_is_normalised(self) -> None:
        chunk = self._make_stored_chunk()
        collection = self._collection_with_chunks([chunk], distances=[0.0])
        retriever = Retriever(collection, score_threshold=0.0)
        result = retriever.retrieve("query")
        assert result.chunks[0].score == pytest.approx(1.0)

    def test_result_query_field_preserved(self) -> None:
        collection = FakeChromaCollection(distances=[])
        retriever = Retriever(collection)
        result = retriever.retrieve("my query")
        assert result.query == "my query"

    def test_took_ms_is_non_negative(self) -> None:
        collection = FakeChromaCollection(distances=[])
        retriever = Retriever(collection)
        result = retriever.retrieve("query")
        assert result.took_ms >= 0

    def test_retrieved_chunk_meets_threshold(self) -> None:
        chunk = self._make_stored_chunk()
        collection = self._collection_with_chunks([chunk], distances=[0.5])
        threshold = 0.45
        retriever = Retriever(collection, score_threshold=threshold)
        result = retriever.retrieve("query")
        if result.has_context:
            for rc in result.chunks:
                assert rc.meets(threshold)

    def test_top_method_returns_highest_scoring_first(self) -> None:
        chunks = [self._make_stored_chunk(idx=i) for i in range(3)]
        # Scores will be 1/(1+0.1), 1/(1+0.3), 1/(1+0.2).
        # Expected descending order: idx 0, idx 2, idx 1.
        distances = [0.1, 0.3, 0.2]
        collection = self._collection_with_chunks(chunks, distances=distances)
        retriever = Retriever(collection, score_threshold=0.0)
        result = retriever.retrieve("query")
        top2 = result.top(2)
        assert top2[0].score >= top2[1].score

    def test_custom_n_results_passed_to_collection(self) -> None:
        collection = FakeChromaCollection(distances=[])
        retriever = Retriever(collection, n_results=3)
        retriever.retrieve("query")
        assert collection.query_calls[0]["n_results"] == 3
