"""RAG (Retrieval-Augmented Generation) package.

Public surface:

- :func:`~src.rag.ingestion.load_documents` — parse JSON files into domain models.
- :func:`~src.rag.ingestion.chunk_document` — split one document into chunks.
- :class:`~src.rag.ingestion.Ingester` — embed chunks and upsert to ChromaDB.
- :func:`~src.rag.ingestion.build_knowledge_base` — one-call helper for the CLI.
- :class:`~src.rag.retriever.Retriever` — query ChromaDB and return a :class:`~src.models.rag.RetrievalResult`.
"""

from src.rag.ingestion import Ingester, build_knowledge_base, chunk_document, load_documents
from src.rag.retriever import Retriever

__all__ = [
    "build_knowledge_base",
    "chunk_document",
    "Ingester",
    "load_documents",
    "Retriever",
]
