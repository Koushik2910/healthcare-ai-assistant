"""scripts/ingest.py — (Re)build the knowledge base.

Usage (PowerShell, from project root with venv active):

    python scripts\\ingest.py

Options:
    --kb-dir PATH    Override the knowledge-base directory (default: data/knowledge_base)
    --chroma-dir PATH  Override the ChromaDB persist directory (default: data/chroma)
    --collection NAME  ChromaDB collection name (default: health_kb)
    --model NAME     SentenceTransformer model name (default: all-MiniLM-L6-v2)
    --dry-run        Load and chunk only; skip embedding and ChromaDB upsert

The script is intentionally idempotent: running it twice overwrites existing
chunks rather than duplicating them (ChromaDB ``upsert`` semantics).

Exit codes:
    0 — success
    1 — fatal error (missing directory, schema validation failure, etc.)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow `python scripts\ingest.py` to resolve `src.*` without an editable install.
# Identical pattern to scripts/preflight.py and scripts/smoke_test_llm.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or rebuild the Healthcare AI Assistant knowledge base."
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=Path("data/knowledge_base"),
        help="Directory containing JSON knowledge-base files (default: data/knowledge_base)",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("data/chroma"),
        help="ChromaDB persist directory (default: data/chroma)",
    )
    parser.add_argument(
        "--collection",
        default="health_kb",
        help="ChromaDB collection name (default: health_kb)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk only; skip embedding and ChromaDB upsert",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("ingest")
    args = _parse_args()

    # ── Validate paths ──────────────────────────────────────────────────────
    if not args.kb_dir.exists():
        logger.error("Knowledge-base directory not found: %s", args.kb_dir)
        return 1

    # ── Dry run: load and chunk only ────────────────────────────────────────
    if args.dry_run:
        from src.rag.ingestion import chunk_document, load_documents

        logger.info("DRY RUN — loading and chunking only (no embedding/upsert)")
        t0 = time.monotonic()
        try:
            docs = load_documents(args.kb_dir)
        except Exception as exc:
            logger.error("Failed to load documents: %s", exc)
            return 1

        total_chunks = 0
        for doc in docs:
            chunks = chunk_document(doc)
            total_chunks += len(chunks)
            print(f"  {doc.doc_id}: {len(chunks)} chunk(s)")

        elapsed = time.monotonic() - t0
        print(
            f"\nDry run complete: {len(docs)} document(s) → "
            f"{total_chunks} chunk(s) in {elapsed:.1f}s"
        )
        return 0

    # ── Full run: embed and upsert ──────────────────────────────────────────
    logger.info("Loading sentence-transformers model '%s'…", args.model)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError:
        logger.error(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )
        return 1

    try:
        embedding_model = SentenceTransformer(args.model)
    except Exception as exc:
        logger.error("Failed to load embedding model '%s': %s", args.model, exc)
        return 1

    logger.info("Connecting to ChromaDB at %s…", args.chroma_dir)
    try:
        import chromadb  # type: ignore[import]
    except ImportError:
        logger.error(
            "chromadb is not installed. Run: pip install chromadb"
        )
        return 1

    try:
        args.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(args.chroma_dir))
        collection = client.get_or_create_collection(
            name=args.collection,
            metadata={"hnsw:space": "l2"},
        )
    except Exception as exc:
        logger.error("Failed to connect to ChromaDB: %s", exc)
        return 1

    logger.info("Building knowledge base from %s…", args.kb_dir)
    try:
        from src.rag.ingestion import build_knowledge_base

        summary = build_knowledge_base(args.kb_dir, collection, embedding_model)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        return 1

    print(
        f"\nKnowledge base built successfully:\n"
        f"  Documents : {summary['documents']}\n"
        f"  Chunks    : {summary['chunks']}\n"
        f"  Time      : {summary['elapsed_ms']} ms\n"
        f"  Collection: {args.collection}\n"
        f"  ChromaDB  : {args.chroma_dir}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
