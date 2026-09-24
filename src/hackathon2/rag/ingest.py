"""Pipeline A -- ingestion / indexing: the one entry point that builds the knowledge index.

    discover + load + classify   loaders.load_knowledge   (document_registry.classify per file)
    chunk                        chunking.chunk_pages
    embed + store                vector_store.ChunkStore.replace   (embedder from embeddings.get_embedder)

Run it from code (`summary = ingest(store=store)`) or from the command line:

    uv run python -m hackathon2.rag.ingest

Idempotent: the index ends up holding exactly the chunks of the current knowledge
directory -- re-running on the same PDFs changes nothing, chunks of removed PDFs are
deleted.

Failure handling:
- A wrong knowledge directory (non-PDF file, unclassifiable PDF, duplicate names)
  raises DocumentLoadError before anything is indexed.
- An unreachable vector database raises VectorStoreUnavailableError before anything is indexed.
- A PDF or page with no extractable text is skipped and listed in the summary; the
  rest is indexed.
- No chunks at all, or an embedding / vector-store write failure, raises
  IngestionError and leaves the previous index unchanged -- never a silently partial
  or empty index. If only the removal of stale chunks fails, IngestionError says so
  (new chunks indexed, stale ones may remain).

With the in-memory backend the index lives in the process: pass the same `store` the
retriever will search, otherwise the index is discarded when ingest() returns.
"""

import logging
import time

from pydantic import BaseModel, Field

from hackathon2.config import Settings, get_settings
from hackathon2.rag.chunking import chunk_pages
from hackathon2.rag.data_models import ExtractionIssue
from hackathon2.rag.errors import IngestionError
from hackathon2.rag.loaders import load_knowledge
from hackathon2.rag.vector_store import ChunkStore, get_vector_store

logger = logging.getLogger(__name__)


class IngestionSummary(BaseModel):
    """What one ingestion run did."""

    documents_discovered: int
    documents_indexed: int
    documents_skipped: list[str] = Field(description="PDFs with no page that produced text.")
    pages_loaded: int
    pages_skipped: int = Field(description="Pages with no extractable text.")
    chunks_indexed: int
    stale_chunks_removed: int
    issues: list[ExtractionIssue] = Field(description="Every extraction problem, per file or page.")
    duration_seconds: float


def ingest(settings: Settings | None = None, store: ChunkStore | None = None) -> IngestionSummary:
    """Load, chunk, embed and index the knowledge directory into `store` (default: get_vector_store())."""
    started = time.perf_counter()
    settings = settings or get_settings()

    loaded = load_knowledge(settings)
    chunks = chunk_pages(loaded.pages)
    if not chunks:
        raise IngestionError(
            f"No chunks produced from {settings.knowledge_dir} ({len(loaded.documents)} PDFs discovered, "
            f"{len(loaded.issues)} extraction issues); the existing index was not touched."
        )

    store = store or get_vector_store(settings)
    removed = store.replace(chunks)  # raises IngestionError stating the index state on failure

    indexed_sources = {c.source for c in chunks}
    summary = IngestionSummary(
        documents_discovered=len(loaded.documents),
        documents_indexed=len(indexed_sources),
        documents_skipped=[source for source in loaded.documents if source not in indexed_sources],
        pages_loaded=len(loaded.pages),
        pages_skipped=sum(1 for issue in loaded.issues if issue.page is not None),
        chunks_indexed=len(chunks),
        stale_chunks_removed=removed,
        issues=loaded.issues,
        duration_seconds=round(time.perf_counter() - started, 3),
    )
    logger.info(
        "Ingested %d/%d documents, %d pages, %d chunks (%d stale removed, %d issues) in %.1fs",
        summary.documents_indexed,
        summary.documents_discovered,
        summary.pages_loaded,
        summary.chunks_indexed,
        summary.stale_chunks_removed,
        len(summary.issues),
        summary.duration_seconds,
    )
    return summary


def main() -> None:
    logging.basicConfig(level=get_settings().log_level)
    print(ingest().model_dump_json(indent=2))


if __name__ == "__main__":
    main()
