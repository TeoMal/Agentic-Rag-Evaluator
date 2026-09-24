"""Vector store behind one small interface, shared by both pipelines.

    store = get_vector_store()
    store.add(chunks)                                   # pipeline A: index DocumentChunks
    store.search("encryption at rest", k=5,             # pipeline B: similarity search
                 where=ChunkFilter(doc_types=["policy"], domains=["security"]))

Callers only see ChunkStore, DocumentChunk and ChunkFilter; which database holds the
vectors is decided in get_vector_store():
- PGVector (Postgres + pgvector, docker-compose `db`) when Settings.sqlalchemy_database_url is set;
- an in-memory store otherwise (index lives in the process, rebuilt on every start).
With no embedding model configured at all, retriever.open_retriever uses KeywordChunkStore
(chunks in memory, no vectors) and searches it with BM25 only.

What is stored: the chunk text is embedded, and every other DocumentChunk field is
kept as metadata, so a search result is rebuilt into a complete DocumentChunk.
chunk_id is the store id, so re-indexing the same chunk overwrites it instead of
duplicating it.

Scores: `score` on returned chunks is the cosine similarity between query and chunk
(higher = more similar), on both backends -- PGVector reports cosine distance, which
is converted as 1 - distance.

Connecting to Postgres (nothing to change in code -- only settings):
- Set POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB.
  Outside docker-compose use POSTGRES_HOST=127.0.0.1, not localhost: compose publishes
  the port on 127.0.0.1 only, and on Windows `localhost` tries IPv6 (::1) first.
- libpq waits indefinitely for an unreachable server by default; set the standard
  libpq variable PGCONNECT_TIMEOUT (seconds) to make it fail instead.
- An unreachable or unusable database raises VectorStoreUnavailableError.

This module only stores and finds chunks. It does not decide what a result means.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from hackathon2.config import Settings, get_settings
from hackathon2.rag.data_models import ChunkFilter, DocumentChunk, metadata_matches
from hackathon2.rag.embeddings import Embeddings, get_embedder
from hackathon2.rag.errors import IngestionError, VectorStoreUnavailableError

# The default k of the knowledge tools in the schemas contract (search_policy, search_vendor_documents).
DEFAULT_K = 5

# Postgres table namespace (langchain-postgres "collection") holding the knowledge-pack chunks.
COLLECTION_NAME = "nfs_knowledge"


class ChunkStore(ABC):
    """Index and search DocumentChunks. Backend-specific details stay in the subclasses."""

    def __init__(self, backend: VectorStore) -> None:
        self._backend = backend

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        """Embed and index chunks; a chunk already stored under the same chunk_id is replaced."""
        if not chunks:
            return
        self._backend.add_documents([_to_document(c) for c in chunks], ids=[c.chunk_id for c in chunks])

    def replace(self, chunks: Sequence[DocumentChunk]) -> int:
        """Make the index hold exactly `chunks`; returns how many stale chunks were removed.

        Upserts every chunk first (both backends embed everything before writing, so a
        failure leaves the index unchanged), then deletes stored chunks not in `chunks`
        -- e.g. from a PDF that was removed. Running it twice on the same chunks is a no-op.

        Raises IngestionError saying which phase failed and what state the index is in.
        """
        try:
            self.add(chunks)
        except Exception as exc:
            raise IngestionError(f"Indexing {len(chunks)} chunks failed; the index was left unchanged: {exc}") from exc
        try:
            stale = sorted(self._stored_ids() - {c.chunk_id for c in chunks})
            if stale:
                self._backend.delete(ids=stale)
        except Exception as exc:
            raise IngestionError(
                f"All {len(chunks)} chunks were indexed, but removing chunks of deleted documents failed "
                f"(the index may still hold them): {exc}"
            ) from exc
        return len(stale)

    def size(self) -> int:
        """Number of chunks in the index (0 means nothing has been ingested)."""
        return len(self._stored_ids())

    def all_chunks(self) -> list[DocumentChunk]:
        """Every indexed chunk with its full metadata, in chunk_id order (no scores)."""
        return [_to_chunk(doc, None) for doc in self._backend.get_by_ids(sorted(self._stored_ids()))]

    def get(self, chunk_id: str) -> DocumentChunk | None:
        """One indexed chunk by id (no score), or None if it is not in the index."""
        docs = self._backend.get_by_ids([chunk_id])
        return _to_chunk(docs[0], None) if docs else None

    def search(self, query: str, k: int = DEFAULT_K, where: ChunkFilter | None = None) -> list[DocumentChunk]:
        """The k chunks most similar to `query` that match `where`, best first, each with its score."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        conditions = where.as_dict() if where else {}
        kwargs = {"filter": self._backend_filter(conditions)} if conditions else {}
        return [_to_chunk(doc, score) for doc, score in self._scored_search(query, k, **kwargs)]

    @abstractmethod
    def _backend_filter(self, conditions: dict[str, list[str]]) -> Any:
        """Translate {field: allowed values} into the backend's filter format."""

    @abstractmethod
    def _scored_search(self, query: str, k: int, **kwargs: Any) -> list[tuple[Document, float]]:
        """Search returning (document, cosine similarity) pairs."""

    @abstractmethod
    def _stored_ids(self) -> set[str]:
        """chunk_ids currently in the index."""


class InMemoryChunkStore(ChunkStore):
    def __init__(self, embedder: Embeddings) -> None:
        super().__init__(InMemoryVectorStore(embedder))

    def _backend_filter(self, conditions: dict[str, list[str]]) -> Callable[[Document], bool]:
        return lambda doc: metadata_matches(doc.metadata, conditions)

    def _scored_search(self, query: str, k: int, **kwargs: Any) -> list[tuple[Document, float]]:
        return self._backend.similarity_search_with_score(query, k=k, **kwargs)  # already cosine similarity

    def _stored_ids(self) -> set[str]:
        return set(self._backend.store)


class PGVectorChunkStore(ChunkStore):
    def __init__(self, embedder: Embeddings, connection: str) -> None:
        # Imported here so the in-memory path needs no Postgres driver.
        from langchain_postgres import PGVector
        from sqlalchemy.exc import SQLAlchemyError

        try:
            # Connects immediately: creates the pgvector extension and the collection if missing.
            backend = PGVector(
                embeddings=embedder, connection=connection, collection_name=COLLECTION_NAME, use_jsonb=True
            )
        except SQLAlchemyError as exc:
            reason = str(getattr(exc, "orig", None) or exc).splitlines()[0]
            raise VectorStoreUnavailableError(f"Cannot use the Postgres vector store: {reason}") from exc
        super().__init__(backend)

    def _backend_filter(self, conditions: dict[str, list[str]]) -> dict[str, Any]:
        clauses = [{field: {"$in": values}} for field, values in conditions.items()]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _scored_search(self, query: str, k: int, **kwargs: Any) -> list[tuple[Document, float]]:
        # langchain-postgres returns cosine distance (the default distance strategy).
        return [
            (doc, 1.0 - distance) for doc, distance in self._backend.similarity_search_with_score(query, k=k, **kwargs)
        ]

    def _stored_ids(self) -> set[str]:
        # langchain-postgres has no public "list ids"; read the collection's rows with its own session and model.
        from sqlalchemy import select

        store = self._backend
        with store._make_sync_session() as session:
            collection = store.get_collection(session)
            if collection is None:
                return set()
            rows = session.execute(
                select(store.EmbeddingStore.id).where(store.EmbeddingStore.collection_id == collection.uuid)
            )
            return {row[0] for row in rows}


class KeywordChunkStore(ChunkStore):
    """Chunks held in memory WITHOUT embeddings -- for keyword-only (BM25) retrieval when no
    embedding model is configured. It cannot do similarity search; Retriever(mode="lexical")
    searches it through lexical.BM25Index instead."""

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}

    def add(self, chunks: Sequence[DocumentChunk]) -> None:
        self._chunks.update((c.chunk_id, c) for c in chunks)

    def replace(self, chunks: Sequence[DocumentChunk]) -> int:
        stale = set(self._chunks) - {c.chunk_id for c in chunks}
        self._chunks = {c.chunk_id: c for c in chunks}
        return len(stale)

    def all_chunks(self) -> list[DocumentChunk]:
        return [self._chunks[chunk_id] for chunk_id in sorted(self._chunks)]

    def get(self, chunk_id: str) -> DocumentChunk | None:
        return self._chunks.get(chunk_id)

    def _backend_filter(self, conditions: dict[str, list[str]]) -> Any:
        raise NotImplementedError

    def _scored_search(self, query: str, k: int, **kwargs: Any) -> list[tuple[Document, float]]:
        raise NotImplementedError("a keyword-only store has no vector search; use Retriever(mode='lexical')")

    def _stored_ids(self) -> set[str]:
        return set(self._chunks)


def get_vector_store(settings: Settings | None = None, embedder: Embeddings | None = None) -> ChunkStore:
    """PGVector when a database is configured, otherwise in memory. Both embed with get_embedder()."""
    settings = settings or get_settings()
    embedder = embedder or get_embedder(settings)
    if settings.sqlalchemy_database_url:
        return PGVectorChunkStore(embedder, settings.sqlalchemy_database_url)
    return InMemoryChunkStore(embedder)


def _to_document(chunk: DocumentChunk) -> Document:
    return Document(id=chunk.chunk_id, page_content=chunk.text, metadata=chunk.model_dump(exclude={"text", "score"}))


def _to_chunk(doc: Document, score: float | None) -> DocumentChunk:
    return DocumentChunk(**doc.metadata, text=doc.page_content, score=score)
