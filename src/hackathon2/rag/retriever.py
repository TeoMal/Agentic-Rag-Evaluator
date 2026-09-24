"""Pipeline B -- retrieval: find evidence chunks for a query. Evidence only, never a verdict.

    retriever = open_retriever()                         # once, at start-up (mode="vector")
    hits = retriever.search("incident notification deadline",
                            doc_types=["policy"], domains=["security"], k=5)
    hybrid = open_retriever(mode="hybrid")               # vector + BM25, fused with RRF
    retriever.get_chunk("information-security-policy#s3#c1")               # one chunk by id
    retriever.get_section("information-security-policy", "3. Encryption")  # a section's chunks, in order

Returns schemas.SearchHit objects (the knowledge tools' contract), each with the chunk
text, chunk_id, doc_id, source, page, section (None when not reliably known),
doc_type, domain, suspicious flag and score.

Three retrieval modes, kept so they can be evaluated against each other:
- "vector" (default, the baseline): semantic similarity only; score = cosine similarity.
- "hybrid": the vector ranking and a BM25 keyword ranking (lexical.py) over the same
  filtered chunks, each FETCH_K deep, merged with Reciprocal Rank Fusion:
      score(chunk) = sum over the two rankings of 1 / (RRF_K + rank)
  (RRF_K = 60, from Cormack, Clarke & Buettcher, SIGIR 2009). Chunks found by both
  rankings rise; ties are broken by chunk_id, so the order is deterministic.
  score = that fused RRF score -- a different scale from cosine similarity, so a
  min_score chosen for one mode does not carry over to the other.
- "lexical": the BM25 keyword ranking alone; score = BM25 score. open_retriever() falls back
  to it when no embedding model is configured or the vector index is unusable, so the
  knowledge pack stays searchable.
Metadata filters apply to every ranking in the same way.

What this module deliberately does NOT do:
- decide anything (no PASS/FAIL, no APPROVE/REJECT) -- that is the agents' and the
  decision gate's job;
- call an LLM or fill gaps from general knowledge -- results come only from the index;
- treat "nothing found" as compliance: an empty list means no evidence was retrieved,
  which callers must record as MISSING (schemas.EvidenceStatus), never as PASS.

Two failure cases are kept apart, so a broken backend never looks like missing evidence:
- the search ran and matched nothing          -> []
- the search could not run (e.g. DB down)     -> RetrievalUnavailableError
"""

import logging
from collections.abc import Sequence
from itertools import zip_longest
from typing import Literal

from hackathon2.config import Settings, get_settings
from hackathon2.rag.citations import RetrievalLog
from hackathon2.rag.data_models import ChunkFilter, DocumentChunk, KnowledgeDomain
from hackathon2.rag.embeddings import embeddings_configured
from hackathon2.rag.errors import DocumentLoadError, RetrievalUnavailableError
from hackathon2.rag.ingest import ingest
from hackathon2.rag.lexical import BM25Index
from hackathon2.rag.vector_store import DEFAULT_K, ChunkStore, KeywordChunkStore, get_vector_store
from hackathon2.schemas import DocType, SearchHit

logger = logging.getLogger(__name__)

RetrievalMode = Literal["vector", "hybrid", "lexical"]

# Candidates considered before re-ranking (source diversity, hybrid fusion): the default
# fetch_k LangChain uses for the same purpose in max_marginal_relevance_search.
FETCH_K = 20

# Reciprocal Rank Fusion constant, from the paper that introduced RRF (Cormack et al., 2009).
RRF_K = 60


class Retriever:
    """Search over the indexed knowledge pack, with metadata filters.

    Every hit returned is recorded in `log` when one is given (see citations.RetrievalLog).
    """

    def __init__(self, store: ChunkStore, log: RetrievalLog | None = None, mode: RetrievalMode = "vector") -> None:
        self._store = store
        self.log = log
        self.mode = mode
        # BM25 index and chunk list, built on first use and shared with with_log() copies
        self._lexical: dict[str, BM25Index | list[DocumentChunk]] = {}

    def with_log(self, log: RetrievalLog) -> "Retriever":
        """A retriever over the same index and mode that records into `log` -- one per assessment run."""
        copy = Retriever(self._store, log=log, mode=self.mode)
        copy._lexical = self._lexical
        return copy

    def search(
        self,
        query: str,
        *,
        doc_types: Sequence[DocType] | None = None,
        domains: Sequence[KnowledgeDomain] | None = None,
        doc_ids: Sequence[str] | None = None,
        vendors: Sequence[str] | None = None,
        k: int = DEFAULT_K,
        diversify_sources: bool = False,
    ) -> list[SearchHit]:
        """The k best chunks for `query` among those matching every given filter.

        Each filter keeps chunks whose field is one of the given values; None does not
        filter. An empty result means no matching evidence was found.

        Results are best first. With diversify_sources, the best FETCH_K candidates are
        taken in turns per source document -- each document's best chunk, then each
        one's second best, ... -- so strong evidence from other documents is not crowded
        out by several similar chunks of one document. Nothing is merged or dropped for
        disagreeing: conflicting statements from different documents are all returned,
        each with its own provenance and score.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        where = ChunkFilter(
            doc_types=_listed(doc_types), domains=_listed(domains), doc_ids=_listed(doc_ids), vendors=_listed(vendors)
        )
        fetch = max(k, FETCH_K) if (diversify_sources or self.mode != "vector") else k
        try:
            if self.mode == "lexical":
                hits: list[SearchHit] = list(self._lexical_index().search(query, fetch, where))
            else:
                hits = list(self._store.search(query, k=fetch, where=where))
            if self.mode == "hybrid":
                hits = _reciprocal_rank_fusion([hits, self._lexical_index().search(query, fetch, where)])
        except Exception as exc:
            raise RetrievalUnavailableError(f"Search could not be run: {exc}") from exc
        hits = _interleave_by_source(hits)[:k] if diversify_sources else hits[:k]
        if self.log is not None:
            self.log.record(hits)
        return hits

    def get_chunk(self, chunk_id: str) -> SearchHit | None:
        """One indexed chunk by id, or None if there is no such chunk."""
        try:
            chunk = self._store.get(chunk_id)
        except Exception as exc:
            raise RetrievalUnavailableError(f"Chunk lookup could not be run: {exc}") from exc
        if chunk is not None and self.log is not None:
            self.log.record([chunk])
        return chunk

    def get_section(self, doc_id: str, section: str) -> list[SearchHit]:
        """Every chunk of one section of one document, in document order ([] if there is none)."""
        chunks = [c for c in self._chunks() if c.doc_id == doc_id and c.section == section]
        chunks.sort(key=lambda c: int(c.chunk_id.rsplit("#c", 1)[1]))
        if self.log is not None:
            self.log.record(chunks)
        return chunks

    def vendor_documents(self) -> dict[str, list[str]]:
        """Each vendor named in the file names ('x' for vendor-x-*.pdf) -> the ids of its own
        documents (doc_type vendor_claim) in the index."""
        documents: dict[str, set[str]] = {}
        for chunk in self._chunks():
            if chunk.doc_type == "vendor_claim" and chunk.vendor:
                documents.setdefault(chunk.vendor, set()).add(chunk.doc_id)
        return {vendor: sorted(ids) for vendor, ids in sorted(documents.items())}

    def _chunks(self) -> list[DocumentChunk]:
        if "chunks" not in self._lexical:
            try:
                self._lexical["chunks"] = self._store.all_chunks()
            except Exception as exc:
                raise RetrievalUnavailableError(f"Index could not be read: {exc}") from exc
        return self._lexical["chunks"]

    def _lexical_index(self) -> BM25Index:
        if "bm25" not in self._lexical:
            self._lexical["bm25"] = BM25Index(self._chunks())
        return self._lexical["bm25"]


def open_retriever(
    settings: Settings | None = None, store: ChunkStore | None = None, mode: RetrievalMode = "vector"
) -> Retriever:
    """A Retriever over `store` (default: get_vector_store()), ingesting first if the index is empty.

    With no store given, the knowledge pack is loaded into a KeywordChunkStore and searched in
    "lexical" mode (BM25 only) when no embedding model is configured (logged as a warning) or
    when the vector index cannot be built or reached (logged as an error) -- keyword evidence
    instead of none. A knowledge pack that cannot be loaded still raises.

    An empty index would make every search return nothing -- indistinguishable from
    missing evidence -- so it is built here; with an explicit `store`, a failure to build
    it raises (IngestionError / DocumentLoadError / VectorStoreUnavailableError) instead of
    returning a retriever that finds nothing. Open a new retriever after re-ingesting:
    the hybrid keyword index is built from the chunks present at first use.
    """
    settings = settings or get_settings()
    if store is not None:
        return _open(settings, store, mode)
    if not embeddings_configured(settings):
        logger.warning(
            "No embedding model configured (AZURE_OPENAI_EMBEDDING_DEPLOYMENT); keyword-only (BM25) retrieval over %s",
            settings.knowledge_dir,
        )
        return _open(settings, KeywordChunkStore(), "lexical")
    try:
        return _open(settings, get_vector_store(settings), mode)
    except DocumentLoadError:
        raise  # the knowledge pack itself is wrong: keyword search would fail the same way
    except Exception as exc:  # noqa: BLE001 -- embedding deployment or vector database unusable
        logger.error(
            "Vector index unavailable (%s: %s); falling back to keyword-only (BM25) retrieval",
            type(exc).__name__,
            exc,
        )
        return _open(settings, KeywordChunkStore(), "lexical")


def _open(settings: Settings, store: ChunkStore, mode: RetrievalMode) -> Retriever:
    """Ingest when the index is empty or no longer matches the knowledge directory -- e.g. a new
    vendor's PDFs were added (the hidden vendor case): a persistent index would never see them."""
    if store.size() == 0:
        logger.info("Index is empty; ingesting %s", settings.knowledge_dir)
        ingest(settings, store)
    else:
        indexed = {chunk.source for chunk in store.all_chunks()}
        on_disk = {path.name for path in settings.knowledge_dir.rglob("*.pdf")}
        if indexed != on_disk:
            logger.info(
                "Knowledge pack changed (new: %s, removed: %s); re-ingesting %s",
                sorted(on_disk - indexed),
                sorted(indexed - on_disk),
                settings.knowledge_dir,
            )
            ingest(settings, store)
    return Retriever(store, mode=mode)


def _reciprocal_rank_fusion(rankings: Sequence[Sequence[SearchHit]]) -> list[SearchHit]:
    """Merge best-first rankings: score = sum of 1 / (RRF_K + rank); ties broken by chunk_id."""
    scores: dict[str, float] = {}
    chunks: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            chunks.setdefault(hit.chunk_id, hit)
    order = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [chunks[chunk_id].model_copy(update={"score": scores[chunk_id]}) for chunk_id in order]


def _interleave_by_source(hits: list[SearchHit]) -> list[SearchHit]:
    """Round-robin over source documents (ordered by their best hit), keeping each document's own ranking."""
    by_source: dict[str, list[SearchHit]] = {}
    for hit in hits:  # hits are best first, so dict order is by each source's best hit
        by_source.setdefault(hit.source, []).append(hit)
    rounds = zip_longest(*by_source.values())
    return [hit for round_ in rounds for hit in round_ if hit is not None]


def _listed(values: Sequence[str] | None) -> list | None:
    if values is None:
        return None
    return [values] if isinstance(values, str) else list(values)
