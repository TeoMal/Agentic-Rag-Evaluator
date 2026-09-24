"""Pipeline B -- retrieval: find evidence chunks for a query. Evidence only, never a verdict.

    retriever = open_retriever()                         # once, at start-up (mode="vector")
    hits = retriever.search("incident notification deadline",
                            doc_types=["policy"], domains=["security"], k=5)
    hybrid = open_retriever(mode="hybrid")               # vector + BM25, fused with RRF

Returns schemas.SearchHit objects (the knowledge tools' contract), each with the chunk
text, chunk_id, doc_id, source, page, section (None when not reliably known),
doc_type, domain, suspicious flag and score.

Two retrieval modes, both kept so they can be evaluated against each other:
- "vector" (default, the baseline): semantic similarity only; score = cosine similarity.
- "hybrid": the vector ranking and a BM25 keyword ranking (lexical.py) over the same
  filtered chunks, each FETCH_K deep, merged with Reciprocal Rank Fusion:
      score(chunk) = sum over the two rankings of 1 / (RRF_K + rank)
  (RRF_K = 60, from Cormack, Clarke & Buettcher, SIGIR 2009). Chunks found by both
  rankings rise; ties are broken by chunk_id, so the order is deterministic.
  score = that fused RRF score -- a different scale from cosine similarity, so a
  min_score chosen for one mode does not carry over to the other.
Metadata filters apply to both rankings in the same way.

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
from hackathon2.rag.data_models import ChunkFilter, KnowledgeDomain
from hackathon2.rag.errors import RetrievalUnavailableError
from hackathon2.rag.ingest import ingest
from hackathon2.rag.lexical import BM25Index
from hackathon2.rag.vector_store import DEFAULT_K, ChunkStore, get_vector_store
from hackathon2.schemas import DocType, SearchHit

logger = logging.getLogger(__name__)

RetrievalMode = Literal["vector", "hybrid"]

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
        self._lexical: dict[str, BM25Index] = {}  # built on first hybrid search, shared with with_log() copies

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
        fetch = max(k, FETCH_K) if (diversify_sources or self.mode == "hybrid") else k
        try:
            hits: list[SearchHit] = list(self._store.search(query, k=fetch, where=where))
            if self.mode == "hybrid":
                hits = _reciprocal_rank_fusion([hits, self._lexical_index().search(query, fetch, where)])
        except Exception as exc:
            raise RetrievalUnavailableError(f"Search could not be run: {exc}") from exc
        hits = _interleave_by_source(hits)[:k] if diversify_sources else hits[:k]
        if self.log is not None:
            self.log.record(hits)
        return hits

    def _lexical_index(self) -> BM25Index:
        if "bm25" not in self._lexical:
            self._lexical["bm25"] = BM25Index(self._store.all_chunks())
        return self._lexical["bm25"]


def open_retriever(
    settings: Settings | None = None, store: ChunkStore | None = None, mode: RetrievalMode = "vector"
) -> Retriever:
    """A Retriever over `store` (default: get_vector_store()), ingesting first if the index is empty.

    An empty index would make every search return nothing -- indistinguishable from
    missing evidence -- so it is built here, and a failure to build it raises
    (IngestionError / DocumentLoadError / VectorStoreUnavailableError) instead of
    returning a retriever that finds nothing. Open a new retriever after re-ingesting:
    the hybrid keyword index is built from the chunks present at first use.
    """
    settings = settings or get_settings()
    store = store or get_vector_store(settings)
    if store.size() == 0:
        logger.info("Vector index is empty; ingesting %s", settings.knowledge_dir)
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
