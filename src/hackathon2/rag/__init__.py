"""Retrieval over the NFS knowledge pack (Deep Agent / RAG engineer).

FR03 index + retrieve, FR04 cite sources, FR05 evidence vs inference vs missing,
FR09 prompt injection in retrieved documents, FR10 contradictions/UNKNOWN, FR14 fail
without crashing (numbering of the final handout). Corpus lives in knowledge/
(Settings.knowledge_dir).
Start from course units:
  section-12-rag/60-doc-loaders-splitters  load + chunk, keep `source` metadata for citations
  section-12-rag/62-postgres-pgvector      PGVector store (Settings.sqlalchemy_database_url)
  section-12-rag/63-agentic-rag            retriever exposed as a tool
  section-12-rag/64-advanced-retrieval     MMR + LLM rerank
Embeddings: hackathon2.llm.get_embeddings() (Azure OpenAI, not Cohere as in class).

Two independent pipelines:
  A. ingestion / indexing (ingest.py)  loaders -> chunking -> security -> embeddings -> vector_store
  B. retrieval / query (retriever.py)  vector_store -> filters -> SearchHit -> citations

This package is self-contained: it imports hackathon2.config, hackathon2.llm and
hackathon2.schemas only, never agents/, mcp_server/ or guardrails/. Callers use the
functions this module exports and receive schemas types, never backend objects:

    from hackathon2.rag import open_retriever, RetrievalUnavailableError
    retriever = open_retriever()                     # at start-up; ingests if the index is empty
    hits = retriever.search("...", doc_types=["policy"], domains=["security"], k=5)  # list[SearchHit]
"""

from hackathon2.rag.citations import CitationError, RetrievalLog, cite, format_citation
from hackathon2.rag.data_models import ChannelResult, EvidenceBundle, RetrievalStatus
from hackathon2.rag.errors import RetrievalUnavailableError
from hackathon2.rag.evidence import EvidenceService
from hackathon2.rag.ingest import IngestionSummary, ingest
from hackathon2.rag.retriever import RetrievalMode, Retriever, open_retriever

__all__ = [
    "ChannelResult",
    "CitationError",
    "EvidenceBundle",
    "EvidenceService",
    "IngestionSummary",
    "RetrievalLog",
    "RetrievalMode",
    "RetrievalStatus",
    "RetrievalUnavailableError",
    "Retriever",
    "cite",
    "format_citation",
    "ingest",
    "open_retriever",
]
