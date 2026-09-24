"""The team's RAG, as (query, k) callables for the retrieval suite:

    uv run python -m evaluation.run retrieval --retriever evaluation.retrievers:rag
    uv run python -m evaluation.run retrieval --retriever evaluation.retrievers:keyword
    uv run python -m evaluation.run retrieval --retriever evaluation.retrievers:hybrid

rag      exactly what the MCP server searches: vector search, or keyword (BM25) when no embedding
         model is configured or reachable -- the mode is printed so the report is not misread
keyword  BM25 only: no embeddings, no network -- the baseline the others must beat
hybrid   vector + BM25 fused with RRF; refuses to run (instead of silently scoring BM25) without a
         working embedding deployment
"""

import sys
from functools import cache

from hackathon2.rag import Retriever, open_retriever
from hackathon2.rag.vector_store import KeywordChunkStore
from hackathon2.schemas import SearchHit


@cache
def _open(name: str) -> Retriever:
    if name == "keyword":
        retriever = open_retriever(store=KeywordChunkStore(), mode="lexical")
    elif name == "hybrid":
        retriever = open_retriever(mode="hybrid")
        if retriever.mode != "hybrid":
            raise RuntimeError(
                "hybrid retrieval needs a working embedding deployment (AZURE_OPENAI_EMBEDDING_DEPLOYMENT)"
            )
    else:
        retriever = open_retriever()
    print(f"retriever '{name}': mode={retriever.mode}", file=sys.stderr)
    return retriever


def rag(query: str, k: int) -> list[SearchHit]:
    return _open("rag").search(query, k=k)


def keyword(query: str, k: int) -> list[SearchHit]:
    return _open("keyword").search(query, k=k)


def hybrid(query: str, k: int) -> list[SearchHit]:
    return _open("hybrid").search(query, k=k)
