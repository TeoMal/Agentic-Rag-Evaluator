"""Lexical (keyword) retrieval: BM25 over the indexed chunks, for hybrid search.

Embeddings capture meaning but are weak on exact tokens -- '24 hours' vs '72 hours',
'TLS 1.2', 'SOC 2', 'IS-010', 'EUR 100,000' -- which decide many findings in this
corpus. BM25 ranks chunks by those exact terms.

Standard Okapi BM25, no tuning:
- k1 = 1.5, b = 0.75: the defaults of the rank_bm25 library;
- idf = ln(1 + (N - n + 0.5) / (n + 0.5)): Lucene's variant, always positive.

Tokens are lower-cased runs of letters/digits, keeping '.', '-' and ',' between
digits/letters so versions, ids and amounts stay whole ('1.2', 'aes-256', '100,000').
No stop-word list and no stemming. The index is built in memory from the chunks the
vector store holds, so it sees exactly the same corpus and metadata.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

from hackathon2.rag.data_models import ChunkFilter, DocumentChunk

K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._term_counts = [Counter(tokenize(c.text)) for c in self._chunks]
        self._lengths = [sum(tc.values()) for tc in self._term_counts]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        doc_freq = Counter(term for tc in self._term_counts for term in tc)
        n = len(self._chunks)
        self._idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int, where: ChunkFilter | None = None) -> list[DocumentChunk]:
        """The k best-scoring chunks matching `where` that share at least one term with the query.

        Each returned chunk carries its BM25 score; ties are broken by chunk_id.
        """
        terms = [t for t in tokenize(query) if t in self._idf]
        scored = []
        for chunk, counts, length in zip(self._chunks, self._term_counts, self._lengths, strict=True):
            if where is not None and not where.matches(
                chunk.model_dump(include={"doc_type", "domain", "doc_id", "vendor"})
            ):
                continue
            score = sum(self._term_score(term, counts[term], length) for term in terms if counts[term])
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda sc: (-sc[0], sc[1].chunk_id))
        return [chunk.model_copy(update={"score": score}) for score, chunk in scored[:k]]

    def _term_score(self, term: str, freq: int, length: int) -> float:
        norm = K1 * (1 - B + B * length / self._avg_length)
        return self._idf[term] * freq * (K1 + 1) / (freq + norm)
