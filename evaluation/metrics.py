"""Pure scoring, no LLM: retrieval rank metrics and citation checks.

Retrieval gold labels name documents (optionally a section), not chunk ids, so they
survive re-chunking. Citation checks prove what code can prove: the cited chunk was
retrieved in this run, the quote really appears in it, and the source matches.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import PurePath

from pydantic import BaseModel, Field

from hackathon2.schemas import Evidence, SearchHit

# The supplied NFS knowledge pack (handout section 6), by doc_id (file name without .pdf).
KNOWLEDGE_PACK = frozenset({
    "procurement-policy", "information-security-policy", "ai-governance-policy", "vendor-risk-policy",
    "data-classification-policy", "vendor-x-proposal", "vendor-x-security-questionnaire", "vendor-x-pricing",
    "vendor-alpha-assessment", "vendor-beta-assessment", "vendor-gamma-assessment",  # historical-vendor-assessments/
})


class Target(BaseModel):
    """A document (optionally one section) that SHOULD be retrieved for a query."""

    doc_id: str
    section: str | None = Field(default=None, description="Case-insensitive prefix of SearchHit.section.")
    grade: int = Field(default=1, ge=1, le=3, description="3 = holds the answer, 1 = useful context.")

    def matches(self, hit: SearchHit) -> bool:
        return hit.doc_id == self.doc_id and (
            self.section is None or (hit.section or "").lower().startswith(self.section.lower())
        )


class RetrievalCase(BaseModel):
    id: str
    domain: str
    query: str
    relevant: list[Target] = Field(min_length=1)


def _dcg(gains: Sequence[int]) -> float:
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))


def rank_scores(hits: Sequence[SearchHit], targets: Sequence[Target], k: int) -> dict[str, float]:
    """hit, recall, MRR and nDCG @k. Each target is credited once, at its first matching
    hit, so many chunks of one document cannot push recall or nDCG above 1."""
    credited: set[int] = set()
    gains: list[int] = []
    first = 0
    for rank, hit in enumerate(hits[:k], start=1):
        matched = {i for i, target in enumerate(targets) if target.matches(hit)}
        first = first or (rank if matched else 0)
        gains.append(max((targets[i].grade for i in matched - credited), default=0))
        credited |= matched
    ideal = _dcg(sorted((t.grade for t in targets), reverse=True)[:k])
    return {
        "hit": float(bool(first)),
        "recall": len(credited) / len(targets),
        "mrr": 1 / first if first else 0.0,
        "ndcg": _dcg(gains) / ideal if ideal else 0.0,
    }


_TAGS = re.compile(r"</?untrusted_document>", re.IGNORECASE)
_PUNCTUATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def normalize(text: str) -> str:
    """Compare text as a reader would: no wrapper tags, case, curly quotes or extra spacing."""
    return " ".join(_TAGS.sub(" ", text).translate(_PUNCTUATION).casefold().split())


def quote_in_text(quote: str, text: str) -> bool:
    """The quote appears in the text. An ellipsis ("A ... B") may skip words; order must hold."""
    haystack, position = normalize(text), 0
    pieces = [p for p in (normalize(x).strip(" .") for x in re.split(r"\.\.\.|…", quote)) if p]
    for piece in pieces:
        position = haystack.find(piece, position)
        if position < 0:
            return False
        position += len(piece)
    return bool(pieces)


def citation_problems(evidence: Evidence, retrieved_ids: set[str], hits: dict[str, SearchHit]) -> list[str]:
    """[] when every check that can run passes. Without the chunk's text in the run's log
    the quote cannot be verified -- that is not counted as a problem."""
    problems = []
    if retrieved_ids and evidence.chunk_id not in retrieved_ids:
        problems.append("fabricated")  # cites a chunk this run never retrieved
    hit = hits.get(evidence.chunk_id)
    if hit is not None:
        if not quote_in_text(evidence.quote, hit.text):
            problems.append("misquoted")
        if (hit.source, hit.doc_type) != (evidence.source, evidence.doc_type):
            problems.append("source_mismatch")
    elif "#" in evidence.chunk_id and evidence.chunk_id.split("#")[0] != PurePath(evidence.source).stem:
        problems.append("source_mismatch")  # the chunk id itself names another document
    return problems
