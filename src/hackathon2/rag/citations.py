"""Citations: structured provenance for retrieved evidence (FR04).

A citation is a schemas.Evidence -- chunk_id, source file name, doc_type, section,
page and the exact quote -- kept structured end to end and turned into text only for
display (format_citation). Every field comes from the retrieved SearchHit; a page or
section that was not available stays None and is left out of the formatted text,
never guessed. The document id is the part of chunk_id before the first '#'.

RetrievalLog records every chunk retrieved during one run, so a citation can be
checked against what was actually retrieved (schemas.Evidence: "chunk_id must be a
chunk actually retrieved during this run"):

    log = RetrievalLog()
    retriever = Retriever(store, log=log)          # every search is recorded
    evidence = cite(hit, quote="...")              # quote must be verbatim chunk text
    log.verify(evidence)                           # [] when the citation matches a retrieved chunk
    format_citation(evidence)                      # 'information-security-policy.pdf, p. 1, 3. Encryption [...]'
"""

import re
from collections.abc import Iterable

from hackathon2.schemas import Evidence, SearchHit

_UNTRUSTED_TAGS = re.compile(r"</?untrusted_document>")


class CitationError(ValueError):
    """A citation does not match the retrieved chunk it claims to cite."""


def cite(hit: SearchHit, quote: str | None = None) -> Evidence:
    """A citation of `hit`. `quote`, when given, must appear verbatim in the chunk text."""
    if quote is not None and quote not in _plain(hit.text):
        raise CitationError(f"Quote is not verbatim text of {hit.chunk_id}: {quote[:80]!r}")
    return hit.to_evidence(quote)


def format_citation(citation: Evidence | SearchHit) -> str:
    """Display text for a citation: 'source, p. N, section [chunk_id]', omitting unknown page/section."""
    parts = [citation.source]
    if citation.page is not None:
        parts.append(f"p. {citation.page}")
    if citation.section:
        parts.append(citation.section)
    return f"{', '.join(parts)} [{citation.chunk_id}]"


class RetrievalLog:
    """Every chunk retrieved during one run, in first-retrieved order."""

    def __init__(self) -> None:
        self._hits: dict[str, SearchHit] = {}

    def record(self, hits: Iterable[SearchHit]) -> None:
        for hit in hits:
            self._hits.setdefault(hit.chunk_id, hit)

    @property
    def chunk_ids(self) -> list[str]:
        """For schemas.RunMetrics.retrieved_chunk_ids."""
        return list(self._hits)

    def get(self, chunk_id: str) -> SearchHit | None:
        return self._hits.get(chunk_id)

    def verify(self, evidence: Evidence) -> list[str]:
        """Problems with a citation; [] when it matches a chunk retrieved in this run."""
        hit = self._hits.get(evidence.chunk_id)
        if hit is None:
            return [f"{evidence.chunk_id} was not retrieved in this run"]
        problems = [
            f"{field} is {getattr(evidence, field)!r}, retrieved chunk has {getattr(hit, field)!r}"
            for field in ("source", "doc_type", "page", "section")
            if getattr(evidence, field) != getattr(hit, field)
        ]
        if evidence.quote not in _plain(hit.text):
            problems.append("quote is not verbatim text of the retrieved chunk")
        return problems


def _plain(text: str) -> str:
    return _UNTRUSTED_TAGS.sub("", text).strip()
