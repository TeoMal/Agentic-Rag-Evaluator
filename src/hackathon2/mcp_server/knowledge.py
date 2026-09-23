"""Backend for the knowledge tools -- the adapter between the MCP server and RAG.

The RAG package (hackathon2.rag, owned by the RAG engineer) reads the PDFs in
knowledge/, chunks them, embeds them and stores them. This module does NOT read
PDFs. It asks the retriever for chunks and turns them into safe SearchHits:

    retriever result (LangChain Document + metadata)
        -> SearchHit (schemas.py)
        -> injection check (second layer, independent of ingestion)
        -> text wrapped in <untrusted_document> tags

What this module expects from hackathon2.rag.retriever (agree on this with the RAG engineer):

    search(query: str, *, k: int, filter: dict | None = None) -> list[tuple[Document, float]]
        Top-k chunks with relevance scores (higher = better), like PGVector's
        similarity_search_with_relevance_scores. `filter` is an exact-match dict on
        metadata, e.g. {"doc_type": "policy", "domain": "security"}.
    get_chunk(chunk_id: str) -> Document | None
    get_section(doc_id: str, section: str) -> list[Document]      # chunks of one section, in order

Metadata every chunk must carry (set at ingestion):

    chunk_id   stable id, e.g. "information-security-policy#s4.2#c1"     (required)
    doc_id     file stem, e.g. "information-security-policy"              (required)
    source     file name, e.g. "information-security-policy.pdf"          (defaults from doc_id)
    doc_type   "policy" | "vendor_claim"                                  (defaults from doc_id)
    domain     "security" | "procurement" | "legal" | "ai_governance" | "data" | "general"
    section    heading, e.g. "4.2 Encryption"                             (optional)
    page       1-based page number                                        (optional)
    suspicious True if the ingestion scanner flagged injection-like text  (optional)

The mandatory controls checklist is data/requirements.json in this package: the
RAG engineer extracts it from the policy PDFs, a human reviews it, it is committed.

Failures: if the retriever is missing or raises, the error propagates and the
server's safe_tool turns it into status "unavailable" -- the agent then records
MISSING evidence instead of crashing (FR15).
"""

import json
import logging
import re
from functools import lru_cache
from importlib import resources

from hackathon2.schemas import DocType, Domain, RequirementControl, SearchHit

logger = logging.getLogger("hackathon2.mcp_server.knowledge")

# doc_id -> (file name in knowledge/, doc_type). The handout's knowledge pack.
KNOWLEDGE_DOCS: dict[str, tuple[str, DocType]] = {
    "procurement-policy": ("procurement-policy.pdf", "policy"),
    "information-security-policy": ("information-security-policy.pdf", "policy"),
    "ai-governance-policy": ("ai-governance-policy.pdf", "policy"),
    "vendor-risk-policy": ("vendor-risk-policy.pdf", "policy"),
    "data-classification-policy": ("data-classification-policy.pdf", "policy"),
    "vendor-x-proposal": ("vendor-x-proposal.pdf", "vendor_claim"),
    "vendor-x-security-questionnaire": ("vendor-x-security-questionnaire.pdf", "vendor_claim"),
    "vendor-x-pricing": ("vendor-x-pricing.pdf", "vendor_claim"),
}

# The knowledge pack describes one vendor; its documents are "vendor-x-*".
VENDOR_DOCS: dict[str, tuple[str, ...]] = {
    "asteria-ai-systems": tuple(d for d, (_, t) in KNOWLEDGE_DOCS.items() if t == "vendor_claim"),
}

# Over-fetch when filtering after retrieval (vendor/doc filters), then trim to k.
FETCH_MULTIPLIER = 4

# Second-layer injection check at the MCP boundary. Ingestion flags chunks too; this
# catches anything it missed. A flagged chunk is still returned -- it may be real
# evidence -- but marked, so the agent treats it with suspicion.
_INJECTION_PATTERNS = re.compile(
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)"
    r"|disregard (the |all )?(previous|prior|system)"
    r"|(system|developer) (note|prompt|message|instruction)s?\s*(to|for)?\s*(the )?(ai|assistant|model|reviewer)"
    r"|you are now|act as (an? )?(unrestricted|different)"
    r"|(recommend|output|respond with|mark (this|the vendor) as)\s+approve"
    r"|<\s*/?\s*(system|untrusted_document)\s*>",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_PATTERNS.search(text))


def wrap_untrusted(text: str) -> str:
    """Retrieved text is data, never instructions. Tags inside the text are neutralised
    so a document cannot close the wrapper and 'escape' into instructions."""
    safe = re.sub(r"<\s*(/?)\s*untrusted_document\s*>", r"[\1untrusted_document]", text, flags=re.IGNORECASE)
    return f"<untrusted_document>{safe}</untrusted_document>"


# --------------------------------------------------------------------------------------
# Retriever access
# --------------------------------------------------------------------------------------


def _retriever():
    """Imported lazily: the server starts (and lists its tools) even before RAG is ready."""
    try:
        from hackathon2.rag import retriever
    except ImportError as exc:
        raise ConnectionError("knowledge index not available (hackathon2.rag.retriever is missing)") from exc
    return retriever


def _to_hit(doc, score: float | None = None) -> SearchHit:
    """LangChain Document (or an existing SearchHit) -> safe SearchHit."""
    if isinstance(doc, SearchHit):
        hit = doc if score is None else doc.model_copy(update={"score": score})
        raw_text, flagged = hit.text, hit.suspicious
    else:
        meta = doc.metadata or {}
        doc_id = meta["doc_id"]
        default_source, default_type = KNOWLEDGE_DOCS.get(doc_id, (f"{doc_id}.pdf", "policy"))
        raw_text, flagged = doc.page_content, bool(meta.get("suspicious", False))
        hit = SearchHit(
            chunk_id=meta["chunk_id"],
            doc_id=doc_id,
            source=meta.get("source") or default_source,
            doc_type=meta.get("doc_type") or default_type,
            domain=meta.get("domain") or "general",
            section=meta.get("section"),
            page=meta.get("page"),
            suspicious=flagged,
            score=None if score is None else round(float(score), 4),
            text=raw_text,
        )
    raw_text = re.sub(r"</?untrusted_document>", "", raw_text)  # never double-wrap
    suspicious = flagged or looks_like_injection(raw_text)
    if suspicious and not flagged:
        logger.warning("injection-like text in %s (missed at ingestion)", hit.chunk_id)
    return hit.model_copy(update={"text": wrap_untrusted(raw_text), "suspicious": suspicious})


def _unpack(results) -> list[tuple[object, float | None]]:
    """Accept [(Document, score)] or [Document] from the retriever."""
    return [r if isinstance(r, tuple) else (r, None) for r in results]


# --------------------------------------------------------------------------------------
# The functions server.py calls -- signatures unchanged since the stub.
# --------------------------------------------------------------------------------------


def search(
    query: str,
    *,
    doc_type: DocType,
    domain: str | None = None,
    vendor_id: str | None = None,
    doc_id: str | None = None,
    k: int = 5,
) -> list[SearchHit]:
    """Top-k chunks of one doc_type, best first. Empty list = searched, nothing relevant."""
    allowed_docs: set[str] | None = None
    if doc_type == "vendor_claim":
        allowed_docs = set(VENDOR_DOCS.get(vendor_id or "", ()))
        if not allowed_docs:
            return []  # no documents on file for this vendor
    if doc_id is not None:
        if allowed_docs is not None and doc_id not in allowed_docs:
            return []
        allowed_docs = {doc_id}

    metadata_filter: dict = {"doc_type": doc_type}
    if domain is not None:
        metadata_filter["domain"] = domain
    if allowed_docs is not None and len(allowed_docs) == 1:
        metadata_filter["doc_id"] = next(iter(allowed_docs))

    results = _unpack(_retriever().search(query, k=k * FETCH_MULTIPLIER, filter=metadata_filter))
    hits = [_to_hit(doc, score) for doc, score in results]
    if allowed_docs is not None:
        hits = [h for h in hits if h.doc_id in allowed_docs]
    return hits[:k]


def get_chunk(chunk_id: str, expand_section: bool = False) -> SearchHit | None:
    """One chunk by id. With expand_section, the whole section it belongs to, joined in
    order and returned under the requested chunk_id (so citations stay valid)."""
    retriever = _retriever()
    doc = retriever.get_chunk(chunk_id)
    if doc is None:
        return None
    hit = _to_hit(doc)
    if not expand_section or not hit.section:
        return hit
    parts = [_to_hit(d) for d in retriever.get_section(hit.doc_id, hit.section)]
    if len(parts) <= 1:
        return hit
    text = "\n".join(re.sub(r"</?untrusted_document>", "", p.text) for p in parts)
    return hit.model_copy(update={"text": wrap_untrusted(text), "suspicious": any(p.suspicious for p in parts)})


@lru_cache
def _load_requirements() -> tuple[RequirementControl, ...]:
    raw = json.loads(resources.files("hackathon2.mcp_server").joinpath("data/requirements.json").read_text("utf-8"))
    controls = tuple(RequirementControl.model_validate(c) for c in raw["controls"])
    ids = [c.id for c in controls]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate control ids in requirements.json")
    return controls


def get_requirements(domain: Domain) -> list[RequirementControl]:
    """Mandatory controls for one domain, from the reviewed checklist."""
    return [c for c in _load_requirements() if c.domain == domain]
