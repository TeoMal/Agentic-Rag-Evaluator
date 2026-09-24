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

Which documents belong to which vendor is NOT hard-coded: data/vendors.json maps a
vendor_id to its documents. For the hidden vendor case, add one entry there (and put
its PDFs in knowledge/ for ingestion) -- no code change.

Historical assessments (knowledge/historical-vendor-assessments/) are NFS's own records:
doc_type "enterprise_record", never vendor claims.

The mandatory controls checklist is data/requirements.json in this package, written
from the five NFS policies and reviewed by a human.

Failures: if the retriever is missing or raises, the error propagates and the
server's safe_tool turns it into status "unavailable" -- the agent then records the
evidence as missing/UNKNOWN instead of crashing (FR14).
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
    "vendor-alpha-assessment": ("historical-vendor-assessments/vendor-alpha-assessment.pdf", "enterprise_record"),
    "vendor-beta-assessment": ("historical-vendor-assessments/vendor-beta-assessment.pdf", "enterprise_record"),
    "vendor-gamma-assessment": ("historical-vendor-assessments/vendor-gamma-assessment.pdf", "enterprise_record"),
}

# Over-fetch when filtering after retrieval (vendor/doc filters), then trim to k.
FETCH_MULTIPLIER = 4

# Second-layer injection check at the MCP boundary. Ingestion may flag chunks too; this
# catches anything it missed. A flagged chunk is still returned -- it may contain real
# evidence -- but marked, so the agent and the decision gate treat it with suspicion.
#
# Validated against the whole knowledge pack: it flags the four injection signals in
# vendor-x-proposal section 7 and nothing else -- including the policy sentences that talk
# ABOUT injection ("...must not override system policies") or say "AI may recommend Approve".
_NOT = r"(?<!\bnot\s)(?<!\bnever\s)(?<!\bcannot\s)"
_INJECTION_PATTERNS = re.compile(
    "|".join(
        [
            # "ignore all previous policies and instructions", "disregard the above rules"
            (
                r"\b(ignore|disregard|forget)\b[^.]{0,40}\b(previous|prior|above|earlier|all|any|these|those|your)\b"
                r"[^.]{0,40}\b(instructions?|polic(y|ies)|rules|guidelines|prompts?|controls)\b"
            ),
            # "bypass the guardrails" -- but not "must NOT override system policies"
            _NOT + r"\b(override|bypass|circumvent)\b[^.]{0,30}\b(instructions?|system prompt|guardrails?|safety)\b",
            # "note for automated review systems", "message to the AI reviewer"
            (
                r"\b(note|message|instructions?|attention)\b[^.]{0,15}\b(for|to)\b[^.]{0,10}"
                r"\b(automated|ai|llm|language model|assistant|agent|bot)s?\b"
            ),
            # "return 'APPROVE - LOW RISK'", "respond with PASS"
            r"\b(return|output|respond with|reply with|answer with)\b\s*['\"`]?\s*(approve[ds]?|low risk|pass(ed)?|compliant)\b",
            # "recommend 'APPROVE'" -- only when quoted (policies legitimately say "AI may recommend Approve")
            r"\b(recommend|say|state|write)\b\s*['\"`]\s*(approve[ds]?|low risk|pass(ed)?)\b",
            # "do not mention data retention"
            r"\bdo not (mention|disclose|report|flag|include|reveal)\b",
            # role hijacking
            r"\byou are now\b|\bnew (system )?instructions\b|\bact as\b[^.]{0,20}\b(unrestricted|different|developer mode)\b",
            # attempts to close our wrapper or fake a system block
            r"<\s*/?\s*(system|untrusted_document)\s*>",
        ]
    ),
    re.IGNORECASE,
)


def injection_signals(text: str) -> list[str]:
    """The injection-like phrases found in a text (line breaks ignored, so a chunk
    boundary or PDF line wrap cannot hide a phrase)."""
    flat = re.sub(r"\s+", " ", text)
    return [m.group(0).strip() for m in _INJECTION_PATTERNS.finditer(flat)]


def looks_like_injection(text: str) -> bool:
    return bool(injection_signals(text))


def wrap_untrusted(text: str) -> str:
    """Retrieved text is data, never instructions. Tags inside the text are neutralised
    so a document cannot close the wrapper and 'escape' into instructions."""
    safe = re.sub(r"<\s*(/?)\s*untrusted_document\s*>", r"[\1untrusted_document]", text, flags=re.IGNORECASE)
    return f"<untrusted_document>{safe}</untrusted_document>"


# --------------------------------------------------------------------------------------
# Vendor registry (data/vendors.json)
# --------------------------------------------------------------------------------------


def _read_data(name: str) -> dict:
    text = resources.files("hackathon2.mcp_server").joinpath(f"data/{name}").read_text("utf-8")
    return {k: v for k, v in json.loads(text).items() if not k.startswith("_")}


@lru_cache
def _vendors() -> dict[str, dict]:
    vendors = _read_data("vendors.json")["vendors"]
    for vendor_id, entry in vendors.items():
        if not entry.get("documents"):
            raise ValueError(f"vendors.json: '{vendor_id}' lists no documents")
    return vendors


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def resolve_vendor(vendor_ref: str) -> str:
    """Accept a vendor id, name or alias ('asteria-ai-systems', 'Asteria AI Systems', 'Asteria')
    and return the registered vendor_id. Unknown vendors raise ValueError with guidance."""
    wanted = _norm(vendor_ref)
    for vendor_id, entry in _vendors().items():
        names = [vendor_id, entry.get("name", ""), *entry.get("aliases", [])]
        if wanted in {_norm(n) for n in names if n}:
            return vendor_id
    raise ValueError(
        f"unknown vendor '{vendor_ref}'. Registered vendors: {sorted(_vendors())}. "
        "A new vendor must be added to mcp_server/data/vendors.json with its document ids."
    )


def vendor_documents(vendor_id: str) -> tuple[str, ...]:
    return tuple(_vendors()[resolve_vendor(vendor_id)]["documents"])


def _default_doc_type(doc_id: str) -> DocType:
    """doc_type when chunk metadata does not say: known docs, then registered vendor
    documents, then naming conventions."""
    if doc_id in KNOWLEDGE_DOCS:
        return KNOWLEDGE_DOCS[doc_id][1]
    if any(doc_id in entry["documents"] for entry in _vendors().values()):
        return "vendor_claim"
    if doc_id.endswith("-assessment"):
        return "enterprise_record"
    return "vendor_claim" if doc_id.startswith("vendor-") else "policy"


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
        default_source = KNOWLEDGE_DOCS.get(doc_id, (f"{doc_id}.pdf", "policy"))[0]
        default_type = _default_doc_type(doc_id)
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
    signals = injection_signals(raw_text)
    suspicious = flagged or bool(signals)
    if signals and not flagged:
        logger.warning("injection-like text in %s (missed at ingestion): %s", hit.chunk_id, signals)
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
        if not vendor_id:
            raise ValueError("vendor_id is required to search vendor documents")
        allowed_docs = set(vendor_documents(vendor_id))  # raises ValueError for an unknown vendor
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
    controls = tuple(RequirementControl.model_validate(c) for c in _read_data("requirements.json")["controls"])
    ids = [c.id for c in controls]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate control ids in requirements.json")
    return controls


def get_requirements(domain: Domain) -> list[RequirementControl]:
    """Mandatory controls for one domain, from the reviewed checklist."""
    return [c for c in _load_requirements() if c.domain == domain]
