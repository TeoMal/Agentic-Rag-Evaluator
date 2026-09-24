"""RAG-internal data shapes (document metadata, pages, chunks).

Public contracts and vocabularies come from hackathon2.schemas and are never
redefined here: chunks are SearchHits, document types are schemas.DocType and
domains are schemas.Domain. This module only adds what the RAG subsystem needs on
top of them.

    DocumentInfo (registry) -> DocumentPage (loaders) -> DocumentChunk (chunking, retrieval)

Document taxonomy (schemas.DocType):
    "policy"             NFS policy documents -- the requirements
    "vendor_claim"       the vendor under assessment's proposal, questionnaire, pricing -- claims to verify
    "enterprise_record"  NFS records, e.g. historical vendor assessments -- precedent, not requirements
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from hackathon2.schemas import DocType, Domain, SearchHit

# The values SearchHit.domain accepts: a specialist domain, or cross-cutting content
# ("data" for data classification, "general" for documents spanning every domain).
# schemas.py has no name for this union; tests/test_rag_data_models.py keeps it equal
# to SearchHit's annotation.
KnowledgeDomain = Domain | Literal["data", "general"]

# Which vendor a document is about, as named in its file name: 'x' for vendor-x-proposal.pdf,
# 'alpha' for vendor-alpha-assessment.pdf; None for NFS policies. RAG-internal, used for
# filtering; how it maps to AssessmentRequest.vendor_id is decided outside the RAG package.
VendorRef = Annotated[str | None, Field(description="Vendor the document is about; None for NFS policies.")]


class DocumentInfo(BaseModel):
    """Metadata for one knowledge-pack file, assigned by the document registry."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1, description="e.g. 'information-security-policy'.")
    source: str = Field(min_length=1, description="File name, e.g. 'information-security-policy.pdf'.")
    doc_type: DocType
    domain: KnowledgeDomain
    vendor: VendorRef = None


class DocumentPage(DocumentInfo):
    """One PDF page with its document metadata -- the output of loaders.py, the input to chunking.

    `page` is kept here because citations ('information-security-policy.pdf, page 8')
    cannot be rebuilt once it is lost. `text` is exactly what the PDF text layer yields,
    never summarised or rewritten, because it is later quoted as evidence. It may be
    empty (e.g. a scanned page); the page is still returned, and reported as an
    ExtractionIssue, so nothing disappears silently.
    """

    page: int = Field(ge=1, description="1-based PDF page number.")
    text: str


class ChunkFilter(BaseModel):
    """Metadata filter for a vector search: fields are ANDed, the values inside a field are ORed.

    An unset field (None) does not filter. Example -- security or data policies only:
        ChunkFilter(doc_types=["policy"], domains=["security", "data"])
    """

    model_config = ConfigDict(frozen=True)

    doc_types: list[DocType] | None = None
    domains: list[KnowledgeDomain] | None = None
    doc_ids: list[str] | None = None
    vendors: list[str] | None = None

    def as_dict(self) -> dict[str, list[str]]:
        """Only the fields that filter, keyed by the chunk metadata field they apply to."""
        fields = {"doc_type": self.doc_types, "domain": self.domains, "doc_id": self.doc_ids, "vendor": self.vendors}
        return {name: list(values) for name, values in fields.items() if values is not None}

    def matches(self, metadata: dict) -> bool:
        """Whether chunk metadata passes the filter (used wherever filtering happens in Python)."""
        return metadata_matches(metadata, self.as_dict())


def metadata_matches(metadata: dict, conditions: dict[str, list[str]]) -> bool:
    """Every {field: allowed values} condition holds for the metadata."""
    return all(metadata.get(field) in values for field, values in conditions.items())


# Outcome of one retrieval channel -- a statement about the search, never about compliance:
#   "found"         at least one relevant chunk was retrieved
#   "insufficient"  the search ran but retrieved no relevant chunk (none at all, or none reaching
#                   min_score). The corpus may still contain the information: represent it as
#                   UNKNOWN / MISSING evidence, never as PASS and never as "does not exist".
#   "error"         the search could not run (e.g. backend down); nothing is known about the evidence.
RetrievalStatus = Literal["found", "insufficient", "error"]


class ChannelResult(BaseModel):
    """What one evidence channel retrieved, with an explicit status."""

    model_config = ConfigDict(frozen=True)

    doc_type: DocType
    status: RetrievalStatus
    hits: list[SearchHit] = Field(
        default_factory=list, description="Everything retrieved, best first -- also when status is 'insufficient'."
    )
    min_score: float | None = Field(default=None, description="Relevance cut-off applied, if any.")
    detail: str = Field(description="Human-readable account of the outcome, for agents and reports.")

    @property
    def relevant_hits(self) -> list[SearchHit]:
        """Hits reaching min_score (all hits when no cut-off was set)."""
        if self.min_score is None:
            return list(self.hits)
        return [h for h in self.hits if h.score is not None and h.score >= self.min_score]


class EvidenceBundle(BaseModel):
    """Evidence for one question, kept in separate channels by document type. Evidence only -- no verdict.

    Every hit is a schemas.SearchHit with full provenance (chunk_id, source, page,
    section, doc_type, domain, score). Each channel carries its own status, so a
    missing or failed channel is explicit; `historical` is None when it was not requested.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    vendor: str = Field(description="Vendor whose claims were searched, as named in its files ('x').")
    policy: ChannelResult = Field(description="NFS policy requirements (doc_type 'policy').")
    vendor_claims: ChannelResult = Field(description="The vendor's own documents (doc_type 'vendor_claim').")
    historical: ChannelResult | None = Field(
        default=None, description="Historical NFS assessments (doc_type 'enterprise_record'); None if not requested."
    )


class ExtractionIssue(BaseModel):
    """A PDF, or one of its pages, from which no text could be extracted (empty or whitespace only)."""

    model_config = ConfigDict(frozen=True)

    source: str
    page: int | None = Field(default=None, description="None when the whole file is affected.")
    reason: str


class KnowledgeLoad(BaseModel):
    """Result of loading the knowledge directory: the pages, plus every extraction problem found.

    Pages with an issue are still included (with whatever text was extracted), so page
    numbering stays complete; `issues` says which ones cannot serve as evidence.
    """

    documents: list[str] = Field(default_factory=list, description="Source file names of every PDF discovered.")
    pages: list[DocumentPage]
    issues: list[ExtractionIssue] = Field(default_factory=list)


class DocumentChunk(SearchHit):
    """One piece of a knowledge-pack document, with its metadata, as indexed and retrieved by the RAG subsystem.

    Inherits every SearchHit field -- text, chunk_id, doc_id, source, page, section,
    doc_type, domain, suspicious, score -- so retrieval results go to callers as-is,
    and adds only `vendor`, used internally for filtering.

    A chunk is not evidence by itself; it becomes a schemas.Evidence only when a
    specialist cites it in a Finding:

        DocumentChunk (stored) -> SearchHit (returned to callers) -> Evidence (cited)

    `section` is set only when a heading was detected reliably, otherwise None. `text`
    holds the raw chunk text; wrapping it in <untrusted_document> tags happens at the
    boundary (security.py / MCP server).

    Frozen because the same instance may be shared (retrieval log, several
    specialists, caches): an in-place edit would silently change what the others
    see. Derive a changed copy instead, e.g. chunk.model_copy(update={"score": 0.9}).
    """

    model_config = ConfigDict(frozen=True)

    vendor: VendorRef = None
