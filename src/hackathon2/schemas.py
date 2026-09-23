"""Shared contracts -- the data shapes every part of the system agrees on.

Every package imports from here; nobody redefines these shapes locally. Change a
contract only by ADDING an optional field (with a default), and announce it to
the team, so nobody else's code breaks.

Who produces / consumes what:

    AssessmentRequest   API (FR01)            -> orchestrator
    SearchHit, ToolResult  MCP server (FR06)  -> specialists
    Evidence            specialists           -> Finding.citations (FR04)
    Finding             specialists           -> decision gate (FR05, FR11)
    DomainReport        each specialist       -> orchestrator (FR08)
    AssessmentDraft     orchestrator LLM      -> decision gate
    Assessment          decision gate (code)  -> human review, API, evaluation (FR12, FR13)
    HumanDecision       reviewer via API      -> resumes the run (FR13)
    AssessmentResponse  API                   -> caller / demo UI / evaluation suite

Two rules the shapes encode:

1. The LLM never sets system fields. It writes an AssessmentDraft; code turns it
   into an Assessment and alone decides `human_approval`, `gate_notes` and
   `degraded_mode`. That is how "no automated approval of High-risk vendors"
   is enforced by code instead of by prompt.
2. Missing evidence is a status, not an absence. A mandatory control with no
   vendor evidence becomes a Finding with status MISSING -- never a silent PASS.

MCP tools (all return a ToolResult envelope; they never raise to the agent):

    get_policy_requirements(domain)                          -> ToolResult[RequirementControl]
    search_policy(query, domain=None, k=5)                   -> ToolResult[SearchHit]
    search_vendor_documents(query, vendor_id, doc_id=None, k=5) -> ToolResult[SearchHit]
    retrieve_document(chunk_id, expand_section=True)         -> ToolResult[SearchHit]
    get_vendor_history(vendor_id)                            -> ToolResult[dict]
    calculate_tco(vendor_id, seats, years)                   -> ToolResult[TCOResult]
    get_budget(category)                                     -> ToolResult[dict]
    retrieve_prior_assessments(vendor_id=None, category=None) -> ToolResult[dict]
    record_assessment(assessment, approval_token)            -> ToolResult[dict]   (restricted)

Note for structured output: pass these models to
`llm.with_structured_output(Model, method="function_calling")`. The stricter
json_schema mode rejects defaults and some constraints.
"""

import re
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------------------
# Vocabulary -- the only allowed values. A typo fails validation instead of passing.
# --------------------------------------------------------------------------------------

Domain = Literal["security", "procurement", "legal", "ai_governance"]

EvidenceStatus = Literal[
    "SUPPORTED",  # vendor evidence satisfies the requirement, with citations
    "INFERRED",  # reasonable conclusion, but no direct evidence -- say so
    "MISSING",  # requirement exists, no vendor evidence found (never a PASS)
    "CONTRADICTED",  # vendor sources disagree with each other
    "NON_COMPLIANT",  # vendor evidence shows the requirement is NOT met
]

Severity = Literal["low", "medium", "high", "critical"]

Recommendation = Literal["APPROVE", "CONDITIONAL_APPROVAL", "REJECT"]

ApprovalStatus = Literal["not_required", "pending", "approved", "rejected"]

DataClassification = Literal["public", "internal", "confidential", "restricted"]

DocType = Literal["policy", "vendor_claim", "enterprise_record"]

ToolStatus = Literal["ok", "unavailable", "denied", "error"]

RunStatus = Literal["completed", "awaiting_approval", "rejected_by_reviewer", "failed"]

SEVERITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# Statuses that make a material claim -- these must carry at least one citation.
CITATION_REQUIRED: frozenset[str] = frozenset({"SUPPORTED", "CONTRADICTED", "NON_COMPLIANT"})

# Statuses the report must surface as "missing or contradictory evidence".
EVIDENCE_GAPS: frozenset[str] = frozenset({"MISSING", "CONTRADICTED"})


def max_severity(severities: list[str], default: Severity = "low") -> Severity:
    """Highest severity in a list ("critical" > "high" > "medium" > "low")."""
    return max(severities, key=SEVERITY_ORDER.__getitem__, default=default)  # type: ignore[return-value]


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------------------
# FR01 -- the request coming in through POST /assessments
# --------------------------------------------------------------------------------------


class AssessmentRequest(BaseModel):
    """A structured vendor assessment request. Unknown fields are rejected."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "vendor_name": "Asteria AI Systems",
                    "use_case": "Enterprise Generative AI platform",
                    "user_count": 2000,
                    "data_classification": "confidential",
                    "contract_years": 3,
                    "requested_by": "procurement@northstar.example",
                    "notes": "The platform may process confidential corporate documents.",
                }
            ]
        },
    )

    vendor_name: str = Field(min_length=2, max_length=120)
    vendor_id: str | None = Field(
        default=None, description="Slug used by MCP tools; derived from vendor_name when omitted."
    )
    use_case: str = Field(min_length=3, max_length=300)
    user_count: int = Field(gt=0, le=1_000_000)
    data_classification: DataClassification
    contract_years: int = Field(default=3, ge=1, le=10)
    requested_by: str | None = Field(default=None, max_length=120)
    notes: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _derive_vendor_id(self) -> "AssessmentRequest":
        if not self.vendor_id:
            self.vendor_id = re.sub(r"[^a-z0-9]+", "-", self.vendor_name.lower()).strip("-")
        return self


# --------------------------------------------------------------------------------------
# FR06 -- MCP tool results (the server returns these; the agent side parses them)
# --------------------------------------------------------------------------------------


class SearchHit(BaseModel):
    """One retrieved chunk, as returned by the knowledge tools."""

    chunk_id: str = Field(description="Stable id, e.g. 'information-security-policy#s4.2#c1'.")
    doc_id: str
    source: str = Field(description="File name, e.g. 'information-security-policy.pdf'.")
    doc_type: DocType
    domain: Domain | Literal["data", "general"] = "general"
    section: str | None = None
    page: int | None = None
    suspicious: bool = Field(default=False, description="Flagged by the injection scanner at ingestion.")
    score: float | None = None
    text: str = Field(description="Chunk text, wrapped in <untrusted_document> tags by the server.")

    def to_evidence(self, quote: str | None = None) -> "Evidence":
        """Turn a hit into a citation. Pass the exact supporting sentence as `quote` when possible."""
        text = quote or re.sub(r"</?untrusted_document>", "", self.text).strip()
        return Evidence(
            chunk_id=self.chunk_id,
            source=self.source,
            doc_type=self.doc_type,
            section=self.section,
            page=self.page,
            quote=text[:500],
        )


class RequirementControl(BaseModel):
    """One mandatory control from the reviewed requirements checklist (get_policy_requirements)."""

    id: str = Field(description="e.g. 'SEC-04'")
    domain: Domain
    control: str = Field(description="e.g. 'Encryption at rest for confidential data'")
    mandatory: bool = True
    source_chunk_id: str = Field(description="Policy chunk the control was extracted from.")


class TCOResult(BaseModel):
    """calculate_tco output -- computed in code from the pricing data, never by the LLM."""

    vendor_id: str
    seats: int
    years: int
    currency: str = "EUR"
    total: float
    breakdown: dict[str, float] = Field(default_factory=dict)
    source_chunk_id: str | None = Field(default=None, description="Pricing chunk the inputs came from.")


class ToolResult(BaseModel):
    """The envelope every MCP tool returns. Tools report failures here; they never raise.

    status:
      ok          -- `results` holds the data (may be empty: "searched, found nothing")
      unavailable -- backend down / timeout; affected controls become MISSING, not PASS (FR15)
      denied      -- caller lacks authorization (e.g. record_assessment without approval)
      error       -- bad arguments or unexpected failure; `error` explains
    """

    status: ToolStatus
    results: list[dict] = Field(default_factory=list)
    error: str | None = None

    @classmethod
    def ok(cls, results: list[BaseModel | dict]) -> "ToolResult":
        return cls(status="ok", results=[r.model_dump() if isinstance(r, BaseModel) else r for r in results])

    @classmethod
    def fail(cls, status: Literal["unavailable", "denied", "error"], error: str) -> "ToolResult":
        return cls(status=status, error=error)

    def hits(self) -> list[SearchHit]:
        """Parse `results` as SearchHits (for the knowledge tools)."""
        return [SearchHit.model_validate(r) for r in self.results]


# --------------------------------------------------------------------------------------
# FR04 / FR05 / FR11 -- evidence and findings (what specialists produce)
# --------------------------------------------------------------------------------------


class Evidence(BaseModel):
    """One citation. `chunk_id` must be a chunk actually retrieved during this run --
    the decision gate checks it against the run's retrieval log."""

    chunk_id: str
    source: str
    doc_type: DocType
    section: str | None = None
    page: int | None = None
    quote: str = Field(max_length=500, description="The exact sentence(s) that support the claim.")


class Finding(BaseModel):
    """The result of checking one control in one domain."""

    domain: Domain
    control_id: str = Field(description="Control id from the requirements checklist, e.g. 'SEC-04'.")
    title: str = Field(max_length=150)
    status: EvidenceStatus
    severity: Severity
    claim: str = Field(max_length=1000, description="What we conclude, in one or two sentences.")
    citations: list[Evidence] = Field(default_factory=list)
    remediation: str | None = Field(
        default=None, max_length=500, description="What the vendor must do / contract must say to close this."
    )

    @property
    def is_evidence_backed(self) -> bool:
        """False when a material status has no citation. The gate downgrades such findings
        to INFERRED rather than failing validation, so the run never crashes on it."""
        return self.status not in CITATION_REQUIRED or bool(self.citations)


class DomainReport(BaseModel):
    """What ONE specialist subagent returns to the orchestrator."""

    domain: Domain
    risk_rating: Severity
    summary: str = Field(max_length=1500)
    findings: list[Finding]

    @field_validator("findings")
    @classmethod
    def _findings_match_domain(cls, findings: list[Finding], info) -> list[Finding]:
        domain = info.data.get("domain")
        wrong = [f.control_id for f in findings if domain and f.domain != domain]
        if wrong:
            raise ValueError(f"findings {wrong} do not belong to domain '{domain}'")
        return findings


# --------------------------------------------------------------------------------------
# FR12 / FR13 -- the assessment (LLM draft -> gate -> final)
# --------------------------------------------------------------------------------------


class Condition(BaseModel):
    """A required remediation or contractual condition attached to the recommendation."""

    kind: Literal["remediation", "contractual"]
    text: str = Field(max_length=500)
    control_ids: list[str] = Field(default_factory=list)
    before_go_live: bool = True


class AssessmentDraft(BaseModel):
    """What the orchestrator LLM produces. Contains NO system-controlled fields."""

    recommendation: Recommendation
    risk_rating: Severity
    domains: list[DomainReport]
    conditions: list[Condition] = Field(default_factory=list)
    executive_summary: str = Field(max_length=3000)


class Assessment(AssessmentDraft):
    """The final, gate-checked assessment. Built by code from an AssessmentDraft."""

    assessment_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    vendor_name: str
    vendor_id: str
    human_approval: ApprovalStatus = "not_required"
    reviewer: str | None = None
    reviewer_comment: str | None = None
    gate_notes: list[str] = Field(
        default_factory=list, description="Every change the decision gate made to the LLM's draft, and why."
    )
    degraded_mode: bool = Field(default=False, description="True when a tool or source was unavailable.")
    created_at: datetime = Field(default_factory=_utcnow)

    @classmethod
    def from_draft(cls, draft: AssessmentDraft, request: AssessmentRequest) -> "Assessment":
        return cls(**draft.model_dump(), vendor_name=request.vendor_name, vendor_id=request.vendor_id or "")

    # --- views the report and the evaluation suite use ---------------------------------

    @property
    def findings(self) -> list[Finding]:
        return [f for d in self.domains for f in d.findings]

    @property
    def evidence_gaps(self) -> list[Finding]:
        """Missing or contradictory evidence (must be clearly identified in the report)."""
        return [f for f in self.findings if f.status in EVIDENCE_GAPS]

    @property
    def non_compliant(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "NON_COMPLIANT"]

    @property
    def cited_chunk_ids(self) -> set[str]:
        return {c.chunk_id for f in self.findings for c in f.citations}

    @property
    def domains_covered(self) -> set[str]:
        return {d.domain for d in self.domains}


class HumanDecision(BaseModel):
    """Body of POST /assessments/{id}/decision -- resumes an interrupted run."""

    model_config = ConfigDict(extra="forbid")

    approved: bool
    reviewer: str = Field(min_length=2, max_length=120)
    comment: str | None = Field(default=None, max_length=2000)


# --------------------------------------------------------------------------------------
# API responses
# --------------------------------------------------------------------------------------


class RunMetrics(BaseModel):
    """Per-run numbers the evaluation suite and telemetry report (latency / cost / tools)."""

    duration_seconds: float | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tools_called: list[str] = Field(default_factory=list, description="MCP tool names, in call order.")
    subagents_called: list[str] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)


class AssessmentResponse(BaseModel):
    """What POST /assessments, GET /assessments/{id} and the decision endpoint return."""

    assessment_id: str
    status: RunStatus
    assessment: Assessment | None = None
    metrics: RunMetrics | None = None
    error: str | None = None
