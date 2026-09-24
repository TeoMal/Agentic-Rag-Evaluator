"""The decision gate: the guardrails' gate_assessment applied to the agents' draft, with this run's evidence.

    result = apply_gate(draft, request, evidence)     # GateResult(assessment, decision)

1. Code repairs what it can check -- every change is written to Assessment.gate_notes:
   - a citation of a chunk NOT retrieved in this run, or of a chunk flagged as a possible prompt
     injection, is removed;
   - a citation whose source / doc_type / section / page differ from the retrieved chunk takes the
     chunk's values (missing values are filled in the same way, counted in one note);
   - a quote that matches the chunk only up to whitespace (PDF line breaks) becomes the chunk's exact
     text; a quote that is not in the chunk at all removes the citation;
   - a SUPPORTED / CONTRADICTED / NON_COMPLIANT finding left without citations becomes INFERRED;
   - a second finding for the same control is dropped.
   The recommendation, ratings and claims are never changed: that is not code's call.
2. hackathon2.guardrails.gate_assessment decides, from the run's provenance (retrieved chunks, the
   mandatory controls, every tool status):
   allow           no human review needed (e.g. an evidenced low-risk REJECT)
   require_review  human_approval = "pending" (approvals, high risk, evidence gaps, tool failures ...)
   deny            blocked: the runner returns status "failed" with the reasons; nobody can approve it.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hackathon2.guardrails import Decision, GateContext, Limits, Reason, gate_assessment
from hackathon2.schemas import (
    CITATION_REQUIRED,
    Assessment,
    AssessmentDraft,
    AssessmentRequest,
    DomainReport,
    Evidence,
    Finding,
    RequirementControl,
    SearchHit,
    ToolStatus,
)

# The ledger and the assessment are collected by our own code, not taken from a caller, so the
# guardrails' default size limits (sized for untrusted payloads) are raised to fit a full run.
GATE_LIMITS = Limits(max_text_chars=65_536, max_payload_chars=2_000_000, max_hits=1_000)

MAX_QUOTE = 500  # schemas.Evidence.quote

_WHY: dict[Reason, str] = {
    Reason.FINAL_APPROVAL: "an APPROVE / CONDITIONAL_APPROVAL recommendation always needs a human decision",
    Reason.HIGH_RISK: "high or critical risk",
    Reason.EVIDENCE_GAP: "missing, inferred or contradictory evidence, or a mandatory control without a finding",
    Reason.TOOL_FAILURE: "a tool was unavailable or returned an error",
    Reason.DEGRADED_MODE: "the run was degraded",
    Reason.INCONSISTENT_ASSESSMENT: "APPROVE despite a NON_COMPLIANT mandatory control",
    Reason.CITATION_REQUIRED: "a material finding has no citation",
    Reason.UNKNOWN_CITATION: "a citation points to a chunk not retrieved in this run",
    Reason.QUARANTINED_CITATION: "a citation points to a chunk flagged as possible prompt injection",
    Reason.CITATION_MISMATCH: "a citation does not match its retrieved chunk",
    Reason.SENSITIVE_DATA: "the assessment contains data marked sensitive",
}


@dataclass(frozen=True)
class RunEvidence:
    """What the run actually did -- collected by code (context.RunContext), never by an LLM."""

    run_id: str
    hits: Mapping[str, SearchHit]  # every chunk retrieved in this run, with its text
    required_controls: tuple[RequirementControl, ...]  # the checklist, from get_policy_requirements
    tool_statuses: tuple[ToolStatus, ...]
    degraded: bool


@dataclass(frozen=True)
class GateResult:
    assessment: Assessment
    decision: Decision

    @property
    def blocked(self) -> bool:
        return self.decision.outcome == "deny"


def explain(decision: Decision) -> str:
    return "; ".join(_WHY.get(r, r.value.replace("_", " ")) for r in decision.reasons)


def apply_gate(draft: AssessmentDraft, request: AssessmentRequest, evidence: RunEvidence) -> GateResult:
    notes: list[str] = []
    filled: list[str] = []  # citations whose missing page/section/... were filled from the chunk
    domains = [_repair_report(report, evidence.hits, notes, filled) for report in draft.domains]
    if filled:
        notes.append(f"{len(filled)} citation(s) completed with page/section from the retrieved chunk.")
    repaired = draft.model_copy(update={"domains": domains})
    assessment = Assessment.from_draft(repaired, request)
    assessment.degraded_mode = evidence.degraded

    cited = {c.chunk_id for f in assessment.findings for c in f.citations}
    context = GateContext(
        run_id=evidence.run_id,
        ledger_run_id=evidence.run_id,
        ledger={chunk_id: evidence.hits[chunk_id] for chunk_id in sorted(cited)},  # every citation is in it
        required_controls=tuple(evidence.required_controls),
        quarantined_chunk_ids=frozenset(chunk_id for chunk_id, hit in evidence.hits.items() if hit.suspicious),
        tool_statuses=tuple(evidence.tool_statuses),
    )
    decision = gate_assessment(assessment, context=context, limits=GATE_LIMITS)

    if decision.outcome == "allow":
        assessment.human_approval = "not_required"
        notes.append("Decision gate: allowed without human review.")
    elif decision.outcome == "require_review":
        assessment.human_approval = "pending"
        notes.append(f"Decision gate: human review required -- {explain(decision)}.")
    else:
        assessment.human_approval = "pending"
        notes.append(f"Decision gate: BLOCKED -- {explain(decision)}. This assessment cannot be approved.")
    assessment.gate_notes.extend(notes)
    return GateResult(assessment, decision)


# --------------------------------------------------------------------------------------
# Repairs
# --------------------------------------------------------------------------------------


def _repair_report(
    report: DomainReport, hits: Mapping[str, SearchHit], notes: list[str], filled: list[str]
) -> DomainReport:
    findings: list[Finding] = []
    seen: set[str] = set()
    for finding in report.findings:
        if finding.control_id in seen:
            notes.append(f"{finding.control_id}: a second finding for this control was dropped.")
            continue
        seen.add(finding.control_id)
        findings.append(_repair_finding(finding, hits, notes, filled))
    return report.model_copy(update={"findings": findings})


def _repair_finding(finding: Finding, hits: Mapping[str, SearchHit], notes: list[str], filled: list[str]) -> Finding:
    citations = []
    for citation in finding.citations:
        fixed = _repair_citation(finding.control_id, citation, hits, notes, filled)
        if fixed is not None:
            citations.append(fixed)
    update: dict = {"citations": citations}
    if finding.status in CITATION_REQUIRED and not citations:
        update["status"] = "INFERRED"
        notes.append(f"{finding.control_id}: {finding.status} without a verifiable citation -> INFERRED.")
    return finding.model_copy(update=update)


def _repair_citation(
    control_id: str, citation: Evidence, hits: Mapping[str, SearchHit], notes: list[str], filled: list[str]
) -> Evidence | None:
    hit = hits.get(citation.chunk_id)
    if hit is None:
        notes.append(f"{control_id}: citation of {citation.chunk_id} removed -- not retrieved in this run.")
        return None
    if hit.suspicious:
        notes.append(
            f"{control_id}: citation of {citation.chunk_id} removed -- the chunk is flagged as possible "
            "prompt injection."
        )
        return None
    quote = _verbatim(citation.quote, hit.text)
    if quote is None:
        notes.append(f"{control_id}: citation of {citation.chunk_id} removed -- the quote is not in that chunk.")
        return None
    fixed = citation.model_copy(
        update={
            "source": hit.source,
            "doc_type": hit.doc_type,
            "section": hit.section,
            "page": hit.page,
            "quote": quote,
        }
    )
    fields = [f for f in ("source", "doc_type", "section", "page") if getattr(citation, f) != getattr(fixed, f)]
    wrong = [f for f in fields if getattr(citation, f) is not None]
    if wrong:
        notes.append(f"{control_id}: {', '.join(wrong)} of {citation.chunk_id} corrected from the retrieved chunk.")
    elif fields:
        filled.append(citation.chunk_id)
    return fixed


def _verbatim(quote: str, text: str) -> str | None:
    """The quote as it appears in `text`: itself, or the text span equal to it up to whitespace."""
    quote = quote.strip()
    if not quote:
        return None
    if quote in text:
        return quote
    words = quote.split()
    match = re.search(r"\s+".join(re.escape(w) for w in words), text)
    if match is None or len(match.group(0)) > MAX_QUOTE:
        return None
    return match.group(0)


def required_controls_from(results: Sequence[dict]) -> tuple[RequirementControl, ...]:
    """RequirementControls from get_policy_requirements results (skipping anything malformed)."""
    controls = []
    for item in results:
        try:
            controls.append(RequirementControl.model_validate(item))
        except ValueError:
            continue
    return tuple(controls)
