"""The decision gate: the guardrails' gate_assessment applied to the agents' draft, with this run's evidence.

    result = apply_gate(draft, request, evidence)     # GateResult(assessment, decision)

1. Code repairs what it can check -- every change is written to Assessment.gate_notes:
   - a citation of a chunk NOT retrieved in this run, or of a chunk flagged as a possible prompt
     injection, is removed;
   - a citation whose source / doc_type / section / page differ from the retrieved chunk takes the
     chunk's values (missing values are filled in the same way, counted in one note);
   - a quote that matches the chunk only up to whitespace (PDF line breaks) becomes the chunk's exact
     text; a quote that is not in the chunk at all removes the citation;
   - a SUPPORTED / CONTRADICTED / NON_COMPLIANT finding left without citations becomes INFERRED, and so does
     one left with no vendor citation: those statuses describe what the vendor does, and an NFS policy
     proves only what NFS requires (FR05); a SUPPORTED / NON_COMPLIANT finding also needs a policy citation,
     because it compares the vendor with a requirement the reader must be able to see;
   - a second finding for the same control is dropped;
   - a finding for a control id that is not in its domain's checklist is dropped (the checklist is the
     scope of the assessment; the specialist may not invent controls);
   - a mandatory control the specialist reported nothing on is added as a MISSING finding, severity
     high (schemas rule 2: missing evidence is a status, never an absence -- and never a pass).
   Consistency rules, applied last (VR-006: missing evidence and mandatory failures drive the rating up):
   - a domain rating below its worst open finding (NON_COMPLIANT / CONTRADICTED / MISSING) is raised to it,
     and the overall rating is raised to the highest domain rating -- ratings are only ever raised;
   - in a CONDITIONAL_APPROVAL, every open finding no condition covers gets one, from its remediation.
   The recommendation and the agents' own claims are never changed: that is not code's call. Every
   correction is a gate note, and the evaluation counts them (evaluation/run.py: gate_corrections),
   so what the model did and what the code fixed stay separately visible.
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
    SEVERITY_ORDER,
    Assessment,
    AssessmentDraft,
    AssessmentRequest,
    Condition,
    DomainReport,
    Evidence,
    Finding,
    RequirementControl,
    SearchHit,
    ToolStatus,
    max_severity,
)

# The ledger and the assessment are collected by our own code, not taken from a caller, so the
# guardrails' default size limits (sized for untrusted payloads) are raised to fit a full run.
GATE_LIMITS = Limits(max_text_chars=65_536, max_payload_chars=2_000_000, max_hits=1_000)

MAX_QUOTE = 500  # schemas.Evidence.quote

# The note for controls added as MISSING; evaluation/checks.py reads it to count skipped controls.
SKIPPED_NOTE = "{ids}: no finding from the {domain} specialist -> MISSING (added by the gate)."

# Notes for the consistency corrections; evaluation/run.py counts them (gate_corrections).
DROPPED_NOTE = "{ids}: not in the {domain} checklist -> finding dropped (added by the gate)."
RAISED_NOTE = "{scope} risk_rating raised from {old} to {new} -- never below {why} (added by the gate)."
COVERED_NOTE = "{ids}: open finding without a condition -> condition added from its remediation (added by the gate)."

# Findings that leave a risk open (evaluation/checks.py uses the same set).
OPEN_STATUSES = frozenset({"NON_COMPLIANT", "CONTRADICTED", "MISSING"})

# Statuses that compare the vendor with an NFS requirement: they need the requirement (a policy citation) as
# well as the vendor's evidence. CONTRADICTED compares vendor documents with each other, so vendor citations do.
COMPARED_WITH_POLICY = frozenset({"SUPPORTED", "NON_COMPLIANT"})

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
    domains = [_drop_unknown_controls(report, evidence.required_controls, notes) for report in domains]
    domains = [_add_skipped_controls(report, evidence.required_controls, notes) for report in domains]
    domains = [_raise_domain_rating(report, notes) for report in domains]
    repaired = _raise_overall_rating(draft.model_copy(update={"domains": domains}), notes)
    repaired = _cover_open_findings(repaired, notes)
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


def _add_skipped_controls(
    report: DomainReport, controls: Sequence[RequirementControl], notes: list[str]
) -> DomainReport:
    reported = {f.control_id for f in report.findings}
    skipped = [c for c in controls if c.mandatory and c.domain == report.domain and c.id not in reported]
    if not skipped:
        return report
    added = [
        Finding(
            domain=report.domain,
            control_id=c.id,
            title=c.control[:150],
            status="MISSING",
            severity="high",
            claim="No finding was reported for this mandatory control, so no evidence was assessed. "
            "It is recorded as UNKNOWN, not as passed.",
            remediation="Assess this control (re-run or manual review) before approval.",
        )
        for c in skipped
    ]
    notes.append(SKIPPED_NOTE.format(ids=", ".join(c.id for c in skipped), domain=report.domain))
    return report.model_copy(update={"findings": [*report.findings, *added]})


def _drop_unknown_controls(
    report: DomainReport, controls: Sequence[RequirementControl], notes: list[str]
) -> DomainReport:
    """Keep findings for the domain's checklist controls (and the "-00" not-assessed placeholder).
    Without a checklist for the domain (the fetch failed), nothing is dropped."""
    known = {c.id for c in controls if c.domain == report.domain}
    if not known:
        return report
    kept, dropped = [], []
    for f in report.findings:
        if f.control_id in known or f.control_id.endswith("-00"):
            kept.append(f)
        else:
            dropped.append(f"{f.control_id} ({f.status})")
    if not dropped:
        return report
    notes.append(DROPPED_NOTE.format(ids=", ".join(dropped), domain=report.domain))
    return report.model_copy(update={"findings": kept})


def _raise_domain_rating(report: DomainReport, notes: list[str]) -> DomainReport:
    open_severities = [f.severity for f in report.findings if f.status in OPEN_STATUSES]
    if not open_severities:
        return report
    worst = max_severity(open_severities)
    if SEVERITY_ORDER[report.risk_rating] >= SEVERITY_ORDER[worst]:
        return report
    notes.append(
        RAISED_NOTE.format(scope=report.domain, old=report.risk_rating, new=worst, why="its worst open finding")
    )
    return report.model_copy(update={"risk_rating": worst})


def _raise_overall_rating(draft: AssessmentDraft, notes: list[str]) -> AssessmentDraft:
    highest = max_severity([d.risk_rating for d in draft.domains]) if draft.domains else draft.risk_rating
    if SEVERITY_ORDER[draft.risk_rating] >= SEVERITY_ORDER[highest]:
        return draft
    notes.append(RAISED_NOTE.format(scope="overall", old=draft.risk_rating, new=highest, why="a domain rating"))
    return draft.model_copy(update={"risk_rating": highest})


def _cover_open_findings(draft: AssessmentDraft, notes: list[str]) -> AssessmentDraft:
    """A conditional approval lists what must be closed: every open finding needs a condition."""
    if draft.recommendation != "CONDITIONAL_APPROVAL":
        return draft
    covered = {control_id for condition in draft.conditions for control_id in condition.control_ids}
    uncovered = [
        f for d in draft.domains for f in d.findings if f.status in OPEN_STATUSES and f.control_id not in covered
    ]
    if not uncovered:
        return draft
    added = [
        Condition(
            kind="remediation",
            text=(f.remediation or f"Close {f.control_id} ({f.title}) with evidence before go-live.")[:500],
            control_ids=[f.control_id],
            before_go_live=True,
        )
        for f in uncovered
    ]
    notes.append(COVERED_NOTE.format(ids=", ".join(f.control_id for f in uncovered)))
    return draft.model_copy(update={"conditions": [*draft.conditions, *added]})


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
    elif finding.status in CITATION_REQUIRED and not any(c.doc_type == "vendor_claim" for c in citations):
        update["status"] = "INFERRED"
        notes.append(f"{finding.control_id}: {finding.status} without a vendor citation -> INFERRED.")
    elif finding.status in COMPARED_WITH_POLICY and not any(c.doc_type == "policy" for c in citations):
        update["status"] = "INFERRED"
        notes.append(f"{finding.control_id}: {finding.status} without a policy citation -> INFERRED.")
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
