"""Validate outputs against caller-supplied provenance; no business synthesis.

The gate never changes recommendations, citation quotes, caller models or workflow
state. Provenance checks cannot prove semantic entailment. Human approval fields in
an Assessment are not authentication and cannot bypass review.
"""

from collections.abc import Mapping

from hackathon2.schemas import CITATION_REQUIRED, Assessment, AssessmentDraft, RequirementControl, SearchHit

from .policy import (
    DEFAULT_LIMITS,
    DEFAULT_PRIVACY_POLICY,
    Decision,
    GateContext,
    Limits,
    PrivacyPolicy,
    Reason,
    _decision,
    _PayloadError,
    _snapshot,
)
from .privacy import check_privacy


def gate_assessment(
    assessment: AssessmentDraft | Assessment,
    *,
    context: GateContext,
    privacy_policy: PrivacyPolicy = DEFAULT_PRIVACY_POLICY,
    limits: Limits = DEFAULT_LIMITS,
) -> Decision:
    """Return deny for invalid/forged data; require_review for risk or evidence gaps.

    A low-risk, fully evidenced REJECT can be allowed as an output. Allow never
    grants permission to record it: separately call authorize_tool_call. Both
    approval recommendations always require review, even on an Assessment with
    human_approval='approved'; trusted approval verification is an action boundary.
    """
    if not isinstance(assessment, AssessmentDraft) or not isinstance(limits, Limits):
        return _decision("deny", Reason.INVALID_OUTPUT)
    try:
        payload = _snapshot(assessment, limits)
        model = Assessment if isinstance(assessment, Assessment) else AssessmentDraft
        draft = model.model_validate(payload, strict=True, extra="forbid")
    except _PayloadError as exc:
        return _decision("deny", exc.reason)
    except Exception:  # noqa: BLE001 - malformed output must deny without exposing validation payloads.
        return _decision("deny", Reason.INVALID_OUTPUT)
    privacy = check_privacy(payload, policy=privacy_policy, limits=limits)
    if privacy.outcome != "allow":
        return privacy

    try:
        if not isinstance(context, GateContext):
            return _decision("deny", Reason.INVALID_PROVENANCE)
        if any(type(v) is not str or not v.strip() for v in (context.run_id, context.ledger_run_id)):
            return _decision("deny", Reason.INVALID_PROVENANCE)
        if context.run_id != context.ledger_run_id or not isinstance(context.ledger, Mapping):
            return _decision("deny", Reason.INVALID_PROVENANCE)
        if len(context.ledger) > limits.max_hits:
            return _decision("deny", Reason.LIMIT_EXCEEDED)
        if type(context.required_controls) is not tuple or type(context.quarantined_chunk_ids) is not frozenset:
            return _decision("deny", Reason.INVALID_PROVENANCE)
        if len(context.required_controls) > limits.max_nodes or len(context.quarantined_chunk_ids) > limits.max_nodes:
            return _decision("deny", Reason.LIMIT_EXCEEDED)
        if type(context.tool_statuses) is not tuple or len(context.tool_statuses) > limits.max_nodes:
            return _decision("deny", Reason.INVALID_PROVENANCE)
        if any(type(v) is not bool for v in (context.evidence_gap, context.scanner_failed, context.verifier_failed)):
            return _decision("deny", Reason.INVALID_PROVENANCE)
        if any(s not in ("ok", "unavailable", "denied", "error") for s in context.tool_statuses):
            return _decision("deny", Reason.INVALID_PROVENANCE)
        if any(type(key) is not str or not key for key in context.quarantined_chunk_ids):
            return _decision("deny", Reason.INVALID_PROVENANCE)
        if any(not isinstance(h, SearchHit) for h in context.ledger.values()):
            return _decision("deny", Reason.INVALID_PROVENANCE)
        if any(not isinstance(c, RequirementControl) for c in context.required_controls):
            return _decision("deny", Reason.INVALID_PROVENANCE)
        data = _snapshot(
            {
                "run_id": context.run_id,
                "ledger_run_id": context.ledger_run_id,
                "ledger": dict(context.ledger),
                "controls": context.required_controls,
                "quarantined": list(context.quarantined_chunk_ids),
                "statuses": context.tool_statuses,
            },
            limits,
        )
        ledger = {
            key: SearchHit.model_validate(hit, strict=True, extra="forbid") for key, hit in data["ledger"].items()
        }
        if any(not key or key != hit.chunk_id for key, hit in ledger.items()):
            return _decision("deny", Reason.INVALID_PROVENANCE)
        controls = [RequirementControl.model_validate(c, strict=True, extra="forbid") for c in data["controls"]]
        control_keys = [(c.domain, c.id) for c in controls]
        if len(set(control_keys)) != len(control_keys):
            return _decision("deny", Reason.INVALID_PROVENANCE)
    except _PayloadError as exc:
        return _decision("deny", exc.reason)
    except Exception:  # noqa: BLE001 - malformed provenance must deny without exposing validation payloads.
        return _decision("deny", Reason.INVALID_PROVENANCE)

    denied = []
    review = []
    if context.scanner_failed:
        denied.append(Reason.SCANNER_FAILURE)
    if context.verifier_failed:
        denied.append(Reason.VERIFIER_FAILURE)
    if any(status != "ok" for status in context.tool_statuses):
        review.append(Reason.TOOL_FAILURE)
    if context.evidence_gap:
        review.append(Reason.EVIDENCE_GAP)
    if isinstance(draft, Assessment):
        if draft.degraded_mode:
            review.append(Reason.DEGRADED_MODE)
        if draft.human_approval == "rejected":
            denied.append(Reason.REVIEW_REJECTED)
    if draft.recommendation in ("APPROVE", "CONDITIONAL_APPROVAL"):
        review.append(Reason.FINAL_APPROVAL)
    if draft.risk_rating in ("high", "critical"):
        review.append(Reason.HIGH_RISK)
    if not draft.domains:
        review.append(Reason.EVIDENCE_GAP)
    seen = set()
    domains = set()
    for report in draft.domains:
        if report.domain in domains:
            denied.append(Reason.INVALID_OUTPUT)
        domains.add(report.domain)
        if not report.findings:
            review.append(Reason.EVIDENCE_GAP)
        if report.risk_rating in ("high", "critical"):
            review.append(Reason.HIGH_RISK)
        for finding in report.findings:
            key = (finding.domain, finding.control_id)
            if key in seen:
                denied.append(Reason.INVALID_OUTPUT)
            seen.add(key)
            if finding.severity in ("high", "critical"):
                review.append(Reason.HIGH_RISK)
            if finding.status in ("MISSING", "INFERRED", "CONTRADICTED"):
                review.append(Reason.EVIDENCE_GAP)
            if finding.status in CITATION_REQUIRED and not finding.citations:
                denied.append(Reason.CITATION_REQUIRED)
            for citation in finding.citations:
                hit = ledger.get(citation.chunk_id)
                if citation.chunk_id in context.quarantined_chunk_ids or (hit is not None and hit.suspicious):
                    denied.append(Reason.QUARANTINED_CITATION)
                elif hit is None:
                    denied.append(Reason.UNKNOWN_CITATION)
                elif (
                    any(getattr(citation, f) != getattr(hit, f) for f in ("source", "doc_type", "section", "page"))
                    or not citation.quote.strip()
                    or citation.quote not in hit.text
                ):
                    denied.append(Reason.CITATION_MISMATCH)
    if any(c.mandatory and (c.domain, c.id) not in seen for c in controls):
        review.append(Reason.EVIDENCE_GAP)
    if denied:
        return _decision("deny", *denied, *review)
    return _decision("require_review", *review) if review else Decision("allow")
