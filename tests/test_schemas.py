"""Contract tests -- no LLM, no database. If one of these breaks, a shared shape changed."""

import pytest
from pydantic import ValidationError

from hackathon2.schemas import (
    Assessment,
    AssessmentDraft,
    AssessmentRequest,
    DomainReport,
    Evidence,
    Finding,
    SearchHit,
    ToolResult,
    max_severity,
)

EVIDENCE = Evidence(
    chunk_id="information-security-policy#s4.2#c1",
    source="information-security-policy.pdf",
    doc_type="policy",
    page=3,
    quote="All confidential data must be encrypted at rest.",
)


def _request(**overrides) -> AssessmentRequest:
    data = {
        "vendor_name": "Asteria AI Systems",
        "use_case": "Enterprise Generative AI platform",
        "user_count": 2000,
        "data_classification": "confidential",
    }
    return AssessmentRequest(**(data | overrides))


def _finding(status: str = "SUPPORTED", citations: list[Evidence] | None = None, **kw) -> Finding:
    return Finding(
        domain=kw.get("domain", "security"),
        control_id=kw.get("control_id", "SEC-04"),
        title="Encryption at rest",
        status=status,
        severity=kw.get("severity", "medium"),
        claim="Asteria encrypts customer data at rest with AES-256.",
        citations=[EVIDENCE] if citations is None else citations,
    )


def test_request_derives_vendor_id_and_rejects_unknown_fields():
    assert _request().vendor_id == "asteria-ai-systems"
    with pytest.raises(ValidationError):
        _request(budget=5)  # extra="forbid"
    with pytest.raises(ValidationError):
        _request(data_classification="secret")  # not in the vocabulary


def test_missing_evidence_is_a_status_and_uncited_claims_are_flagged():
    assert _finding("MISSING", citations=[]).is_evidence_backed  # missing needs no vendor citation
    assert not _finding("SUPPORTED", citations=[]).is_evidence_backed  # a claim with no source
    with pytest.raises(ValidationError):
        _finding("PASS")  # there is no PASS -- only evidence statuses


def test_domain_report_rejects_findings_from_another_domain():
    with pytest.raises(ValidationError):
        DomainReport(domain="legal", risk_rating="low", summary="ok", findings=[_finding(domain="security")])


def test_assessment_from_draft_exposes_gaps_and_keeps_system_fields_out_of_the_draft():
    draft = AssessmentDraft(
        recommendation="CONDITIONAL_APPROVAL",
        risk_rating="high",
        domains=[
            DomainReport(
                domain="security",
                risk_rating="high",
                summary="Encryption fine, residency unclear.",
                findings=[_finding(), _finding("MISSING", citations=[], control_id="SEC-07", severity="high")],
            )
        ],
        executive_summary="Conditional approval pending data-residency evidence.",
    )
    assert "human_approval" not in AssessmentDraft.model_fields  # the LLM cannot set it

    assessment = Assessment.from_draft(draft, _request())
    assert assessment.human_approval == "not_required"  # the gate sets it, not the model
    assert [f.control_id for f in assessment.evidence_gaps] == ["SEC-07"]
    assert assessment.cited_chunk_ids == {EVIDENCE.chunk_id}
    assert assessment.domains_covered == {"security"}


def test_tool_result_envelope_round_trip():
    hit = SearchHit(
        chunk_id=EVIDENCE.chunk_id,
        doc_id="information-security-policy",
        source=EVIDENCE.source,
        doc_type="policy",
        domain="security",
        page=3,
        text="<untrusted_document>All confidential data must be encrypted at rest.</untrusted_document>",
    )
    result = ToolResult.model_validate(ToolResult.ok([hit]).model_dump())
    assert result.status == "ok"
    evidence = result.hits()[0].to_evidence()
    assert evidence.quote == "All confidential data must be encrypted at rest."  # tags stripped

    down = ToolResult.fail("unavailable", "vector store timeout")
    assert down.status == "unavailable" and down.results == []


def test_max_severity():
    assert max_severity(["low", "critical", "high"]) == "critical"
    assert max_severity([]) == "low"
