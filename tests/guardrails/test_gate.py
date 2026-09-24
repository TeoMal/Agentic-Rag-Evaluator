from dataclasses import replace

import pytest

from hackathon2.guardrails import Limits, Reason, gate_assessment
from hackathon2.schemas import Assessment, AssessmentDraft, RequirementControl


def test_valid_low_risk_rejection_is_allowed_without_mutation(draft, gate_context):
    before = draft.model_dump()
    ledger = {key: hit.model_dump() for key, hit in gate_context.ledger.items()}
    assert gate_assessment(draft, context=gate_context).outcome == "allow"
    assert draft.model_dump() == before
    assert {key: hit.model_dump() for key, hit in gate_context.ledger.items()} == ledger


@pytest.mark.parametrize("recommendation", ["APPROVE", "CONDITIONAL_APPROVAL"])
def test_approval_recommendations_always_require_human_review(draft, gate_context, recommendation):
    draft.recommendation = recommendation
    result = gate_assessment(draft, context=gate_context)
    assert result.outcome == "require_review" and Reason.FINAL_APPROVAL in result.reasons


@pytest.mark.parametrize("level", ["assessment", "domain", "finding"])
@pytest.mark.parametrize("severity", ["high", "critical"])
def test_high_risk_cannot_hide_in_lower_levels(draft, gate_context, level, severity):
    if level == "assessment":
        draft.risk_rating = severity
    elif level == "domain":
        draft.domains[0].risk_rating = severity
    else:
        draft.domains[0].findings[0].severity = severity
    assert gate_assessment(draft, context=gate_context).reasons == (Reason.HIGH_RISK,)


@pytest.mark.parametrize(
    "field, value",
    [
        ("source", "forged.pdf"),
        ("page", 99),
        ("section", "forged"),
        ("doc_type", "policy"),
        ("quote", "This is not in the original text."),
        ("quote", " "),
    ],
)
def test_forged_citation_metadata_or_quote_denies(draft, gate_context, field, value):
    setattr(draft.domains[0].findings[0].citations[0], field, value)
    result = gate_assessment(draft, context=gate_context)
    assert result.outcome == "deny"
    assert Reason.CITATION_MISMATCH in result.reasons


def test_unknown_and_quarantined_citations(draft, gate_context, hit):
    assert gate_assessment(draft, context=replace(gate_context, ledger={})).reasons == (Reason.UNKNOWN_CITATION,)
    for context in (
        replace(gate_context, quarantined_chunk_ids=frozenset({hit.chunk_id})),
        replace(gate_context, ledger={hit.chunk_id: hit.model_copy(update={"suspicious": True})}),
    ):
        assert gate_assessment(draft, context=context).reasons == (Reason.QUARANTINED_CITATION,)


@pytest.mark.parametrize("status", ["SUPPORTED", "NON_COMPLIANT", "CONTRADICTED"])
def test_shared_citation_required_statuses(draft, gate_context, status):
    finding = draft.domains[0].findings[0]
    finding.status = status
    finding.citations = []
    assert Reason.CITATION_REQUIRED in gate_assessment(draft, context=gate_context).reasons
    assert finding.status == status and finding.citations == []


@pytest.mark.parametrize("status", ["MISSING", "INFERRED", "CONTRADICTED"])
def test_evidence_uncertainty_requires_review(draft, gate_context, status):
    draft.domains[0].findings[0].status = status
    result = gate_assessment(draft, context=gate_context)
    assert result.outcome == "require_review" and Reason.EVIDENCE_GAP in result.reasons


def test_omitted_mandatory_controls_and_empty_output_are_not_success(draft, gate_context):
    required = RequirementControl(id="SEC-02", domain="security", control="Residency", source_chunk_id="policy#c2")
    assert gate_assessment(
        draft,
        context=replace(gate_context, required_controls=(*gate_context.required_controls, required)),
    ).reasons == (Reason.EVIDENCE_GAP,)
    draft.domains = []
    assert gate_assessment(draft, context=gate_context).outcome == "require_review"


@pytest.mark.parametrize("status", ["unavailable", "denied", "error"])
def test_tool_failure_blocks_automatic_completion(draft, gate_context, status):
    assert gate_assessment(
        draft,
        context=replace(gate_context, tool_statuses=("ok", status)),
    ).reasons == (Reason.TOOL_FAILURE,)


@pytest.mark.parametrize(
    "flag, reason", [("scanner_failed", Reason.SCANNER_FAILURE), ("verifier_failed", Reason.VERIFIER_FAILURE)]
)
def test_failed_safety_checks_deny(draft, gate_context, flag, reason):
    result = gate_assessment(draft, context=replace(gate_context, **{flag: True}))
    assert result.outcome == "deny" and result.reasons == (reason,)


def test_model_controlled_approval_cannot_bypass_gate(draft, gate_context):
    assessment = Assessment(
        **draft.model_dump(),
        vendor_name="Fixture",
        vendor_id="vendor-1",
        human_approval="approved",
    )
    assessment.recommendation = "APPROVE"
    assert gate_assessment(assessment, context=gate_context).outcome == "require_review"
    assessment.recommendation = "REJECT"
    assessment.degraded_mode = True
    assert Reason.DEGRADED_MODE in gate_assessment(assessment, context=gate_context).reasons
    assessment.human_approval = "rejected"
    assert gate_assessment(assessment, context=gate_context).outcome == "deny"


@pytest.mark.parametrize("value", [None, {}, "approved", AssessmentDraft.model_construct(recommendation="APPROVE")])
def test_malformed_output_denies(value, gate_context):
    assert gate_assessment(value, context=gate_context).reasons == (Reason.INVALID_OUTPUT,)


def test_mutated_models_are_revalidated(draft, gate_context):
    draft.domains[0].findings[0].status = "PASS"
    assert gate_assessment(draft, context=gate_context).reasons == (Reason.INVALID_OUTPUT,)


@pytest.mark.parametrize(
    "change",
    [
        {"ledger_run_id": "other-run"},
        {"run_id": ""},
        {"tool_statuses": ("unknown",)},
        {"scanner_failed": "false"},
        {"required_controls": (), "ledger": {"bad": None}},
    ],
)
def test_malformed_or_cross_run_provenance_denies(draft, gate_context, change):
    assert gate_assessment(draft, context=replace(gate_context, **change)).reasons == (Reason.INVALID_PROVENANCE,)


def test_ledger_key_mismatch_and_duplicate_requirements_deny(draft, gate_context, hit):
    assert gate_assessment(
        draft,
        context=replace(gate_context, ledger={"wrong-key": hit}),
    ).reasons == (Reason.INVALID_PROVENANCE,)
    assert gate_assessment(
        draft,
        context=replace(gate_context, required_controls=gate_context.required_controls * 2),
    ).reasons == (Reason.INVALID_PROVENANCE,)


def test_duplicate_findings_deny(draft, gate_context):
    draft.domains[0].findings.append(draft.domains[0].findings[0].model_copy(deep=True))
    assert gate_assessment(draft, context=gate_context).reasons == (Reason.INVALID_OUTPUT,)


def test_bounded_output_and_sensitive_quotes(draft, gate_context):
    assert gate_assessment(draft, context=gate_context, limits=Limits(max_nodes=5)).reasons == (Reason.LIMIT_EXCEEDED,)
    quote = "SYNTHETIC_SECRET_fixture"
    draft.domains[0].findings[0].citations[0].quote = quote
    assert gate_assessment(draft, context=gate_context).reasons == (Reason.SENSITIVE_DATA,)
    assert draft.domains[0].findings[0].citations[0].quote == quote


def test_escaped_text_is_not_substituted_for_an_exact_quote(draft, gate_context, hit):
    hit.text = "Costs < 100 & annual."
    citation = draft.domains[0].findings[0].citations[0]
    citation.quote = hit.text
    assert gate_assessment(draft, context=gate_context).outcome == "allow"
    citation.quote = "Costs &lt; 100 &amp; annual."
    assert gate_assessment(draft, context=gate_context).reasons == (Reason.CITATION_MISMATCH,)


def _explicit_noncompliance(draft, gate_context):
    """Synthetic evidence; the tests do not claim semantic grounding by the gate."""
    result = draft.model_copy(deep=True)
    finding = result.domains[0].findings[0]
    hit = next(iter(gate_context.ledger.values())).model_copy(
        update={"text": "Customer data is not encrypted at rest."}
    )
    finding.status = "NON_COMPLIANT"
    finding.claim = hit.text
    finding.citations = [hit.to_evidence()]
    return result, replace(gate_context, ledger={hit.chunk_id: hit})


def test_approve_with_explicit_mandatory_noncompliance_is_invalid_not_rewritten(draft, gate_context):
    draft, context = _explicit_noncompliance(draft, gate_context)
    draft.recommendation = "APPROVE"
    before = draft.model_dump()
    ledger_before = {key: hit.model_dump() for key, hit in context.ledger.items()}
    result = gate_assessment(draft, context=context)
    assert result.outcome == "deny"
    assert Reason.INCONSISTENT_ASSESSMENT in result.reasons
    assert draft.model_dump() == before and draft.recommendation == "APPROVE"
    assert {key: hit.model_dump() for key, hit in context.ledger.items()} == ledger_before


def test_supported_approval_is_consistent_but_still_requires_review(draft, gate_context):
    draft.recommendation = "APPROVE"
    before = draft.model_dump()
    result = gate_assessment(draft, context=gate_context)
    assert result.outcome == "require_review"
    assert result.reasons == (Reason.FINAL_APPROVAL,)
    assert draft.model_dump() == before


@pytest.mark.parametrize("recommendation, expected", [("CONDITIONAL_APPROVAL", "require_review"), ("REJECT", "allow")])
def test_noncompliance_does_not_imply_blanket_rejection_of_other_recommendations(
    draft,
    gate_context,
    recommendation,
    expected,
):
    draft, context = _explicit_noncompliance(draft, gate_context)
    draft.recommendation = recommendation
    result = gate_assessment(draft, context=context)
    assert result.outcome == expected
    assert Reason.INCONSISTENT_ASSESSMENT not in result.reasons
    if recommendation == "CONDITIONAL_APPROVAL":
        assert Reason.FINAL_APPROVAL in result.reasons


@pytest.mark.parametrize("change", [{"mandatory": False}, {"domain": "legal"}, {"id": "SEC-other"}])
def test_consistency_rule_requires_reviewed_mandatory_domain_and_id_match(draft, gate_context, change):
    draft, context = _explicit_noncompliance(draft, gate_context)
    draft.recommendation = "APPROVE"
    controls = tuple(control.model_copy(update=change) for control in context.required_controls)
    result = gate_assessment(draft, context=replace(context, required_controls=controls))
    assert result.outcome == "require_review"
    assert Reason.INCONSISTENT_ASSESSMENT not in result.reasons
    assert Reason.FINAL_APPROVAL in result.reasons
    if change != {"mandatory": False}:
        assert Reason.EVIDENCE_GAP in result.reasons


@pytest.mark.parametrize("domain, control_id", [("legal", "LEGAL-9"), ("procurement", "COST-X")])
def test_consistency_check_uses_supplied_controls_without_domain_or_id_constants(
    draft, gate_context, domain, control_id
):
    draft, context = _explicit_noncompliance(draft, gate_context)
    draft.recommendation = "APPROVE"
    draft.domains[0].domain = domain
    finding = draft.domains[0].findings[0]
    finding.domain = domain
    finding.control_id = control_id
    controls = tuple(c.model_copy(update={"domain": domain, "id": control_id}) for c in context.required_controls)
    assert (
        Reason.INCONSISTENT_ASSESSMENT
        in gate_assessment(
            draft,
            context=replace(context, required_controls=controls),
        ).reasons
    )


@pytest.mark.parametrize("status", ["MISSING", "INFERRED", "CONTRADICTED"])
def test_approval_with_uncertain_mandatory_evidence_still_requires_review(draft, gate_context, status):
    draft.recommendation = "APPROVE"
    finding = draft.domains[0].findings[0]
    finding.status = status
    if status != "CONTRADICTED":
        finding.citations = []
    before = draft.model_dump()
    result = gate_assessment(draft, context=gate_context)
    assert result.outcome == "require_review"
    assert {Reason.EVIDENCE_GAP, Reason.FINAL_APPROVAL} <= set(result.reasons)
    assert Reason.INCONSISTENT_ASSESSMENT not in result.reasons
    assert draft.model_dump() == before


def test_llm_approval_field_cannot_override_assessment_inconsistency(draft, gate_context):
    draft, context = _explicit_noncompliance(draft, gate_context)
    draft.recommendation = "APPROVE"
    assessment = Assessment(**draft.model_dump(), vendor_name="Fixture", vendor_id="fixture", human_approval="approved")
    result = gate_assessment(assessment, context=context)
    assert result.outcome == "deny" and Reason.INCONSISTENT_ASSESSMENT in result.reasons
