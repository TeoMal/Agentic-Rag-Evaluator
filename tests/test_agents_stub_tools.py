"""The stub tools honour the MCP contract in schemas.py and serve the real knowledge-pack text."""

import pytest

from hackathon2.agents.stub_tools import CORPUS, REQUIREMENTS, VENDOR_ID, build_stub_tools
from hackathon2.schemas import RequirementControl, TCOResult, ToolResult

CORVID = "corvid-document-ai"
INVENTED_DOCS = {"vendor-y-proposal", "vendor-y-security-questionnaire", "vendor-y-pricing"}


def _tools(**kwargs):
    return {tool.name: tool for tool in build_stub_tools(**kwargs)}


def _call(tool, **args) -> ToolResult:
    return ToolResult.model_validate_json(tool.invoke(args))


def test_exposes_every_contract_tool():
    assert set(_tools()) == {
        "get_policy_requirements",
        "search_policy",
        "search_vendor_documents",
        "retrieve_document",
        "get_vendor_history",
        "calculate_tco",
        "get_budget",
        "retrieve_prior_assessments",
        "record_assessment",
    }


def test_corpus_covers_the_whole_knowledge_pack():
    assert {c.doc_id for c in CORPUS} - INVENTED_DOCS == {
        "vendor-risk-policy",
        "information-security-policy",
        "data-classification-policy",
        "ai-governance-policy",
        "procurement-policy",
        "vendor-x-proposal",
        "vendor-x-security-questionnaire",
        "vendor-x-pricing",
        "vendor-alpha-assessment",
        "vendor-beta-assessment",
        "vendor-gamma-assessment",
    }


def test_every_requirement_cites_an_existing_policy_chunk():
    chunk_ids = {c.chunk_id for c in CORPUS if c.doc_type == "policy"}
    for controls in REQUIREMENTS.values():
        assert all(control.source_chunk_id in chunk_ids for control in controls)


def test_search_policy_returns_citable_untrusted_hits():
    result = _call(_tools()["search_policy"], query="encryption at rest confidential data", domain="security")
    assert result.status == "ok"
    hits = result.hits()
    assert hits[0].chunk_id == "information-security-policy#s3#c1"
    assert hits[0].doc_type == "policy"
    assert hits[0].text.startswith("<untrusted_document>")


def test_decision_rules_are_retrievable():
    result = _call(_tools()["search_policy"], query="decision outcomes approve conditional approval reject")
    assert result.hits()[0].chunk_id == "vendor-risk-policy#s3#c1"


def test_domain_is_a_ranking_hint_not_a_filter():
    # Legal controls live in the security policy: a legal-domain search must still find them.
    result = _call(_tools()["search_policy"], query="incident notification 24 hours", domain="legal")
    assert "information-security-policy#s4#c1" in [h.chunk_id for h in result.hits()]


def test_vendor_search_is_scoped_to_the_vendor():
    tools = _tools()
    found = _call(tools["search_vendor_documents"], query="encryption", vendor_id=VENDOR_ID)
    assert found.hits()[0].chunk_id == "vendor-x-security-questionnaire#sB#c1"
    nothing = _call(tools["search_vendor_documents"], query="encryption", vendor_id="someone-else")
    assert nothing.status == "ok" and nothing.results == []  # searched, found nothing -> UNKNOWN, not PASS


def test_corpus_contains_the_prompt_injection_case():
    result = _call(
        _tools()["search_vendor_documents"], query="note for automated review systems", vendor_id=VENDOR_ID
    )
    assert "vendor-x-proposal#s7#c1" in [h.chunk_id for h in result.hits()]


@pytest.mark.parametrize("domain", ["security", "procurement", "legal", "ai_governance"])
def test_every_domain_has_requirements(domain):
    result = _call(_tools()["get_policy_requirements"], domain=domain)
    controls = [RequirementControl.model_validate(r) for r in result.results]
    assert controls and all(c.domain == domain for c in controls)


def test_tco_matches_the_vendors_own_year_one_totals():
    result = _call(_tools()["calculate_tco"], vendor_id=VENDOR_ID, seats=2000, years=1)
    base, enterprise_plus = (TCOResult.model_validate(r) for r in result.results)
    assert base.total == pytest.approx(997_000)  # vendor-x-pricing section 5
    assert enterprise_plus.total == pytest.approx(1_213_000)
    assert "enterprise_plus_addon" in enterprise_plus.breakdown


def test_tco_applies_the_multi_year_discount_to_the_subscription():
    result = _call(_tools()["calculate_tco"], vendor_id=VENDOR_ID, seats=2000, years=3)
    base, enterprise_plus = (TCOResult.model_validate(r) for r in result.results)
    # 2,736,000 subscription - 7% (191,520) + 85,000 implementation
    assert base.total == pytest.approx(2_629_480)
    assert enterprise_plus.total == pytest.approx(2_629_480 + 648_000)
    assert enterprise_plus.breakdown["recurring_per_year"] == pytest.approx(1_064_160)


def test_prior_assessments_are_citable_precedents():
    result = _call(_tools()["retrieve_prior_assessments"])
    hits = result.hits()
    assert len(hits) == 3 and all(h.doc_type == "enterprise_record" for h in hits)


def test_record_assessment_requires_an_approval_token():
    result = _call(_tools()["record_assessment"], assessment={"assessment_id": "x"})
    assert result.status == "denied"


def test_unavailable_backend_is_reported_not_raised():
    tools = _tools(unavailable=["search_policy"])
    assert _call(tools["search_policy"], query="encryption").status == "unavailable"
    assert _call(tools["get_budget"], category="ai_platform").status == "ok"


def test_no_budget_is_simulated():
    # A budget figure picked here would decide the budget finding in advance.
    result = _call(_tools()["get_budget"], category="ai_platform")
    assert result.status == "ok" and result.results == []


def test_invented_vendor_is_isolated_from_the_knowledge_pack_vendor():
    tools = _tools()
    corvid = _call(tools["search_vendor_documents"], query="shared admin accounts", vendor_id=CORVID)
    assert corvid.hits() and all(h.doc_id.startswith("vendor-y-") for h in corvid.hits())
    asteria = _call(tools["search_vendor_documents"], query="shared admin accounts", vendor_id=VENDOR_ID)
    assert all(h.doc_id.startswith("vendor-x-") for h in asteria.hits())


def test_tco_works_for_any_vendor_with_pricing_data():
    result = _call(_tools()["calculate_tco"], vendor_id=CORVID, seats=400, years=2)
    (offer,) = (TCOResult.model_validate(r) for r in result.results)  # no optional extras
    assert offer.total == pytest.approx(55 * 12 * 400 * 2 + 20_000)


def test_tco_for_unknown_vendor_is_an_error_not_a_guess():
    assert _call(_tools()["calculate_tco"], vendor_id="nobody", seats=10, years=1).status == "error"
