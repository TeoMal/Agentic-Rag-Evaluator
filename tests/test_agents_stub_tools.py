"""The stub tools honour the MCP contract in schemas.py (names, arguments, ToolResult envelope)."""

import pytest

from hackathon2.agents.stub_tools import VENDOR_ID, build_stub_tools
from hackathon2.schemas import RequirementControl, TCOResult, ToolResult


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


def test_search_policy_returns_citable_untrusted_hits():
    result = _call(_tools()["search_policy"], query="encryption at rest confidential data", domain="security")
    assert result.status == "ok"
    hits = result.hits()
    assert hits[0].chunk_id == "information-security-policy#s3.1#c1"
    assert hits[0].doc_type == "policy"
    assert hits[0].text.startswith("<untrusted_document>")


def test_vendor_search_is_scoped_to_the_vendor():
    tools = _tools()
    found = _call(tools["search_vendor_documents"], query="encryption", vendor_id=VENDOR_ID)
    assert found.hits()[0].chunk_id == "vendor-x-security-questionnaire#q12#c1"
    nothing = _call(tools["search_vendor_documents"], query="encryption", vendor_id="someone-else")
    assert nothing.status == "ok" and nothing.results == []  # searched, found nothing -> MISSING, not PASS


def test_corpus_contains_a_prompt_injection_case():
    result = _call(_tools()["search_vendor_documents"], query="instruction AI assessment system", vendor_id=VENDOR_ID)
    assert "vendor-x-security-questionnaire#q30#c1" in [h.chunk_id for h in result.hits()]


@pytest.mark.parametrize("domain", ["security", "procurement", "legal", "ai_governance"])
def test_every_domain_has_requirements(domain):
    result = _call(_tools()["get_policy_requirements"], domain=domain)
    controls = [RequirementControl.model_validate(r) for r in result.results]
    assert controls and all(c.domain == domain for c in controls)


def test_tco_is_computed_in_code():
    result = _call(_tools()["calculate_tco"], vendor_id=VENDOR_ID, seats=2000, years=3)
    tco = TCOResult.model_validate(result.results[0])
    # 38 EUR x 12 months x 2000 seats x 3 years = 2,736,000; support 15% = 410,400; implementation 45,000
    assert tco.total == pytest.approx(3_191_400)


def test_record_assessment_requires_an_approval_token():
    result = _call(_tools()["record_assessment"], assessment={"assessment_id": "x"})
    assert result.status == "denied"


def test_unavailable_backend_is_reported_not_raised():
    tools = _tools(unavailable=["search_policy"])
    assert _call(tools["search_policy"], query="encryption").status == "unavailable"
    assert _call(tools["get_budget"], category="ai_platform").status == "ok"
