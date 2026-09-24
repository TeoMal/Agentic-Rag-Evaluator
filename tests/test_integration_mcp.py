"""Agents <-> MCP server <-> RAG, wired as in a real run: the MCP server is started as a stdio
subprocess (mcp_provider) and searches the real knowledge pack. Keyword retrieval (no embedding
model), so no network is needed.
"""

from pathlib import Path

import pytest

from hackathon2.agents.context import content_to_text, parse_tool_result
from hackathon2.agents.recording import record
from hackathon2.agents.tools import ORCHESTRATOR_TOOLS, RESTRICTED_TOOLS, SPECIALIST_TOOLS, mcp_provider
from hackathon2.schemas import Assessment, AssessmentDraft, AssessmentRequest, DomainReport, Finding, ToolResult

KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge"
SECRET = "x" * 64


@pytest.fixture(autouse=True)
def server_env(monkeypatch, tmp_path):
    """The subprocess inherits this environment: keyword retrieval, a temp register, a signing key.
    (Empty values override .env, so a developer's embedding resource or database is never used.)"""
    for name in ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "AZURE_EMBEDDING_ENDPOINT", "AZURE_EMBEDDING_API_KEY"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("POSTGRES_HOST", "")
    monkeypatch.setenv("KNOWLEDGE_DIR", str(KNOWLEDGE))
    monkeypatch.setenv("MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_APPROVAL_SECRET", SECRET)


async def _call(tool, **args) -> ToolResult:
    return parse_tool_result(content_to_text(await tool.ainvoke(args)))


async def test_agents_get_real_evidence_from_the_mcp_server():
    async with mcp_provider()() as loaded:
        tools = {t.name: t for t in loaded}
        allowed = set(ORCHESTRATOR_TOOLS).union(*SPECIALIST_TOOLS.values())
        assert allowed <= set(tools), "every tool an agent is allowed must be served"
        assert not RESTRICTED_TOOLS & set(tools), "record_assessment must never reach an agent"

        policy = await _call(tools["search_policy"], query="encryption in transit TLS", domain="security")
        assert policy.status == "ok"
        assert policy.results[0]["chunk_id"].startswith("information-security-policy#")
        assert policy.results[0]["text"].startswith("<untrusted_document>")

        vendor = await _call(
            tools["search_vendor_documents"], query="automated review systems", vendor_id="Asteria AI Systems"
        )
        flagged = [h for h in vendor.results if h["suspicious"]]
        assert flagged and flagged[0]["doc_id"] == "vendor-x-proposal"  # the planted injection is marked

        section = await _call(tools["retrieve_document"], chunk_id=policy.results[0]["chunk_id"])
        assert section.status == "ok" and section.results[0]["chunk_id"] == policy.results[0]["chunk_id"]

        rules = await _call(tools["get_approval_requirements"], annual_value=500_000, data_classification="internal")
        assert rules.status == "ok" and rules.results[0]["competitive_sourcing_required"]


async def test_an_unreachable_mcp_server_degrades_instead_of_crashing(monkeypatch):
    from hackathon2.mcp_server import client

    def broken(*args, **kwargs):
        raise client.McpUnavailableError("server could not start")

    monkeypatch.setattr(client, "open_mcp_toolbox", broken)
    async with mcp_provider()() as loaded:
        result = await _call({t.name: t for t in loaded}["search_policy"], query="encryption")
    assert result.status == "unavailable"


def _approved(**changes) -> Assessment:
    request = AssessmentRequest(vendor_name="Test Vendor", use_case="Pilot", user_count=5, data_classification="public")
    finding = Finding(
        domain="security",
        control_id="SEC-01",
        title="Encryption",
        status="MISSING",
        severity="high",
        claim="No evidence.",
    )
    draft = AssessmentDraft(
        recommendation="REJECT",
        risk_rating="high",
        executive_summary="Rejected.",
        domains=[DomainReport(domain="security", risk_rating="high", summary="Gap.", findings=[finding])],
    )
    return Assessment.from_draft(draft, request).model_copy(update={"human_approval": "approved", **changes})


async def test_an_approved_assessment_is_recorded_once(tmp_path):
    assessment = _approved(reviewer="cro@northstar")
    async with mcp_provider()("system") as loaded:  # agents never get record_assessment; code does, as "system"
        tools = {t.name: t for t in loaded}
        assert await record(assessment, tools, principal="cro@northstar") is None
        assert "already recorded" in await record(assessment, tools, principal="cro@northstar")
    assert assessment.assessment_id in (tmp_path / "assessments.jsonl").read_text(encoding="utf-8")


async def test_recording_is_refused_without_approval_or_signing_key(monkeypatch):
    from hackathon2.agents.stub_tools import build_stub_tools

    tools = {t.name: t for t in build_stub_tools()}
    assert "pending" in await record(_approved(human_approval="pending"), tools, principal="x")
    monkeypatch.setenv("MCP_APPROVAL_SECRET", "too-short")
    assert "MCP_APPROVAL_SECRET" in await record(_approved(), tools, principal="x")
