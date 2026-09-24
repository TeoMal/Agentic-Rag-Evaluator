"""Start the MCP server exactly as the agent will (stdio subprocess) and call every tool once.

    uv run python scripts/mcp_smoke.py

Prints each tool's status and result count. Exits non-zero if a tool is missing or
an unexpected status comes back -- handy before opening a pull request.
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "get_policy_requirements",
    "search_policy",
    "search_vendor_documents",
    "retrieve_document",
    "get_vendor_history",
    "calculate_tco",
    "get_approval_requirements",
    "retrieve_prior_assessments",
    "record_assessment",
}

VENDOR = "asteria-ai-systems"

EXPECTED_RESOURCES = {"nfs://documents", "nfs://requirements", "nfs://vendors", "nfs://procurement-rules"}
EXPECTED_TEMPLATES = {"nfs://documents/{doc_id}", "nfs://requirements/{domain}"}
EXPECTED_PROMPTS = {"specialist_brief", "assessment_plan"}

# (uri, expected status). Documents are read from the PDFs in knowledge/ (run from the repo root).
RESOURCE_READS = [
    ("nfs://documents", "ok"),
    ("nfs://documents/vendor-x-proposal", "ok"),
    ("nfs://documents/not-a-document", "error"),
    ("nfs://requirements", "ok"),
    ("nfs://requirements/legal", "ok"),
    ("nfs://vendors", "ok"),
    ("nfs://procurement-rules", "ok"),
]

# The knowledge search tools need hackathon2.rag.retriever (and an indexed corpus).
# Until RAG is ready they must answer "unavailable" -- never crash, never invent data.
RAG = {"ok", "unavailable"}

# (tool, arguments, accepted statuses)
CALLS = [
    ("get_policy_requirements", {"domain": "security"}, {"ok"}),
    ("search_policy", {"query": "encryption at rest", "domain": "security"}, RAG),
    ("search_vendor_documents", {"query": "data retention", "vendor_id": VENDOR}, RAG),
    ("search_vendor_documents", {"query": "data retention", "vendor_id": "Asteria AI Systems"}, RAG),
    ("search_vendor_documents", {"query": "data retention", "vendor_id": "unknown-vendor"}, {"error"}),
    ("retrieve_document", {"chunk_id": "does-not-exist"}, {"error", "unavailable"}),
    ("get_vendor_history", {"vendor_id": VENDOR}, {"ok"}),
    ("get_vendor_history", {"vendor_id": "Vendor Beta"}, {"ok"}),
    ("calculate_tco", {"vendor_id": VENDOR, "seats": 2000, "years": 3, "add_ons": ["enterprise_plus"]}, {"ok"}),
    ("calculate_tco", {"vendor_id": VENDOR, "seats": 2000, "years": 1, "add_ons": ["gold"]}, {"error"}),
    ("calculate_tco", {"vendor_id": "unknown-vendor", "seats": 10, "years": 1}, {"error"}),
    (
        "calculate_tco",
        {"vendor_id": "hidden-vendor", "seats": 500, "years": 2, "per_user_monthly": 20, "one_time_fees": 10000},
        {"ok"},
    ),
    ("get_approval_requirements", {"annual_value": 1213000, "data_classification": "confidential"}, {"ok"}),
    ("retrieve_prior_assessments", {}, {"ok"}),
    ("record_assessment", {"assessment": {}, "approval_token": ""}, {"error"}),
]


async def main() -> int:
    # env: the SDK otherwise passes only a minimal environment (no .env values, no PYTHONPATH).
    params = StdioServerParameters(command=sys.executable, args=["-m", "hackathon2.mcp_server"], env=dict(os.environ))
    failures = 0
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = {t.name for t in (await session.list_tools()).tools}
        missing = EXPECTED_TOOLS - names
        print(f"tools: {len(names)} listed" + (f", MISSING: {sorted(missing)}" if missing else ""))
        failures += len(missing)

        for tool, args, expected in CALLS:
            response = await session.call_tool(tool, args)
            payload = json.loads(response.content[0].text)
            ok = payload["status"] in expected
            failures += not ok
            detail = (payload["error"] or f"{len(payload['results'])} result(s)")[:90]
            print(f"  {'PASS' if ok else 'FAIL'}  {tool:28} status={payload['status']:<11} {detail}")

        # --- resources and prompts ---
        uris = {str(r.uri) for r in (await session.list_resources()).resources}
        templates = {t.uriTemplate for t in (await session.list_resource_templates()).resourceTemplates}
        prompts = {p.name for p in (await session.list_prompts()).prompts}
        missing = (EXPECTED_RESOURCES - uris) | (EXPECTED_TEMPLATES - templates) | (EXPECTED_PROMPTS - prompts)
        print(
            f"resources: {len(uris)} + {len(templates)} templates, prompts: {len(prompts)}"
            + (f", MISSING: {sorted(missing)}" if missing else "")
        )
        failures += len(missing)
        for uri, expected in RESOURCE_READS:
            payload = json.loads((await session.read_resource(uri)).contents[0].text)
            ok = payload["status"] == expected
            failures += not ok
            detail = (payload["error"] or f"{len(payload['results'])} result(s)")[:90]
            print(f"  {'PASS' if ok else 'FAIL'}  {uri:40} status={payload['status']:<11} {detail}")
        brief = await session.get_prompt("specialist_brief", {"domain": "security", "vendor_name": "Test Vendor"})
        ok = "SEC-01" in brief.messages[0].content.text
        failures += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  prompt specialist_brief lists the security controls")

    print("all good" if not failures else f"{failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
