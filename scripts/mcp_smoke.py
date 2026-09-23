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
    "get_budget",
    "retrieve_prior_assessments",
    "record_assessment",
}

VENDOR = "asteria-ai-systems"

# The knowledge search tools need hackathon2.rag.retriever (and an indexed corpus).
# Until RAG is ready they must answer "unavailable" -- never crash, never invent data.
RAG = {"ok", "unavailable"}

# (tool, arguments, accepted statuses)
CALLS = [
    ("get_policy_requirements", {"domain": "security"}, {"ok"}),
    ("search_policy", {"query": "encryption at rest", "domain": "security"}, RAG),
    ("search_vendor_documents", {"query": "data residency hosting", "vendor_id": VENDOR}, RAG),
    ("search_vendor_documents", {"query": "data residency", "vendor_id": "unknown-vendor"}, {"ok"}),
    ("retrieve_document", {"chunk_id": "does-not-exist"}, {"error", "unavailable"}),
    ("get_vendor_history", {"vendor_id": VENDOR}, {"ok"}),
    ("calculate_tco", {"vendor_id": VENDOR, "seats": 2000, "years": 3}, {"ok"}),
    ("calculate_tco", {"vendor_id": "unknown-vendor", "seats": 10, "years": 1}, {"unavailable"}),
    ("get_budget", {"category": "generative-ai-platform"}, {"ok"}),
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

    print("all good" if not failures else f"{failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
