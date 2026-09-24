"""Where the agents' tools come from, and which agent may use which tool.

A ToolsProvider is an async callable returning the full toolset for one run. Today it is the stub
toolset; after the merge it is the MCP server. Nothing else in the agents package changes.

Tool access is an allowlist per agent. `record_assessment` is never given to any LLM: only code
calls it, after human review (handout section 9: restrict sensitive MCP tools).
"""

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from hackathon2.agents.config import AgentSettings
from hackathon2.agents.stub_tools import build_stub_tools
from hackathon2.schemas import Domain

logger = logging.getLogger(__name__)

ToolsProvider = Callable[[], Awaitable[list[BaseTool]]]

KNOWLEDGE_TOOLS = ("get_policy_requirements", "search_policy", "search_vendor_documents", "retrieve_document")

# The orchestrator reads the decision rules and precedents itself; domain evidence is the specialists' job.
ORCHESTRATOR_TOOLS = ("search_policy", "retrieve_document", "retrieve_prior_assessments", "get_vendor_history")

SPECIALIST_TOOLS: dict[Domain, tuple[str, ...]] = {
    "security": (*KNOWLEDGE_TOOLS, "get_vendor_history"),
    "procurement": (
        *KNOWLEDGE_TOOLS,
        "calculate_tco",
        "get_budget",
        "get_vendor_history",
        "retrieve_prior_assessments",
    ),
    "legal": KNOWLEDGE_TOOLS,
    "ai_governance": KNOWLEDGE_TOOLS,
}

# Never handed to an LLM, whatever the allowlists above say.
RESTRICTED_TOOLS = frozenset({"record_assessment"})


def pick_tools(tools: Mapping[str, BaseTool], names: Iterable[str]) -> list[BaseTool]:
    """The allowed subset of `tools`, in allowlist order. Unknown names are skipped with a warning,
    so a tool renamed or not yet built on the MCP side degrades the run instead of breaking it."""
    picked: list[BaseTool] = []
    for name in names:
        if name in RESTRICTED_TOOLS:
            raise ValueError(f"'{name}' is restricted and must not be given to an agent")
        tool = tools.get(name)
        if tool is None:
            logger.warning("tool '%s' is not provided by the current tool source; skipping it", name)
            continue
        picked.append(tool)
    return picked


def stub_provider(unavailable: Iterable[str] = (), *, fail_all: bool = False) -> ToolsProvider:
    down = tuple(unavailable)

    async def _load() -> list[BaseTool]:
        return build_stub_tools(down, fail_all=fail_all)

    return _load


def mcp_provider(url: str) -> ToolsProvider:
    """Tools from the MCP server. If the server cannot be reached, every tool reports "unavailable":
    the run degrades (all controls MISSING) instead of crashing -- and never falls back to fake data."""

    async def _load() -> list[BaseTool]:
        try:
            client = MultiServerMCPClient({"nfs": {"transport": "streamable_http", "url": url}})
            return list(await client.get_tools())
        except Exception as exc:  # noqa: BLE001 -- degrade to "unavailable" tools, never crash
            logger.error("MCP server at %s is unreachable (%s: %s); all tools will report unavailable",
                         url, type(exc).__name__, exc)
            return build_stub_tools(fail_all=True)

    return _load


def default_provider(settings: AgentSettings) -> ToolsProvider:
    if settings.tool_source == "mcp":
        if not settings.mcp_url:
            raise ValueError("AGENT_TOOL_SOURCE=mcp needs AGENT_MCP_URL")
        return mcp_provider(settings.mcp_url)
    logger.warning("agents are using STUB tools with invented data (AGENT_TOOL_SOURCE=stub)")
    return stub_provider()
