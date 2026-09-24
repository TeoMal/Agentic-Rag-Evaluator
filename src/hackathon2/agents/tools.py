"""Where the agents' tools come from, and which agent may use which tool.

A ToolsProvider opens a toolset as an async context manager -- the tools work until it exits:

    async with provider() as tools: ...            # the agents' tools for one run
    async with provider("system") as tools: ...    # for our own code: adds record_assessment

mcp_provider (the default) starts the NFS MCP server for the run; stub_provider is the offline
test double. Nothing else in the agents package depends on which one is used.

Tool access is an allowlist per agent. `record_assessment` is never given to any LLM: only code
calls it, after human review (handout section 9: restrict sensitive MCP tools).
"""

import logging
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Literal

from langchain_core.tools import BaseTool

from hackathon2.agents.config import AgentSettings
from hackathon2.agents.stub_tools import build_stub_tools
from hackathon2.schemas import Domain

logger = logging.getLogger(__name__)

Caller = Literal["agents", "system"]
ToolsProvider = Callable[[Caller], AbstractAsyncContextManager[list[BaseTool]]]

KNOWLEDGE_TOOLS = ("get_policy_requirements", "search_policy", "search_vendor_documents", "retrieve_document")

# The orchestrator reads the decision rules and precedents itself; domain evidence is the specialists' job.
ORCHESTRATOR_TOOLS = ("search_policy", "retrieve_document", "retrieve_prior_assessments", "get_vendor_history")

SPECIALIST_TOOLS: dict[Domain, tuple[str, ...]] = {
    "security": (*KNOWLEDGE_TOOLS, "get_vendor_history"),
    "procurement": (
        *KNOWLEDGE_TOOLS,
        "calculate_tco",
        "get_approval_requirements",
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

    @asynccontextmanager
    async def _open(caller: Caller = "agents") -> AsyncIterator[list[BaseTool]]:
        yield build_stub_tools(down, fail_all=fail_all)  # pick_tools keeps record_assessment from agents

    return _open


def mcp_provider(timeout_seconds: float = 30) -> ToolsProvider:
    """Tools from the NFS MCP server: one server subprocess (stdio) and one session per run
    (mcp_server.client.open_mcp_toolbox, which runs the same server in-process if the subprocess
    cannot start). If neither works, every tool reports "unavailable": the run degrades (all
    controls MISSING) instead of crashing -- and never falls back to fake data."""

    @asynccontextmanager
    async def _open(caller: Caller = "agents") -> AsyncIterator[list[BaseTool]]:
        from hackathon2.mcp_server.client import open_mcp_toolbox

        async with AsyncExitStack() as stack:
            try:
                toolbox = await stack.enter_async_context(open_mcp_toolbox(timeout_seconds))
                tools = await toolbox.get_mcp_tools(caller)  # the "agents" role never includes record_assessment
                for reason in toolbox.degraded_reasons:
                    logger.warning("MCP: %s", reason)
            except Exception as exc:  # noqa: BLE001 -- degrade to "unavailable" tools, never crash
                logger.error(
                    "MCP server unavailable (%s: %s); all tools will report unavailable", type(exc).__name__, exc
                )
                tools = build_stub_tools(fail_all=True)
            yield tools

    return _open


def default_provider(settings: AgentSettings) -> ToolsProvider:
    if settings.tool_source == "stub":
        logger.warning("agents are using STUB tools with invented data (AGENT_TOOL_SOURCE=stub)")
        return stub_provider()
    return mcp_provider()
