"""Agent-side access to the NFS MCP server -- the only way the rest of the code gets MCP tools.

Usage (one server process for a whole assessment run):

    from hackathon2.mcp_server.client import open_mcp_toolbox

    async with open_mcp_toolbox() as toolbox:
        security_tools = await toolbox.get_mcp_tools("security")   # LangChain tools for a subagent
        ...
        result = await toolbox.call("system", "record_assessment", {...})   # code-only call
        toolbox.calls                                                        # what was called, for metrics

What every tool call goes through (the interceptor, in this order):

  1. role check   -- a tool outside the caller's allow-list is refused with status "denied",
                     without ever reaching the server (defence in depth next to list filtering);
  2. the call     -- over the open stdio session, with a read timeout;
  3. failure net  -- a transport failure (server crashed, timeout) becomes status "unavailable",
                     so the agent records MISSING evidence instead of the run crashing (FR15);
  4. record       -- an entry in `toolbox.calls` (role, tool, status, duration) for the
                     evaluation's tool-correctness and latency metrics.

Tracing: these tools are ordinary LangChain tools, so the Langfuse CallbackHandler the
agent runs with already records every call (input, output with our status envelope,
duration) as a tool span. Nothing extra is needed here.

Roles: the four specialists and the orchestrator are LLM roles. "system" is for our own code
only (the human-approval endpoint); it is the only role that may call record_assessment, so no
LLM is ever handed that tool.
"""

import json
import logging
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent

from hackathon2.schemas import ToolResult, ToolStatus

logger = logging.getLogger("hackathon2.mcp_client")

SERVER_NAME = "nfs-enterprise"  # must match server.SERVER_NAME (a test checks it)
DEFAULT_TIMEOUT_SECONDS = 30

Role = Literal["orchestrator", "security", "procurement", "legal", "ai_governance", "system"]

_EVIDENCE_TOOLS = frozenset(
    {"get_policy_requirements", "search_policy", "search_vendor_documents", "retrieve_document"}
)

TOOLS_BY_ROLE: dict[Role, frozenset[str]] = {
    "orchestrator": frozenset({"get_policy_requirements", "retrieve_prior_assessments"}),
    "security": _EVIDENCE_TOOLS | {"get_vendor_history"},
    "procurement": _EVIDENCE_TOOLS | {"calculate_tco", "get_budget", "get_vendor_history"},
    "legal": _EVIDENCE_TOOLS | {"retrieve_prior_assessments"},
    "ai_governance": _EVIDENCE_TOOLS | {"retrieve_prior_assessments"},
    "system": frozenset({"record_assessment", "retrieve_prior_assessments"}),
}

RESTRICTED_TOOLS = frozenset({"record_assessment"})
LLM_ROLES: tuple[Role, ...] = ("orchestrator", "security", "procurement", "legal", "ai_governance")


class McpUnavailableError(RuntimeError):
    """The MCP server could not be started or initialised."""


@dataclass
class ToolCall:
    """One recorded tool call -- feeds RunMetrics.tools_called and the evaluation suite."""

    role: Role
    tool: str
    status: ToolStatus
    duration_ms: float
    args: dict[str, Any] = field(default_factory=dict)


def server_connection(timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """How to launch the server: the same Python, as a module, with OUR environment.
    (The MCP SDK otherwise passes a minimal environment -- no .env values, no PYTHONPATH.)"""
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "hackathon2.mcp_server"],
        "env": dict(os.environ),
        "session_kwargs": {"read_timeout_seconds": timedelta(seconds=timeout_seconds)},
    }


def _as_call_result(result: ToolResult) -> CallToolResult:
    """A ToolResult in the shape the server itself would have sent."""
    return CallToolResult(content=[TextContent(type="text", text=result.model_dump_json())], isError=False)


def _status_of(result: Any) -> ToolStatus:
    """Read our envelope's status from a raw MCP result."""
    if getattr(result, "isError", False):
        return "error"  # rejected by FastMCP before our code ran (e.g. wrong argument type)
    try:
        return json.loads(result.content[0].text)["status"]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return "error"


class McpToolbox:
    """Tools of one open server session, handed out per role."""

    def __init__(self, session: ClientSession) -> None:
        self._session = session
        self.calls: list[ToolCall] = []

    def interceptor(self, role: Role):
        """The wrapper around every tool call made on behalf of `role` (see module docstring)."""
        allowed = TOOLS_BY_ROLE[role]

        async def intercept(request: MCPToolCallRequest, handler) -> Any:
            started = time.perf_counter()
            if request.name not in allowed:
                logger.warning("role %s tried to call %s -- denied", role, request.name)
                result = _as_call_result(
                    ToolResult.fail("denied", f"role '{role}' is not authorised to call {request.name}")
                )
            else:
                try:
                    result = await handler(request)
                except Exception as exc:  # noqa: BLE001 -- any transport failure must become "unavailable"
                    logger.warning("MCP call %s failed: %r", request.name, exc)
                    result = _as_call_result(
                        ToolResult.fail("unavailable", f"{request.name}: MCP server unavailable ({type(exc).__name__})")
                    )

            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            self.calls.append(ToolCall(role, request.name, _status_of(result), duration_ms, dict(request.args)))
            return result

        return intercept

    async def get_mcp_tools(self, role: Role) -> list[BaseTool]:
        """LangChain tools for one role -- exactly its allow-list, each call guarded.
        Raises if the server no longer offers a tool the role needs (a broken contract
        should fail loudly at start-up, not halfway through an assessment)."""
        if role not in TOOLS_BY_ROLE:
            raise ValueError(f"unknown role '{role}', expected one of {sorted(TOOLS_BY_ROLE)}")
        tools = await load_mcp_tools(self._session, server_name=SERVER_NAME, tool_interceptors=[self.interceptor(role)])
        offered = {t.name for t in tools}
        missing = TOOLS_BY_ROLE[role] - offered
        if missing:
            raise RuntimeError(f"MCP server does not offer {sorted(missing)} needed by role '{role}'")
        return [t for t in tools if t.name in TOOLS_BY_ROLE[role]]

    async def call(self, role: Role, tool: str, args: dict[str, Any]) -> ToolResult:
        """Call a tool from code (no LLM involved), through the same guard. Returns the envelope."""

        async def handler(request: MCPToolCallRequest) -> CallToolResult:
            return await self._session.call_tool(request.name, request.args)

        request = MCPToolCallRequest(name=tool, args=args, server_name=SERVER_NAME)
        raw = await self.interceptor(role)(request, handler)
        if raw.isError:
            text = raw.content[0].text if raw.content else "tool error"
            return ToolResult.fail("error", text)
        return ToolResult.model_validate_json(raw.content[0].text)


@asynccontextmanager
async def open_mcp_toolbox(
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS, connection: dict | None = None
) -> AsyncIterator[McpToolbox]:
    """Start the MCP server (one subprocess) and keep one session open for the whole block.

    Raises McpUnavailableError if the server cannot start -- the caller decides what to do
    (step 5 adds an in-process fallback). Errors raised inside the block propagate unchanged.
    """
    client = MultiServerMCPClient({SERVER_NAME: connection or server_connection(timeout_seconds)})
    entered = False
    try:
        async with client.session(SERVER_NAME) as session:
            entered = True
            yield McpToolbox(session)
    except Exception as exc:
        if entered:
            raise
        raise McpUnavailableError(f"could not start MCP server '{SERVER_NAME}': {exc!r}") from exc