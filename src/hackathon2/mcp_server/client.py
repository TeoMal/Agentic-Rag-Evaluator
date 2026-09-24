"""Agent-side access to the NFS MCP server -- the only way the rest of the code gets MCP tools.

Usage (one server process for a whole assessment run):

    from hackathon2.mcp_server.client import open_mcp_toolbox

    async with open_mcp_toolbox() as toolbox:
        security_tools = await toolbox.get_mcp_tools("security")   # LangChain tools for a subagent
        ...
        result = await toolbox.call("system", "record_assessment", {...})   # code-only call
        toolbox.calls                                                        # what was called, for metrics
        toolbox.retrieved_chunk_ids   # chunks whose TEXT a tool returned -- the decision gate checks citations
        await toolbox.read_resource("nfs://requirements/security")           # reference material
        await toolbox.get_prompt("specialist_brief", {"domain": "security", "vendor_name": "..."})

What every tool call goes through (the interceptor, in this order):

  1. role check   -- a tool outside the caller's allow-list is refused with status "denied",
                     without ever reaching the server (defence in depth next to list filtering);
  2. the call     -- over the open stdio session, with a read timeout;
  3. failure net  -- a transport failure is retried once (not timeouts); if it still fails it
                     becomes status "unavailable", so the agent records MISSING evidence instead of
                     the run crashing (FR14);
  4. record       -- an entry in `toolbox.calls` (role, tool, status, duration) for the
                     evaluation's tool-correctness and latency metrics, and the ids the result
                     contained (see "Citations" below).

Citations: `toolbox.retrieved_chunk_ids` lists every chunk whose text a tool returned in this
run (search results, retrieve_document). `toolbox.referenced_ids` lists ids a tool returned
without the text (pricing sources, control sources, prior assessments, policy citations).
The decision gate should accept a citation only if its chunk_id is in `citable_ids`; anything
else was not seen by the agent and is downgraded.

Fallback (FR14): if the server subprocess cannot start, open_mcp_toolbox runs the SAME server
inside this process over an in-memory MCP connection. The agent code cannot tell the difference
and every rule (roles, approval tokens, safe_tool) still applies; the toolbox is marked degraded.

Degraded mode: `toolbox.degraded` is True when the fallback was used or any tool answered
"unavailable"; `toolbox.degraded_reasons` says why. Copy both into the final assessment:

    assessment.degraded_mode = toolbox.degraded
    assessment.gate_notes += [f"degraded: {r}" for r in toolbox.degraded_reasons]

Tracing: these tools are ordinary LangChain tools, so the Langfuse CallbackHandler the
agent runs with already records every call (input, output with our status envelope,
duration) as a tool span. Nothing extra is needed here.

Roles: the four specialists and the orchestrator are LLM roles. "agents" is what the agent
runner loads (agents/tools.py): every LLM role's tools on one session -- it then hands each agent
only that agent's own allow-list. "system" is for our own code only (the human-approval step);
it is the only role that may call record_assessment, so no LLM is ever handed that tool.
"""

import asyncio
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
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent

from hackathon2.schemas import ToolResult, ToolStatus

logger = logging.getLogger("hackathon2.mcp_client")

SERVER_NAME = "nfs-enterprise"  # must match server.SERVER_NAME (a test checks it)
DEFAULT_TIMEOUT_SECONDS = 30
TRANSPORT_RETRIES = 1
RETRY_BACKOFF_SECONDS = 0.5

Role = Literal["orchestrator", "security", "procurement", "legal", "ai_governance", "agents", "system"]

_EVIDENCE_TOOLS = frozenset(
    {"get_policy_requirements", "search_policy", "search_vendor_documents", "retrieve_document"}
)

TOOLS_BY_ROLE: dict[Role, frozenset[str]] = {
    "orchestrator": frozenset({"get_policy_requirements", "retrieve_prior_assessments"}),
    "security": _EVIDENCE_TOOLS | {"get_vendor_history"},
    "procurement": _EVIDENCE_TOOLS | {"calculate_tco", "get_approval_requirements", "get_vendor_history"},
    "legal": _EVIDENCE_TOOLS | {"retrieve_prior_assessments"},
    "ai_governance": _EVIDENCE_TOOLS | {"retrieve_prior_assessments"},
    "system": frozenset({"record_assessment", "retrieve_prior_assessments"}),
}

RESTRICTED_TOOLS = frozenset({"record_assessment"})
LLM_ROLES: tuple[Role, ...] = ("orchestrator", "security", "procurement", "legal", "ai_governance")
TOOLS_BY_ROLE["agents"] = frozenset().union(*(TOOLS_BY_ROLE[role] for role in LLM_ROLES))


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


def _envelope_of(result: Any) -> tuple[ToolStatus, str | None]:
    """(status, error) of our envelope inside a raw MCP result."""
    if getattr(result, "isError", False):
        return "error", "rejected by the MCP server (invalid arguments)"
    try:
        payload = json.loads(result.content[0].text)
        return payload["status"], payload.get("error")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return "error", "unreadable tool result"


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()


class McpToolbox:
    """Tools of one open server session, handed out per role."""

    def __init__(
        self,
        session: ClientSession,
        mode: Literal["stdio", "in_process"] = "stdio",
        degraded_reasons: list[str] | None = None,
    ) -> None:
        self._session = session
        self.mode = mode
        self.calls: list[ToolCall] = []
        self.degraded_reasons: list[str] = list(degraded_reasons or [])
        self.retrieved_chunk_ids: list[str] = []
        self.referenced_ids: list[str] = []

    @property
    def citable_ids(self) -> set[str]:
        """Every id a citation may legitimately point to in this run."""
        return set(self.retrieved_chunk_ids) | set(self.referenced_ids)

    def _note_ids(self, result: Any) -> None:
        """Remember the ids in a successful tool result (for the citation check)."""
        try:
            rows = json.loads(result.content[0].text).get("results", [])
        except (AttributeError, IndexError, TypeError, ValueError):
            return
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            if row.get("chunk_id") and "text" in row and row["chunk_id"] not in self.retrieved_chunk_ids:
                self.retrieved_chunk_ids.append(row["chunk_id"])
            refs = [row.get(k) for k in ("source_chunk_id", "assessment_id", "source_document")]
            refs += row.get("citations", []) if isinstance(row.get("citations"), list) else []
            for ref in refs:
                if isinstance(ref, str) and ref and ref not in self.referenced_ids:
                    self.referenced_ids.append(ref)

    @property
    def degraded(self) -> bool:
        """True if the fallback server is in use or any evidence source was unavailable."""
        return bool(self.degraded_reasons)

    def _note_degraded(self, reason: str) -> None:
        if reason not in self.degraded_reasons:
            self.degraded_reasons.append(reason)

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
                for attempt in range(TRANSPORT_RETRIES + 1):
                    try:
                        result = await handler(request)
                        break
                    except Exception as exc:  # noqa: BLE001 -- any transport failure must become "unavailable"
                        if attempt < TRANSPORT_RETRIES and not _is_timeout(exc):
                            logger.warning("MCP call %s failed (%r), retrying once", request.name, exc)
                            await asyncio.sleep(RETRY_BACKOFF_SECONDS)
                            continue
                        logger.warning("MCP call %s failed: %r", request.name, exc)
                        result = _as_call_result(
                            ToolResult.fail(
                                "unavailable", f"{request.name}: MCP server unavailable ({type(exc).__name__})"
                            )
                        )
                        break

            status, error = _envelope_of(result)
            if status == "ok":
                self._note_ids(result)
            if status == "unavailable":
                self._note_degraded(error or f"{request.name}: unavailable")
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            self.calls.append(ToolCall(role, request.name, status, duration_ms, dict(request.args)))
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

    async def read_resource(self, uri: str) -> ToolResult:
        """Read an MCP resource, e.g. "nfs://requirements/security" or "nfs://documents/<doc_id>".
        Returns the same envelope as the tools; failures come back as "unavailable"/"error"."""
        return await _read_resource(self._session, uri)

    async def get_prompt(self, name: str, arguments: dict[str, Any]) -> str:
        """Render a server prompt, e.g. get_prompt("specialist_brief", {"domain": "security",
        "vendor_name": "Asteria AI Systems"}), as plain text for a system prompt."""
        return await _get_prompt(self._session, name, arguments)


async def _read_resource(session: ClientSession, uri: str) -> ToolResult:
    try:
        response = await session.read_resource(uri)
        return ToolResult.model_validate_json(response.contents[0].text)
    except Exception as exc:  # noqa: BLE001 -- unknown URI or transport failure: report, do not crash
        return ToolResult.fail("unavailable", f"resource {uri}: {type(exc).__name__}: {exc}")


async def _get_prompt(session: ClientSession, name: str, arguments: dict[str, Any]) -> str:
    response = await session.get_prompt(name, {k: str(v) for k, v in arguments.items()})
    return "\n\n".join(m.content.text for m in response.messages if getattr(m.content, "text", None))


@asynccontextmanager
async def _in_process_session(timeout_seconds: float) -> AsyncIterator[ClientSession]:
    """The same FastMCP server, run inside this process over an in-memory MCP connection."""
    from hackathon2.mcp_server.server import mcp as local_server  # imported only when needed

    async with create_connected_server_and_client_session(
        local_server, read_timeout_seconds=timedelta(seconds=timeout_seconds)
    ) as session:
        yield session


@asynccontextmanager
async def open_mcp_toolbox(
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    connection: dict | None = None,
    fallback: bool = True,
) -> AsyncIterator[McpToolbox]:
    """Start the MCP server (one subprocess) and keep one session open for the whole block.

    If the subprocess cannot start and `fallback` is True, the same server runs in-process
    instead and the toolbox is marked degraded. Raises McpUnavailableError only if both fail
    (or the subprocess fails and fallback=False). Errors raised inside the block propagate unchanged.
    """
    client = MultiServerMCPClient({SERVER_NAME: connection or server_connection(timeout_seconds)})
    started = False
    body_error: BaseException | None = None
    try:
        async with client.session(SERVER_NAME) as session:
            started = True
            try:
                yield McpToolbox(session)
            except BaseException as exc:
                body_error = exc
                raise
            return
    except Exception as exc:
        if body_error is not None:
            raise body_error from None  # unwrap the SDK's ExceptionGroup: callers get their own error
        if started:
            raise
        if not fallback:
            raise McpUnavailableError(f"could not start MCP server '{SERVER_NAME}': {exc!r}") from exc
        subprocess_error = exc

    reason = f"MCP server subprocess could not start ({type(subprocess_error).__name__}); using in-process fallback"
    logger.warning("%s: %r", reason, subprocess_error)
    started = False
    try:
        async with _in_process_session(timeout_seconds) as session:
            started = True
            try:
                yield McpToolbox(session, mode="in_process", degraded_reasons=[reason])
            except BaseException as exc:
                body_error = exc
                raise
    except Exception as exc:
        if body_error is not None:
            raise body_error from None
        if started:
            raise
        raise McpUnavailableError(
            f"MCP server unavailable: subprocess failed ({subprocess_error!r}) and in-process fallback failed ({exc!r})"
        ) from exc
