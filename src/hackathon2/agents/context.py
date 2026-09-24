"""Per-run bookkeeping: what the agents retrieved, which tools failed, what it cost.

Every tool handed to an agent is wrapped by `instrument_tool`, which:
  - logs the call (RunMetrics.tools_called);
  - records every chunk a tool returned, with its text -- the decision gate checks citations against this log;
  - turns an exception inside a tool into a ToolResult "unavailable" instead of crashing the run (FR14);
  - notes non-ok results, so the assessment can be marked degraded_mode.

The wrapper works the same for the stub tools and for tools loaded from the MCP server.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, get_args

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ValidationError

from hackathon2.schemas import AssessmentRequest, RunMetrics, SearchHit, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

_TOOL_STATUSES = frozenset(get_args(ToolStatus))


@dataclass
class RunContext:
    request: AssessmentRequest
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    retrieved_chunk_ids: set[str] = field(default_factory=set)
    retrieved_hits: dict[str, SearchHit] = field(default_factory=dict)  # chunk_id -> hit, fullest text seen
    tool_statuses: list[ToolStatus] = field(default_factory=list)  # every tool result, in call order
    tools_called: list[str] = field(default_factory=list)
    tool_failures: list[str] = field(default_factory=list)  # "unavailable" / "error" results
    denied_calls: list[str] = field(default_factory=list)  # "denied" results (authorization)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def degraded(self) -> bool:
        """A tool or source was unavailable (schemas: Assessment.degraded_mode). A call rejected as
        invalid ("error") is recorded in tool_failures but is the agent's mistake, not a lost source."""
        return "unavailable" in self.tool_statuses

    def record_tool_result(self, tool_name: str, result: ToolResult) -> None:
        self.tool_statuses.append(result.status)
        if result.status == "ok":
            for item in result.results:
                if isinstance(item, dict) and isinstance(item.get("chunk_id"), str):
                    self.retrieved_chunk_ids.add(item["chunk_id"])
                    self._remember_hit(item)
        elif result.status == "denied":
            self.denied_calls.append(f"{tool_name}: {result.error}")
        else:
            self.tool_failures.append(f"{tool_name}: {result.status} - {result.error}")

    def _remember_hit(self, item: dict) -> None:
        """Keep the chunk's text for citation checks. A chunk seen again with more text (the whole
        section, from retrieve_document) replaces the shorter version: it contains it."""
        if "text" not in item:
            return
        try:
            hit = SearchHit.model_validate(item)
        except ValidationError:
            return
        known = self.retrieved_hits.get(hit.chunk_id)
        if known is None or len(hit.text) > len(known.text):
            self.retrieved_hits[hit.chunk_id] = hit

    def metrics(self, duration_seconds: float, subagents_called: list[str]) -> RunMetrics:
        return RunMetrics(
            duration_seconds=round(duration_seconds, 2),
            llm_calls=self.llm_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            tools_called=list(self.tools_called),
            subagents_called=subagents_called,
            retrieved_chunk_ids=sorted(self.retrieved_chunk_ids),
        )


# --------------------------------------------------------------------------------------
# Tool output parsing (stub tools return JSON strings; MCP tools may return content blocks)
# --------------------------------------------------------------------------------------


def content_to_text(content: Any) -> str:
    """Flatten whatever a tool or message returned into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, tuple) and content:  # (content, artifact)
        return content_to_text(content[0])
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict):
                parts.append(json.dumps(block, default=str))
            elif hasattr(block, "text"):  # MCP TextContent objects
                parts.append(str(block.text))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if isinstance(content, BaseModel):
        return content.model_dump_json()
    if isinstance(content, dict):
        return json.dumps(content, default=str)
    return str(content)


def parse_tool_result(text: str) -> ToolResult | None:
    """The ToolResult envelope inside a tool's output, or None if the output is something else."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("status") not in _TOOL_STATUSES:
        return None
    try:
        return ToolResult.model_validate(data)
    except ValidationError:
        return None


# --------------------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------------------


def instrument_tool(tool: BaseTool, ctx: RunContext) -> BaseTool:
    """Wrap a tool so its calls are logged in `ctx` and its failures never propagate."""

    async def _run(**kwargs: Any) -> str:
        ctx.tools_called.append(tool.name)
        try:
            raw = await tool.ainvoke(kwargs)
        except Exception as exc:  # noqa: BLE001 -- FR14: a failing tool must not crash the assessment
            logger.warning("tool %s failed: %s: %s", tool.name, type(exc).__name__, exc)
            result = ToolResult.fail(
                "unavailable",
                f"{tool.name} failed ({type(exc).__name__}). Treat the controls that depend on it as MISSING.",
            )
            ctx.record_tool_result(tool.name, result)
            return result.model_dump_json()

        text = content_to_text(raw)
        result = parse_tool_result(text)
        if result is not None:
            ctx.record_tool_result(tool.name, result)
        return text

    return StructuredTool.from_function(
        coroutine=_run,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


class UsageCallback(BaseCallbackHandler):
    """Counts LLM calls and tokens for RunMetrics (latency / cost, handout section 10)."""

    run_inline = True  # mutate ctx on the calling thread, in order

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.ctx.llm_calls += 1
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if usage:
                    self.ctx.input_tokens += usage.get("input_tokens", 0) or 0
                    self.ctx.output_tokens += usage.get("output_tokens", 0) or 0
