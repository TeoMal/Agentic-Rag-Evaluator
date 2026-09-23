# agents/ - orchestrator deep agent and specialist subagents

Covers FR02 (plan), FR07 (delegation), FR08 (domains), FR12 (decision), and the agent side of
FR05, FR11, FR13 and FR15. Everything here is built on the shared contracts in `hackathon2/schemas.py`
and runs today without RAG, MCP or guardrails, on stub tools.

## Run it

```bash
uv run python -m hackathon2.agents              # one Asteria assessment, real model, stub tools
uv run pytest tests/test_agents_*.py            # no model, no network
```

```python
from hackathon2.agents.runner import AssessmentRunner

runner = AssessmentRunner()
response = await runner.run(request)                   # AssessmentResponse
if response.status == "awaiting_approval":
    response = await runner.decide(response.assessment_id, human_decision)
```

## Flow

```
AssessmentRequest
  -> orchestrator deep agent (create_deep_agent)
       write_todos plan
       task -> security-risk-agent        --+
       task -> procurement-finance-agent    |  each returns a DomainReport
       task -> legal-compliance-agent       |  (structured output, ToolStrategy)
       task -> ai-governance-agent        --+
       -> FinalDecision (recommendation, risk, conditions, summary -- no findings)
  -> reports read back from the conversation; a domain without a valid report -> MISSING finding
  -> AssessmentDraft = FinalDecision + reports verbatim
  -> decision gate (guardrails.apply_gate, or the provisional one)
  -> AssessmentResponse: completed | awaiting_approval | failed
```

The orchestrator never rewrites findings: code attaches the specialists' reports verbatim, so a finding or
citation cannot be lost or softened during synthesis.

## Files

| File | What |
|---|---|
| `runner.py` | `AssessmentRunner.run()` / `decide()` / `get()` -- the entry point |
| `orchestrator.py` | `build_orchestrator()`, `FinalDecision`, `render_request()` |
| `specialists.py` | the 4 specialists: names, descriptions, tools, `DomainReport` output |
| `prompts.py` | system prompts |
| `tools.py` | tool providers (stub / MCP) and the per-agent tool allowlists |
| `context.py` | per-run log: retrieved chunk_ids, tool failures, tokens; `instrument_tool()` |
| `collect.py` | reads DomainReports back from `task` results; MISSING placeholder |
| `gate_fallback.py` | provisional gate until guardrails ships `apply_gate` |
| `stub_tools.py` | offline tools + an invented mini-corpus (**test data only**) |
| `report.py` | executive report as Markdown |

## Merge points with the rest of the team

| With | What we agreed / need | Where |
|---|---|---|
| MCP | tool names and arguments exactly as in the `schemas.py` docstring; results are a `ToolResult` (JSON string or text content block); server reachable at `AGENT_MCP_URL` over streamable HTTP | `tools.py` |
| MCP | how `record_assessment` approval tokens are issued (placeholder: `human-review:<reviewer>`) | `runner.py::_record` |
| RAG | `chunk_id` values stable within a run; `SearchHit.text` wrapped in `<untrusted_document>` | used by prompts and `context.py` |
| Guardrails | `apply_gate(draft, request, retrieved_chunk_ids, degraded) -> Assessment` exported from `hackathon2.guardrails` -- picked up automatically | `gate_fallback.py` |
| Guardrails | agent middleware goes into `AssessmentRunner(middleware=..., subagent_middleware=...)` | `runner.py` |
| API | endpoints call `run()` / `decide()` / `get()`; results are in memory for now | `service.py` (not ours) |
| Evaluation | `AssessmentResponse` with `RunMetrics` (tools, subagents, chunk_ids, tokens, duration) per run | `runner.py` |

## Settings (AGENT_*)

| Variable | Default | |
|---|---|---|
| `AGENT_TOOL_SOURCE` | `stub` | `mcp` to use the MCP server |
| `AGENT_MCP_URL` | - | required with `mcp` |
| `AGENT_RECURSION_LIMIT` | `250` | orchestrator step budget |
| `AGENT_TEMPERATURE` | `0.0` | empty for reasoning models that reject a temperature |

## Stub corpus

`stub_tools.py` holds invented chunks shaped to hit every evidence status: a supported control
(encryption), non-compliant ones (certification, model training on customer data), a contradiction (EU-only
hosting vs US overflow inference), missing evidence (24h incident notification, audit logging), a budget
overrun, and a prompt injection in a vendor answer (Q30). Replace it with the real knowledge pack by
switching `AGENT_TOOL_SOURCE`; no code changes.
