# agents/ - orchestrator deep agent and specialist subagents

Covers FR02 (plan), FR07 (risk domains), FR11 (decision) and the agent side of FR05, FR10, FR12 and
FR14, plus the specialist agents of handout section 8. Everything here is built on the shared contracts in
`hackathon2/schemas.py` and runs today without RAG, MCP or guardrails, on stub tools that serve the real
knowledge-pack text.

## Run it

```bash
uv run python -m hackathon2.agents              # one Asteria assessment, real model, stub tools
uv run python -m hackathon2.agents corvid       # the invented second vendor (generalisation check)
uv run pytest tests -k agents                   # no model, no network
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
       search_policy / retrieve_document     NFS decision rules (VR-006, mandatory rules such as IS-010 s8)
       retrieve_prior_assessments            precedents (vendor alpha / beta / gamma)
       phase 1: task -> security-risk-agent     --+
                task -> legal-compliance-agent    |  each returns a DomainReport
                task -> ai-governance-agent     --+  (structured output, ToolStrategy)
       phase 2: task -> procurement-finance-agent   with the phase-1 conditions that cost money,
                                                    so the COMPLIANT configuration is priced
       reconcile the reports
       -> FinalDecision (recommendation, risk, cited decision_basis, conditions, summary -- no findings)
  -> reports read back from the conversation; a domain without a valid report -> MISSING (UNKNOWN) finding
  -> AssessmentDraft = FinalDecision + reports verbatim + "Decision basis" line
  -> decision gate (guardrails.apply_gate, or the provisional one)
  -> AssessmentResponse: completed | awaiting_approval | failed
```

The orchestrator never rewrites findings: code attaches the specialists' reports verbatim, so a finding or
citation cannot be lost or softened during synthesis.

## No hard-coded answers

The handout forbids hard-coding expected answers, and a hidden vendor is assessed on the day. So:

- Prompts state general assessment principles only: no vendor names, no figures, phrases or examples from the
  knowledge pack. `tests/test_agents_no_leaks.py` fails if one appears in a system prompt, a subagent
  description, the FinalDecision schema or a tool description, or if agent code names a vendor.
- Decision rules, the rating scale and precedents are retrieved at run time and cited in `decision_basis`, so
  the recommendation is traceable (PR-001 section 6).
- The stub simulates no budget: a number picked by us would decide the budget finding in advance.
- `python -m hackathon2.agents corvid` assesses an invented second vendor with different problems, using the
  same prompts. If it needs a prompt change to work, the prompts were fitted to Asteria.

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
| `stub_tools.py` | offline tools serving the knowledge-pack text, one chunk per section |
| `report.py` | executive report as Markdown (MISSING shown as UNKNOWN, NFS vocabulary) |

## Merge points with the rest of the team

| With | What we agreed / need | Where |
|---|---|---|
| MCP | tool names and arguments exactly as in the `schemas.py` docstring; results are a `ToolResult` (JSON string or text content block); server reachable at `AGENT_MCP_URL` over streamable HTTP | `tools.py` |
| MCP | how `record_assessment` approval tokens are issued (placeholder: `human-review:<reviewer>`) | `runner.py::_record` |
| RAG | `chunk_id` values stable within a run; `SearchHit.text` wrapped in `<untrusted_document>` | used by prompts and `context.py` |
| RAG / MCP | vendor documents are named `vendor-x-*`, not after the vendor: ingestion must tag them with the vendor_id (`asteria-ai-systems`) for `search_vendor_documents(vendor_id=...)` to find them -- and the same for the hidden vendor | ingestion |
| MCP | `calculate_tco` returns one result per offered configuration (base, then tiers/add-ons named in `breakdown`) | `stub_tools.py::_calculate_tco` |
| MCP | `retrieve_prior_assessments()` without vendor_id returns the historical assessments as citable chunks | `stub_tools.py` |
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

## Stub corpus (temporary)

`stub_tools.py` serves the text of the supplied knowledge pack (5 policies, 3 vendor documents, 3 historical
assessments), one chunk per numbered section, with chunk ids like `information-security-policy#s6#c1` or
`vendor-x-security-questionnaire#sE#c1`. Search is keyword overlap; RAG replaces it.

It also contains an INVENTED vendor, Corvid Document AI (`vendor-y-*`, vendor_id `corvid-document-ai`), which
is not part of the knowledge pack. Its documents differ on purpose: a shared admin account, a 48-hour incident
notice, no SOC 2, a claim that data never leaves the EU next to US-hosted inference, and an embedded
instruction to the assessor. It exists only to check that the agents generalise.

Simulated, because the knowledge pack does not contain them: the requirements checklist behind
`get_policy_requirements` (extracted from the policies only, every control citing its policy chunk), the
vendor history and the pricing tables behind `calculate_tco` (transcribed from each vendor's pricing
document; it reproduces Asteria's own year-one totals). No budget record exists, so budget fit must come out
as MISSING. All of this belongs to the MCP server after the merge.

Switch to the real MCP server with `AGENT_TOOL_SOURCE=mcp`; no code changes.
