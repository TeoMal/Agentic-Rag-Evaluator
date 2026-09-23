# Architecture

Everything runs locally in Docker (app + Postgres/pgvector). The only external service is the
Azure OpenAI model the agent calls.

## Request flow (target design, from the handout)

```
POST vendor assessment request (FR01)
        |
        v
Orchestrator deep agent ── write_todos plan (FR02) ── virtual FS for working notes
        |  task tool: delegate to specialists (FR07)
        +──────────────+──────────────+──────────────+
        v              v              v              v
   Security       Procurement /    Legal /         AI Governance
   Risk agent     Finance agent    Compliance      agent            (FR08: >= 3 domains)
        |              |              |              |
        +──── RAG over knowledge/ (FR03-05, cite every material claim)
        +──── MCP tools: search_policy, get_vendor_history, calculate_tco, record_assessment (FR06)
        |
        v
Guardrails: injection-in-documents ignored (FR10), tool authorization, unsupported claims
flagged, structured-output validation, fail safe on missing evidence (FR09, FR11, FR15)
        |
        v
Risk synthesis -> APPROVE / CONDITIONAL APPROVAL / REJECT + rating (FR12)
        |
        v
High risk or final approval -> interrupt() for human review (FR13), resumed via checkpointer
        |
        v
Assessment report  ──>  evaluation suite (FR14)
```

## Code ownership (handout section 13)

| Package | Owner | Requirements | Start from course unit |
|---|---|---|---|
| `service.py`, `agents/` | Technical Lead | FR01, FR02, FR12, FR13 | 47 deepagents-project, 41 sandboxing (interrupt_on) |
| `rag/`, structured outputs | Deep Agent / RAG | FR03, FR04, FR05, FR11 | 60 loaders/splitters, 62 pgvector, 63 agentic RAG, 64 MMR/rerank, 09 structured output |
| `mcp_server/` | MCP engineer | FR06 | 56 MCP server, 57 MCP client, 59 subagents + MCP tools |
| `agents/` specialists, `guardrails/` | A2A / Guardrails | FR07, FR09, FR10, FR15 | 45 subagents, 53 HITL guardrails, 55 red-team hardening, 52 PII/limits/fallback |
| `evaluation/`, pipeline | Evaluation engineer | FR14 | 49 LLM-as-judge, 50 eval pipeline, this repo's local pipeline |

## Technology choices

| Concern | Choice | Why |
|---|---|---|
| LLM | Azure OpenAI `AzureChatOpenAI` (`llm.get_chat_model`) | the course key/endpoint; unit 47 / step 9 |
| Embeddings | `AzureOpenAIEmbeddings` (`llm.get_embeddings`) | same resource; class used Cohere |
| Agent runtime | `deepagents` (planning, virtual FS, subagents, skills) | handout: "must be a multi-step Deep Agent" |
| Agent-to-agent | deepagents subagents via the `task` tool | handout FR07 accepts "A2A **or an equivalent**" |
| Vector store | pgvector (`langchain-postgres`) in the compose `db` service | course units 43/62 |
| HITL state | LangGraph Postgres checkpointer (same `db`) | durable interrupts (unit 19) |
| MCP | `mcp` FastMCP server + `langchain-mcp-adapters` client | units 56/57/59 |
| API | FastAPI + uvicorn (`--factory`) | unit 31; `/health` for Docker's health check |
| Tracing | none yet — Langfuse (units 21–22) is the local option if needed | runs entirely on the laptop |
| Packaging | uv, `uv.lock`, `uv sync --locked` everywhere | reproducible builds; units 32/33 |

## Local pipeline

```
uv run scripts/deploy.py
  1 preflight      Docker running (starts Docker Desktop if not), .env complete
  2 lock + tests   uv lock --check, pytest
  3 build          docker compose build app      (tag = git SHA, or SHA-dirty-timestamp)
  4 database       docker compose up --wait db   (pgvector, data kept in a volume)
  5 app            force-recreate the app container, wait for its HEALTHCHECK
  6 verify         GET /health must report the tag this run built
```

`ci.yml` (on GitHub, currently disabled) runs the same checks on pull requests: locked install,
ruff, pytest, image build, boot + `/health` tag check.

Design rules the pipeline follows:

1. **The lockfile is the truth.** Every install is `--locked`; a stale `uv.lock` fails CI and the
   image build instead of silently resolving something different.
2. **Proof, not hope.** Each image carries its tag; every deploy waits until `/health` reports that
   exact tag.
3. **No secrets in images or logs.** `.dockerignore` is an allowlist, so `.env` never enters the
   build context; the deploy log masks the API key.
4. **Local-only exposure.** Both ports (8020 app, 5446 db) bind to 127.0.0.1.
