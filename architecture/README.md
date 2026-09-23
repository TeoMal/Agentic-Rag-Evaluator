# Architecture

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
Assessment report  ──>  Application Insights traces  ──>  evaluation suite (FR14)
```

## Code ownership (handout section 13)

| Package | Owner | Requirements | Start from course unit |
|---|---|---|---|
| `service.py`, `agents/` | Technical Lead | FR01, FR02, FR12, FR13 | 47 deepagents-project, 41 sandboxing (interrupt_on) |
| `rag/`, structured outputs | Deep Agent / RAG | FR03, FR04, FR05, FR11 | 60 loaders/splitters, 62 pgvector, 63 agentic RAG, 64 MMR/rerank, 09 structured output |
| `mcp_server/` | MCP engineer | FR06 | 56 MCP server, 57 MCP client, 59 subagents + MCP tools |
| `agents/` specialists, `guardrails/` | A2A / Guardrails | FR07, FR09, FR10, FR15 | 45 subagents, 53 HITL guardrails, 55 red-team hardening, 52 PII/limits/fallback |
| `evaluation/`, `telemetry.py`, pipeline | Evaluation / Azure | FR14, section 11 | 49 LLM-as-judge, 50 eval pipeline, 38 monitoring, this repo's DevOps |

## Technology choices

| Concern | Choice | Why |
|---|---|---|
| LLM | Azure OpenAI `AzureChatOpenAI` (`llm.get_chat_model`) | handout requires Azure; course unit 47 / step 9 |
| Embeddings | `AzureOpenAIEmbeddings` (`llm.get_embeddings`) | same resource; class used Cohere |
| Agent runtime | `deepagents` (planning, virtual FS, subagents, skills) | handout: "must be a multi-step Deep Agent" |
| Agent-to-agent | deepagents subagents via the `task` tool | handout FR07 accepts "A2A **or an equivalent**"; swap in `a2a-sdk` later if wanted |
| Vector store | pgvector (`langchain-postgres`) locally | course units 43/62; on Azure the index is built in memory at start-up (no managed DB, to keep cost at zero) |
| HITL state | LangGraph checkpointer: Postgres locally, in-memory on Azure | durable interrupts (unit 19); Azure runs 1 replica so memory is consistent |
| MCP | `mcp` FastMCP server + `langchain-mcp-adapters` client | units 56/57/59 |
| API | FastAPI + uvicorn (`--factory`) | unit 31; health endpoint for probes |
| Observability | `azure-monitor-opentelemetry` -> Application Insights | handout section 11; replaces Langfuse from units 21/22 |
| Packaging | uv, `uv.lock`, `uv sync --locked` everywhere | reproducible builds; units 32/33 |

## DevOps pipeline

```
 developer laptop                         GitHub                                    Azure
 ────────────────                         ──────                                    ─────
 uv run scripts/deploy.py                 PR -> ci.yml                              rg (assigned group)
   preflight (Docker, .env)                 uv sync --locked, ruff, pytest            ├ ACR  acr<unique>
   uv lock --check + pytest                 docker build (GHA cache)                  ├ Log Analytics
   docker compose build (tag = git SHA)     run image, /health == SHA                 ├ App Insights  <── OTel from app
   db up --wait, app force-recreate                                                   ├ Container Apps env
   /health == this tag  ✓                 merge -> cd.yml                             └ Container App (1 replica)
                                            ci.yml (reused)                                ▲
 uv run scripts/deploy.py azure  ───────>   deploy.py azure --tag SHA  ──────────────────────┘
   (same script, locally or in CD)            Bicep infra -> build amd64 -> push -> Bicep app -> /health == SHA
```

Design rules the pipeline follows:

1. **One implementation.** `scripts/deploy.py` is used by developers and by `cd.yml` alike.
2. **The lockfile is the truth.** Every install is `--locked`; a stale `uv.lock` fails CI and the
   image build instead of silently resolving something different.
3. **Proof, not hope.** Each image carries its tag; every deploy (local, CI smoke test, Azure) waits
   until `/health` reports that exact tag.
4. **No secrets in images, logs or command lines.** `.dockerignore` is an allowlist; the API key
   reaches Azure through a temp ARM parameters file and a Container Apps secret; logs mask it.
5. **Safe on shared subscriptions.** Contributor-only permissions suffice (no role assignments);
   every resource is tagged `project=hackathon2` and teardown deletes only those.
