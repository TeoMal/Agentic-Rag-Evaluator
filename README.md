# Hackathon 2 — Vendor Risk & Procurement Deep Agent

An evidence-grounded Deep Agent that assesses **Asteria AI Systems** for **Northstar Financial
Services** and recommends **APPROVE / CONDITIONAL APPROVAL / REJECT**, built from Deep Agents,
RAG, MCP, specialist agents, guardrails and an automated evaluation suite. It runs **locally** in
Docker; the only cloud dependency is the Azure OpenAI model it calls.

> Status: the **uv project and the local build/deploy pipeline are in place** (test, containerise,
> run, verify). The agent itself is built next, in the packages under `src/hackathon2/` — see
> [architecture/](architecture/README.md) for the plan and which course unit each piece starts from.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) and Docker Desktop.

```bash
uv sync                      # create .venv from uv.lock (Python 3.13 is fetched if missing)
uv run scripts/deploy.py     # test -> rebuild image -> recreate container -> verify, in one pass
```

The first run creates `.env` from `.env.example` and asks once for `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_ENDPOINT`. Then open **http://127.0.0.1:8020/ui** to request an assessment, read the
report and approve or reject it (API docs: http://127.0.0.1:8020/docs).

| Endpoint | |
|---|---|
| `GET /ui` | web page: request form → assessment report → human review |
| `POST /assessments` | run an assessment (FR01 → FR11); a failed run returns `status: failed` (FR14) |
| `GET /assessments/{id}` | an assessment's current state |
| `POST /assessments/{id}/decision` | approve / reject an assessment awaiting review (FR12) |
| `GET /` · `GET /health` | liveness · image tag and subsystem checks |

Assessments are kept in memory: restarting the app forgets them.

## Observability — Langfuse

Handout §11 asks for Azure Application Insights; this project uses **Langfuse** instead, run locally
from `docker-compose.langfuse.yml` (its own Compose project, so `down --volumes` never deletes traces).

```bash
uv run scripts/deploy.py langfuse        # start it (first run pulls ~2 GB) -> http://localhost:3020
uv run scripts/deploy.py                 # the app picks up the LANGFUSE_* keys from .env
```

Login `admin@hackathon2.local` / `hackathon2-langfuse` (local demo values; the project and its API keys
are created on first start and match `.env.example`). Each assessment is **one trace**
(`vendor-assessment`):

| §11 asks for | In the trace |
|---|---|
| agent workflow executions | the root span: request in, outcome out; ERROR level on a failed run |
| LLM calls, latency, tokens | a generation per model call (orchestrator and specialists), with tokens and cost |
| MCP/tool calls, specialist interactions, duration | a tool observation per MCP call; each specialist runs under the `task` tool |
| exceptions, failed paths | `run_status`, `gate_blocked`, `degraded` scores; the error as status message |
| evaluation results | `evaluation.live` scores the run's own trace (`assessment.*`, `grounding.*`); other suites get an `evaluation:<suite>` trace |
| human review | `awaiting_human_review`, then `human_decision` when the reviewer decides |

`/health` reports `tracing: langfuse` when the keys are set. Tracing is optional and fail-safe: blank
keys or `LANGFUSE_TRACING_ENABLED=false` turn it off, and an unreachable Langfuse never affects a run.
Code: `src/hackathon2/observability.py`. For Langfuse Cloud, see the comments in `.env.example`.

PowerShell / bash shortcuts do the same thing: `.\scripts\deploy.ps1` · `bash scripts/deploy.sh`.

## The deploy script — one pass, every time

| Command | What it does |
|---|---|
| `uv run scripts/deploy.py` | preflight (starts Docker Desktop if needed, fills `.env`) → `uv lock --check` + pytest → rebuild image → start Postgres/pgvector → force-recreate the app → wait until `/health` reports **this run's image tag** |
| `uv run scripts/deploy.py status` | containers + live `/health` |
| `uv run scripts/deploy.py down [--volumes]` | stop the local stack (`--volumes` also wipes the DB) |
| `uv run scripts/deploy.py langfuse` · `langfuse-down [--volumes]` | start / stop local Langfuse (`--volumes` also wipes the traces) |

Flags: `--dry-run` (print every mutating command, run only read-only checks) · `--skip-tests` ·
`--no-cache` (rebuild all layers, re-pull bases) · `--tag TAG` · `--yes` · `--timeout SECONDS`.

**Why the image tag matters.** Tags are the git SHA (`<sha>-dirty-<timestamp>` for uncommitted
work). The tag is baked into the image and echoed by `/health`, so a deploy only succeeds when the
freshly built image is the one answering — a stale container can never pass for a fresh deploy.
Every run is logged to `logs/deploy-NNNN-*.log`.

Plain Docker works too: `docker compose up -d --build` (images are then tagged `dev`).

## Hidden vendor case (Session D)

No code or data change is needed for a new vendor:

1. Put its PDFs in `knowledge/`, named like the pack: `vendor-<tag>-proposal.pdf`,
   `vendor-<tag>-security-questionnaire.pdf`, `vendor-<tag>-pricing.pdf`.
2. `uv run scripts/deploy.py` -- the image carries the new PDFs, and the index is rebuilt
   automatically because the files changed.
3. Request the assessment at **http://127.0.0.1:8020/ui** with the vendor's name. Vendors not in
   `mcp_server/data/vendors.json` are matched to their documents by the `<tag>` or by their name
   appearing in them -- never to another vendor's documents (then the tool says so, with the
   candidates). Pricing comes from its pricing document (`calculate_tco` explicit mode).
4. Score the run: save the request as JSON, then
   `uv run python -m evaluation.live --request hidden.json`.

## Layout

```
src/hackathon2/        the service (FastAPI, uvicorn --factory)
  config.py            all settings (env / .env)          llm.py       Azure OpenAI chat + embeddings
  observability.py     Langfuse tracing + scores
  health.py            /health subsystem checks             service.py   API + /ui (static/index.html)
  agents/              orchestrator + specialists, decision gate adapter, recording
  mcp_server/          NFS MCP server (tools, resources, prompts) + the agents' stdio client
  rag/                 ingestion + retrieval (vector, hybrid, or keyword-only without embeddings)
  guardrails/          injection scan, tool authorization, the decision gate
tests/                 ~50 essential pytest tests, one file per area: schemas, devops, api, agents,
                       gate, guardrails, integration (agents <-> MCP <-> RAG, hidden vendor), evaluation
evaluation/            evaluation suite (FR14) -> results in evaluation-results/
knowledge/             the NFS knowledge pack PDFs (RAG corpus, baked into the image)
architecture/          design + course-unit map
scripts/               deploy.py (+ .ps1/.sh wrappers)
.github/workflows/     ci.yml -- lint, tests, image smoke test on PRs (currently disabled)
Dockerfile             multi-stage, uv --locked, non-root, HEALTHCHECK
docker-compose.yml     app + Postgres/pgvector
docker-compose.langfuse.yml   local Langfuse (tracing UI on :3020)
```

## Local development without Docker

```bash
docker compose up -d db      # just the database (127.0.0.1:5446)
uv run hackathon2            # API with auto-reload on http://127.0.0.1:8000
uv run pytest                # tests
uv run ruff check --fix .    # lint (CI runs the same check)
uv add <package>             # add a dependency -- commit pyproject.toml AND uv.lock
```

## CI

`.github/workflows/ci.yml` runs on pull requests into `main`: `uv sync --locked` → ruff → pytest →
build the production image → boot it and require `/health` to report the commit SHA. It is
**disabled on GitHub for now**; turn it back on with `gh workflow enable CI`.

## Handout checklist (section 15 deliverables)

| Deliverable | Where |
|---|---|
| README.md | this file |
| architecture/ | [architecture/README.md](architecture/README.md) |
| src/ · tests/ | `src/hackathon2/` · `tests/` |
| evaluation/ · evaluation-results/ | [evaluation/README.md](evaluation/README.md) · `evaluation-results/` |
| Dockerfile · docker-compose.yml | repo root |
| .env.example (no secrets) | repo root |
| pyproject.toml (+ uv.lock) | repo root |
| deployment/ | not used — this project runs locally only (`scripts/deploy.py`) |
