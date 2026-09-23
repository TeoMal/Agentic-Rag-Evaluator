# Hackathon 2 — Vendor Risk & Procurement Deep Agent

An evidence-grounded Deep Agent that assesses **Asteria AI Systems** for **Northstar Financial
Services** and recommends **APPROVE / CONDITIONAL APPROVAL / REJECT**, built from Deep Agents,
RAG, MCP, specialist agents, guardrails and an automated evaluation suite, and deployed on Azure
with Application Insights.

> Status: the **uv project and the full DevOps pipeline are in place** (build, test, containerise,
> deploy locally or to Azure, CI/CD, observability wiring). The agent itself is built next, in the
> packages under `src/hackathon2/` — see [architecture/](architecture/README.md) for the plan and
> which course unit each piece starts from.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/), Docker Desktop; for Azure also the
[Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) (`az login`).

```bash
uv sync                      # create .venv from uv.lock (Python 3.13 is fetched if missing)
uv run scripts/deploy.py     # test -> rebuild image -> recreate container -> verify, in one pass
```

The first run creates `.env` from `.env.example` and asks once for `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_ENDPOINT`. Then open **http://127.0.0.1:8020/docs**.

PowerShell / bash shortcuts do the same thing: `.\scripts\deploy.ps1` · `bash scripts/deploy.sh`.

## The deploy script — one pass, every time

`scripts/deploy.py` is the single implementation used on every OS **and** by GitHub Actions.

| Command | What it does |
|---|---|
| `uv run scripts/deploy.py` | **local**: preflight (starts Docker Desktop if needed, fills `.env`) → `uv lock --check` + pytest → rebuild image → start Postgres/pgvector → force-recreate the app → wait until `/health` reports **this run's image tag** |
| `uv run scripts/deploy.py azure` | Bicep infra (ACR, Log Analytics, App Insights, Container Apps env) → build `linux/amd64` → push to ACR → roll out a new revision → verify `https://<app>/health` reports this tag |
| `uv run scripts/deploy.py status [--azure]` | containers + live `/health` (and the Azure app's) |
| `uv run scripts/deploy.py down [--volumes]` | stop the local stack (`--volumes` also wipes the DB) |
| `uv run scripts/deploy.py teardown` | delete this project's Azure resources (tagged `project=hackathon2`) — asks first; the rest of a shared group is untouched |

Flags: `--dry-run` (print every mutating command, run only read-only checks) · `--skip-tests` ·
`--no-cache` (rebuild all layers, re-pull bases) · `--tag TAG` · `--yes` · `--timeout SECONDS`.

**Why the image tag matters.** Tags are the git SHA (`<sha>-dirty-<timestamp>` for uncommitted
work). The tag is baked into the image and echoed by `/health`, so a deploy only succeeds when the
freshly built image is the one answering — a stale container can never pass for a fresh deploy,
and Azure always gets a new revision. Every run is logged to `logs/deploy-NNNN-*.log`.

## Layout

```
src/hackathon2/        the service (FastAPI, uvicorn --factory)
  config.py            all settings (env / .env)          llm.py       Azure OpenAI chat + embeddings
  telemetry.py         Azure Monitor OpenTelemetry          health.py    /health subsystem checks
  service.py           API: GET /  GET /health
  agents/  rag/  mcp_server/  guardrails/                 to be built -- one package per team role
tests/                 pytest (19 tests: config, API, deploy script)
evaluation/            evaluation suite (FR14) -> results in evaluation-results/
knowledge/             the NFS knowledge pack PDFs (RAG corpus, baked into the image)
architecture/          design + course-unit map
deployment/            main.bicep + Azure setup guide
scripts/               deploy.py (+ .ps1/.sh wrappers)
.github/workflows/     ci.yml (PRs), cd.yml (main -> Azure)
Dockerfile             multi-stage, uv --locked, non-root, HEALTHCHECK
docker-compose.yml     app + Postgres/pgvector (local)
```

## Local development without Docker

```bash
docker compose up -d db      # just the database (127.0.0.1:5446)
uv run hackathon2            # API with auto-reload on http://127.0.0.1:8000
uv run pytest                # tests
uv run ruff check --fix .    # lint (CI runs the same check)
uv add <package>             # add a dependency -- commit pyproject.toml AND uv.lock
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`, every PR into `main`): `uv sync --locked` → ruff → pytest →
  build the production image → boot it and require `/health` to report the commit SHA.
- **CD** (`.github/workflows/cd.yml`, every push to `main`): runs CI, then
  `uv run scripts/deploy.py azure --yes --tag <sha>` against Azure; the deploy log is kept as a
  build artifact. One-time GitHub setup: [deployment/README.md](deployment/README.md).

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
| deployment/ | [deployment/README.md](deployment/README.md), `deployment/main.bicep` |
