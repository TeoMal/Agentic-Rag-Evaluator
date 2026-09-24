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

PowerShell / bash shortcuts do the same thing: `.\scripts\deploy.ps1` · `bash scripts/deploy.sh`.

## The deploy script — one pass, every time

| Command | What it does |
|---|---|
| `uv run scripts/deploy.py` | preflight (starts Docker Desktop if needed, fills `.env`) → `uv lock --check` + pytest → rebuild image → start Postgres/pgvector → force-recreate the app → wait until `/health` reports **this run's image tag** |
| `uv run scripts/deploy.py status` | containers + live `/health` |
| `uv run scripts/deploy.py down [--volumes]` | stop the local stack (`--volumes` also wipes the DB) |

Flags: `--dry-run` (print every mutating command, run only read-only checks) · `--skip-tests` ·
`--no-cache` (rebuild all layers, re-pull bases) · `--tag TAG` · `--yes` · `--timeout SECONDS`.

**Why the image tag matters.** Tags are the git SHA (`<sha>-dirty-<timestamp>` for uncommitted
work). The tag is baked into the image and echoed by `/health`, so a deploy only succeeds when the
freshly built image is the one answering — a stale container can never pass for a fresh deploy.
Every run is logged to `logs/deploy-NNNN-*.log`.

Plain Docker works too: `docker compose up -d --build` (images are then tagged `dev`).

## Layout

```
src/hackathon2/        the service (FastAPI, uvicorn --factory)
  config.py            all settings (env / .env)          llm.py       Azure OpenAI chat + embeddings
  health.py            /health subsystem checks             service.py   API + /ui (static/index.html)
  agents/  rag/  mcp_server/  guardrails/                 to be built -- one package per team role
tests/                 pytest (17 tests: config, API, deploy script)
evaluation/            evaluation suite (FR14) -> results in evaluation-results/
knowledge/             the NFS knowledge pack PDFs (RAG corpus, baked into the image)
architecture/          design + course-unit map
scripts/               deploy.py (+ .ps1/.sh wrappers)
.github/workflows/     ci.yml -- lint, tests, image smoke test on PRs (currently disabled)
Dockerfile             multi-stage, uv --locked, non-root, HEALTHCHECK
docker-compose.yml     app + Postgres/pgvector
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
