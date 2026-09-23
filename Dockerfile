# syntax=docker/dockerfile:1
#
# Two stages. The builder resolves and installs with uv; the runtime image gets
# only the finished virtualenv and the knowledge pack -- no uv, no build tools,
# no source tree, no tests -- runs as non-root, and carries its own IMAGE_TAG so
# GET /health proves which build is serving (scripts/deploy.py checks it).

ARG PYTHON_VERSION=3.13

# Pinned, not :latest -- `uv sync --locked` exists to make the image match
# uv.lock exactly, and a floating resolver would quietly undermine that.
FROM ghcr.io/astral-sh/uv:0.12.11 AS uv

FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /bin/uv
# Bytecode is compiled at build time so containers start faster; copy mode
# because the cache is a mount; never download a Python -- use the image's.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Layer 1 -- dependencies only. Re-runs ONLY when pyproject.toml or uv.lock
# change, so ordinary code (and README) edits rebuild in seconds.
# --locked fails the build if uv.lock is stale instead of resolving something else.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Layer 2 -- the project itself, installed non-editable so the runtime stage
# needs nothing but the venv. README.md is needed here because pyproject.toml
# declares it as the package readme (building the project reads it).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:${PYTHON_VERSION}-slim AS runtime
# Fixed uid so file ownership is predictable if a volume is ever mounted in.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser knowledge ./knowledge
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    KNOWLEDGE_DIR=/app/knowledge
USER appuser
EXPOSE 8000

# python:slim has no curl -- reuse the interpreter that is already there.
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)"]

# Last on purpose: the tag changes on every build, and nothing after it can be cached.
ARG IMAGE_TAG=dev
ENV IMAGE_TAG=${IMAGE_TAG}

# --factory: the app is built at startup by create_app(), not at import time.
CMD ["uvicorn", "hackathon2.service:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
