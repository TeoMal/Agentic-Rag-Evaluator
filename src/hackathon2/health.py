"""Subsystem checks behind GET /health.

Every check reports a state instead of raising: /health is also the liveness
signal for Docker and Azure, and an optional dependency being down (Postgres
is absent on Azure by design) must not get the container restarted.
"""

from typing import Literal

import psycopg

from hackathon2.config import Settings

DatabaseState = Literal["ok", "unavailable", "not_configured"]


def check_database(settings: Settings, timeout_seconds: int = 2) -> DatabaseState:
    if not settings.database_url:
        return "not_configured"
    try:
        with psycopg.connect(settings.database_url, connect_timeout=timeout_seconds) as conn:
            conn.execute("SELECT 1")
        return "ok"
    except psycopg.Error:
        return "unavailable"


def count_knowledge_documents(settings: Settings) -> int:
    folder = settings.knowledge_dir
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in {".pdf", ".md", ".txt"})
