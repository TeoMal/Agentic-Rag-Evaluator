"""FastAPI entry point -- `uvicorn --factory hackathon2.service:create_app`.

Factory mode: nothing happens at import time (no .env read), so tests can build
apps from their own Settings.

Only the operational surface lives here for now (liveness + health). The vendor
assessment endpoints (FR01 request in, FR12 structured assessment out, FR13
human approval) get added next to these as the agent lands.
"""

import logging

from fastapi import FastAPI

from hackathon2 import __version__
from hackathon2.config import Settings, get_settings
from hackathon2.health import check_database, count_knowledge_documents

SERVICE_NAME = "hackathon2"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    app = FastAPI(
        title="Hackathon 2 - Vendor Risk & Procurement Deep Agent",
        version=__version__,
    )

    @app.get("/", tags=["ops"])
    def liveness() -> dict:
        """Cheap liveness probe (Docker HEALTHCHECK)."""
        return {"status": "ok"}

    @app.get("/health", tags=["ops"])
    def health() -> dict:
        """What is actually running, and which subsystems are live."""
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "version": __version__,
            "image_tag": settings.image_tag,
            "environment": settings.app_env,
            "checks": {
                "llm": "configured" if settings.llm_configured else "not_configured",
                "database": check_database(settings),
                "knowledge_documents": count_knowledge_documents(settings),
            },
        }

    return app
