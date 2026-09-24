"""FastAPI entry point -- `uvicorn --factory hackathon2.service:create_app`.

Factory mode: nothing happens at import time (no .env read), so tests can build
apps from their own Settings -- and from their own runner.

    GET  /                               liveness; also the container healthcheck target
    GET  /health                         what is running, which subsystems are live
    GET  /ui                             web page: the request form and the assessment report
    POST /assessments                    FR01 request in -> FR11 structured assessment out
    GET  /assessments/{id}               an assessment's current state
    POST /assessments/{id}/decision      FR12 human review of an assessment awaiting approval

Assessments live in the runner's memory: a restart forgets them.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from hackathon2 import __version__
from hackathon2.agents.runner import AssessmentRunner
from hackathon2.config import Settings, get_settings
from hackathon2.health import check_database, count_knowledge_documents
from hackathon2.schemas import AssessmentRequest, AssessmentResponse, HumanDecision

SERVICE_NAME = "hackathon2"

#: Served at /ui, not /, because GET / is the container healthcheck target and has to keep
#: returning JSON.
UI_PATH = Path(__file__).parent / "static" / "index.html"

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, runner: AssessmentRunner | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    app = FastAPI(
        title="Hackathon 2 - Vendor Risk & Procurement Deep Agent",
        version=__version__,
    )

    def get_runner() -> AssessmentRunner:
        """The runner, built on the first assessment -- so the ops endpoints work without an LLM."""
        nonlocal runner
        if runner is None:
            if not settings.llm_configured:
                raise HTTPException(
                    status_code=503,
                    detail="Azure OpenAI is not configured: set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env",
                )
            runner = AssessmentRunner()
        return runner

    def find(assessment_id: str) -> AssessmentResponse:
        found = runner.get(assessment_id) if runner is not None else None
        if found is None:
            raise HTTPException(status_code=404, detail=f"No assessment with id {assessment_id!r}")
        return found

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

    @app.get("/ui", response_class=HTMLResponse, tags=["ui"])
    def ui() -> HTMLResponse:
        """The web page: request form, assessment report and human review."""
        try:
            return HTMLResponse(UI_PATH.read_text(encoding="utf-8"))
        except OSError as exc:
            # The API is the product; a missing static file must not take it down.
            logger.warning("UI unavailable: %s", exc)
            raise HTTPException(status_code=404, detail="UI is not available in this build") from None

    @app.post("/assessments", tags=["assessments"])
    async def create_assessment(request: AssessmentRequest) -> AssessmentResponse:
        """Run one vendor assessment end to end (about a minute with the real model).

        A run that fails comes back as status "failed" with the error (FR14), not as an HTTP error.
        """
        return await get_runner().run(request)

    @app.get("/assessments/{assessment_id}", tags=["assessments"])
    def get_assessment(assessment_id: str) -> AssessmentResponse:
        return find(assessment_id)

    @app.post("/assessments/{assessment_id}/decision", tags=["assessments"])
    async def decide(assessment_id: str, decision: HumanDecision) -> AssessmentResponse:
        """Approve or reject an assessment that is awaiting human review (FR12)."""
        find(assessment_id)
        try:
            return await get_runner().decide(assessment_id, decision)
        except ValueError as exc:  # not awaiting approval (already decided, or completed)
            raise HTTPException(status_code=409, detail=str(exc)) from None

    return app
