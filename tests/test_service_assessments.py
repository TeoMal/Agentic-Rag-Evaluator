"""The assessment API and the web UI, over HTTP: request in, assessment out, human review.

The runner is the real AssessmentRunner (stub tools, provisional gate); only the agent run is
canned, so these tests check the API wiring -- the agent's own wiring is test_agents_workflow.py.
"""

import re

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from hackathon2.agents.config import AgentSettings
from hackathon2.agents.gate_fallback import provisional_gate
from hackathon2.agents.runner import AssessmentRunner
from hackathon2.agents.tools import stub_provider
from hackathon2.schemas import (
    Assessment,
    AssessmentRequest,
    AssessmentResponse,
    DomainReport,
    Finding,
    RunMetrics,
)
from hackathon2.service import create_app

REQUEST = {
    "vendor_name": "Asteria AI Systems",
    "use_case": "Enterprise Generative AI platform",
    "user_count": 2000,
    "data_classification": "confidential",
    "contract_years": 3,
    "notes": "The platform may process confidential corporate documents.",
}


class HeldForReview(AssessmentRunner):
    """A runner whose agent run always ends held for human review."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            tools_provider=stub_provider(), gate=provisional_gate, settings=AgentSettings(_env_file=None), **kwargs
        )
        self.requests: list[AssessmentRequest] = []

    async def run(self, request: AssessmentRequest) -> AssessmentResponse:
        self.requests.append(request)
        finding = Finding(
            domain="security",
            control_id="SEC-02",
            title="Certification",
            status="MISSING",
            severity="high",
            claim="No SOC 2 Type II report was supplied.",
        )
        assessment = Assessment(
            recommendation="CONDITIONAL_APPROVAL",
            risk_rating="high",
            domains=[
                DomainReport(
                    domain="security",
                    risk_rating="high",
                    summary="Certification evidence is missing.",
                    findings=[finding],
                )
            ],
            executive_summary="Conditional approval, pending the SOC 2 report.",
            vendor_name=request.vendor_name,
            vendor_id=request.vendor_id or "",
            human_approval="pending",
        )
        return self._store(
            AssessmentResponse(
                assessment_id=assessment.assessment_id,
                status="awaiting_approval",
                assessment=assessment,
                metrics=RunMetrics(duration_seconds=1.0),
            )
        )


class BrokenModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        raise ConnectionError("model endpoint unreachable")


@pytest.fixture
def runner() -> HeldForReview:
    return HeldForReview()


@pytest.fixture
def client(make_settings, runner) -> TestClient:
    return TestClient(create_app(make_settings(), runner=runner))


def test_request_runs_and_the_assessment_can_be_fetched(client, runner):
    body = client.post("/assessments", json=REQUEST).json()
    assert body["status"] == "awaiting_approval"
    assert body["assessment"]["vendor_id"] == "asteria-ai-systems"  # derived from the name (FR01)
    assert runner.requests[0].user_count == 2000
    assert client.get(f"/assessments/{body['assessment_id']}").json() == body


@pytest.mark.parametrize(
    "change",
    [
        {"use_case": None},  # missing
        {"user_count": 0},
        {"data_classification": "secret"},
        {"approve_automatically": True},  # unknown fields are rejected
    ],
)
def test_invalid_request_is_rejected_before_the_agent_runs(client, runner, change):
    payload = {k: v for k, v in (REQUEST | change).items() if v is not None}
    assert client.post("/assessments", json=payload).status_code == 422
    assert runner.requests == []


def test_unknown_assessment_is_404(client):
    assert client.get("/assessments/nope").status_code == 404
    decision = {"approved": True, "reviewer": "cro@northstar"}
    assert client.post("/assessments/nope/decision", json=decision).status_code == 404


def test_approval_completes_the_assessment_once(client):
    assessment_id = client.post("/assessments", json=REQUEST).json()["assessment_id"]
    decision = {"approved": True, "reviewer": "cro@northstar", "comment": "SOC 2 report received."}
    body = client.post(f"/assessments/{assessment_id}/decision", json=decision).json()
    assert body["status"] == "completed"
    assert body["assessment"]["human_approval"] == "approved"
    assert body["assessment"]["reviewer"] == "cro@northstar"
    assert client.post(f"/assessments/{assessment_id}/decision", json=decision).status_code == 409


def test_rejection_closes_the_assessment(client):
    assessment_id = client.post("/assessments", json=REQUEST).json()["assessment_id"]
    decision = {"approved": False, "reviewer": "cro@northstar"}
    body = client.post(f"/assessments/{assessment_id}/decision", json=decision).json()
    assert body["status"] == "rejected_by_reviewer"
    assert body["assessment"]["human_approval"] == "rejected"


def test_a_decision_needs_a_named_reviewer(client):
    assessment_id = client.post("/assessments", json=REQUEST).json()["assessment_id"]
    response = client.post(f"/assessments/{assessment_id}/decision", json={"approved": True})
    assert response.status_code == 422


def test_model_failure_comes_back_as_a_failed_assessment(make_settings):
    # FR14 through the API: the real runner, a model that cannot be reached.
    runner = AssessmentRunner(
        model=BrokenModel(messages=iter([])),
        tools_provider=stub_provider(),
        gate=provisional_gate,
        settings=AgentSettings(_env_file=None),
    )
    client = TestClient(create_app(make_settings(), runner=runner))
    body = client.post("/assessments", json=REQUEST).json()
    assert body["status"] == "failed"
    assert "model endpoint unreachable" in body["error"]
    assert client.get(f"/assessments/{body['assessment_id']}").json()["status"] == "failed"


def test_without_an_llm_assessments_are_unavailable_but_the_service_is_up(make_settings):
    client = TestClient(create_app(make_settings()))
    assert client.post("/assessments", json=REQUEST).status_code == 503
    assert client.get("/assessments/anything").status_code == 404
    assert client.get("/health").status_code == 200


def test_ui_form_matches_the_request_contract(client):
    response = client.get("/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    form = re.search(r'<form id="request-form".*?</form>', response.text, re.DOTALL).group()
    assert set(re.findall(r'name="(\w+)"', form)) == set(AssessmentRequest.model_fields)
