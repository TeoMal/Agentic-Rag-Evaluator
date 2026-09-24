"""The assessment API and the web UI over HTTP. The agent run is canned; everything else is real."""

import re

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from hackathon2.agents.config import AgentSettings
from hackathon2.agents.runner import AssessmentRunner
from hackathon2.agents.tools import stub_provider
from hackathon2.schemas import Assessment, AssessmentRequest, AssessmentResponse, DomainReport, Finding
from hackathon2.service import create_app

REQUEST = {
    "vendor_name": "Asteria AI Systems",
    "use_case": "Enterprise Generative AI platform",
    "user_count": 2000,
    "data_classification": "confidential",
}


class HeldForReview(AssessmentRunner):
    """The real runner, except that every agent run ends held for human review."""

    def __init__(self) -> None:
        super().__init__(tools_provider=stub_provider(), settings=AgentSettings(_env_file=None))

    async def run(self, request: AssessmentRequest) -> AssessmentResponse:
        finding = Finding(
            domain="security",
            control_id="SEC-02",
            title="Certification",
            status="MISSING",
            severity="high",
            claim="No SOC 2 report was supplied.",
        )
        report = DomainReport(domain="security", risk_rating="high", summary="Gap.", findings=[finding])
        assessment = Assessment(
            recommendation="CONDITIONAL_APPROVAL",
            risk_rating="high",
            domains=[report],
            executive_summary="Pending SOC 2.",
            vendor_name=request.vendor_name,
            vendor_id=request.vendor_id or "",
            human_approval="pending",
        )
        return self._store(
            AssessmentResponse(
                assessment_id=assessment.assessment_id, status="awaiting_approval", assessment=assessment
            )
        )


class BrokenModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        raise ConnectionError("model endpoint unreachable")


def _client(make_settings, runner=None) -> TestClient:
    return TestClient(create_app(make_settings(), runner=runner or HeldForReview()))


def test_a_request_is_assessed_and_can_be_fetched(make_settings):
    client = _client(make_settings)
    body = client.post("/assessments", json=REQUEST).json()
    assert body["status"] == "awaiting_approval"
    assert body["assessment"]["vendor_id"] == "asteria-ai-systems"  # derived from the name (FR01)
    assert client.get(f"/assessments/{body['assessment_id']}").json() == body
    assert client.get("/assessments/nope").status_code == 404


def test_invalid_requests_are_rejected_and_no_llm_means_503(make_settings):
    client = _client(make_settings)
    for change in ({"user_count": 0}, {"data_classification": "secret"}, {"approve_automatically": True}):
        assert client.post("/assessments", json=REQUEST | change).status_code == 422
    assert TestClient(create_app(make_settings())).post("/assessments", json=REQUEST).status_code == 503


def test_human_review_approves_once(make_settings):
    client = _client(make_settings)
    assessment_id = client.post("/assessments", json=REQUEST).json()["assessment_id"]
    url = f"/assessments/{assessment_id}/decision"
    assert client.post(url, json={"approved": True}).status_code == 422  # a named reviewer is required
    body = client.post(url, json={"approved": True, "reviewer": "cro@northstar"}).json()
    assert (body["status"], body["assessment"]["human_approval"]) == ("completed", "approved")
    assert client.post(url, json={"approved": False, "reviewer": "cro@northstar"}).status_code == 409


def test_a_model_failure_comes_back_as_a_failed_assessment(make_settings):
    runner = AssessmentRunner(
        model=BrokenModel(messages=iter([])), tools_provider=stub_provider(), settings=AgentSettings(_env_file=None)
    )
    body = _client(make_settings, runner).post("/assessments", json=REQUEST).json()
    assert body["status"] == "failed" and "model endpoint unreachable" in body["error"]  # FR14


def test_the_ui_form_matches_the_request_contract(make_settings):
    response = _client(make_settings).get("/ui")
    assert response.headers["content-type"].startswith("text/html")
    form = re.search(r'<form id="request-form".*?</form>', response.text, re.DOTALL).group()
    assert set(re.findall(r'name="(\w+)"', form)) == set(AssessmentRequest.model_fields)
