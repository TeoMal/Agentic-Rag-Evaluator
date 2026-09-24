"""End-to-end wiring of the deep agent with a scripted model: no LLM, no network.

The scripted model replays AI messages in call order. It is shared by the orchestrator and its
subagents, so the script reads like the real run: plan -> delegate -> specialist searches and
reports -> next delegation -> final decision. What is tested is OUR wiring (delegation, tool
instrumentation, report collection, gate, human review), not the model's judgement -- that is
what the evaluation suite measures with the real model.
"""

from itertools import count

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from hackathon2.agents.config import AgentSettings
from hackathon2.agents.gate_fallback import provisional_gate
from hackathon2.agents.runner import AssessmentRunner
from hackathon2.agents.tools import stub_provider
from hackathon2.schemas import AssessmentRequest, HumanDecision

REQUEST = AssessmentRequest(
    vendor_name="Asteria AI Systems",
    use_case="Enterprise Generative AI platform",
    user_count=2000,
    data_classification="confidential",
    contract_years=3,
    notes="The platform may process confidential corporate documents.",
)

ENCRYPTION = "vendor-x-security-questionnaire#sB#c1"
DECISION_RULES = "vendor-risk-policy#s3#c1"

SECURITY_REPORT = {
    "domain": "security",
    "risk_rating": "high",
    "summary": "Encryption is supported; certification evidence is missing.",
    "findings": [
        {
            "domain": "security",
            "control_id": "SEC-01",
            "title": "Encryption",
            "status": "SUPPORTED",
            "severity": "low",
            "claim": "Data is encrypted in transit with TLS 1.2+ and at rest with AES-256.",
            "citations": [
                {
                    "chunk_id": ENCRYPTION,
                    "source": "vendor-x-security-questionnaire.pdf",
                    "doc_type": "vendor_claim",
                    "quote": "B1 TLS 1.2+: YES. B2 Encryption at rest: YES, AES-256.",
                }
            ],
        },
        {
            "domain": "security",
            "control_id": "SEC-02",
            "title": "Certification",
            "status": "MISSING",
            "severity": "high",
            "claim": "The vendor claims ISO 27001 and SOC 2 Type II, but the reports were not supplied.",
            "remediation": "Obtain the current SOC 2 Type II report and ISO 27001 certificate before go-live.",
        },
    ],
}

PROCUREMENT_REPORT = {
    "domain": "procurement",
    "risk_rating": "medium",
    "summary": "No evidence of competitive quotes.",
    "findings": [
        {
            "domain": "procurement",
            "control_id": "PROC-01",
            "title": "Competitive sourcing",
            "status": "MISSING",
            "severity": "medium",
            "claim": "No competitive quotes or sole-source justification were found.",
            "remediation": "Document competitive quotes or a sole-source justification.",
        }
    ],
}

FINAL_DECISION = {
    "recommendation": "CONDITIONAL_APPROVAL",
    "risk_rating": "high",
    "decision_basis": [
        {
            "chunk_id": DECISION_RULES,
            "source": "vendor-risk-policy.pdf",
            "doc_type": "policy",
            "section": "VR-006 3. Decision outcomes",
            "quote": "CONDITIONAL APPROVAL: No prohibited control failure, but remediation or contractual conditions "
            "are required before or shortly after go-live.",
        }
    ],
    "conditions": [
        {"kind": "remediation", "text": "Provide a current SOC 2 Type II report.", "control_ids": ["SEC-02"]}
    ],
    "executive_summary": "Conditional approval: certification and sourcing evidence are missing.",
}


class ScriptedModel(GenericFakeChatModel):
    """Replays scripted AIMessages; tools are accepted and ignored."""

    def bind_tools(self, tools, **kwargs):
        return self


class BrokenModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        raise RuntimeError("model endpoint unreachable")


def _script(domains=("security", "procurement")) -> list[AIMessage]:
    ids = count(1)

    def call(name: str, args: dict) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(ids)}"}])

    todos = [{"content": f"Assess the {d} domain", "status": "pending"} for d in domains]
    script = [
        call("write_todos", {"todos": [*todos, {"content": "Synthesise the decision", "status": "pending"}]}),
        call("search_policy", {"query": "decision outcomes approve conditional approval reject"}),
    ]
    if "security" in domains:
        script += [
            call("task", {"subagent_type": "security-risk-agent", "description": "Assess security for Asteria AI "
                          "Systems (vendor_id asteria-ai-systems), 2000 users, confidential data, 3 years."}),
            call("search_vendor_documents", {"query": "encryption at rest", "vendor_id": "asteria-ai-systems"}),
            call("DomainReport", SECURITY_REPORT),
        ]
    if "procurement" in domains:
        script += [
            call("task", {"subagent_type": "procurement-finance-agent", "description": "Assess procurement for "
                          "Asteria AI Systems (vendor_id asteria-ai-systems), 2000 users, 3 years."}),
            call("DomainReport", PROCUREMENT_REPORT),
        ]
    script.append(call("FinalDecision", FINAL_DECISION))
    return script


def _runner(model, **provider_kwargs) -> AssessmentRunner:
    return AssessmentRunner(
        model=model,
        tools_provider=stub_provider(**provider_kwargs),
        domains=("security", "procurement"),
        gate=provisional_gate,
        settings=AgentSettings(_env_file=None),
    )


async def test_end_to_end_assessment_with_human_review():
    runner = _runner(ScriptedModel(messages=iter(_script())))
    response = await runner.run(REQUEST)

    assert response.status == "awaiting_approval", response.error
    assessment = response.assessment
    assert [d.domain for d in assessment.domains] == ["security", "procurement"]
    assert assessment.recommendation == "CONDITIONAL_APPROVAL"
    assert assessment.human_approval == "pending"
    assert not assessment.degraded_mode

    metrics = response.metrics
    assert metrics.subagents_called == ["security-risk-agent", "procurement-finance-agent"]
    assert "search_vendor_documents" in metrics.tools_called
    assert metrics.tools_called[0] == "search_policy"  # the orchestrator reads the decision rules first
    assert {ENCRYPTION, DECISION_RULES} <= set(metrics.retrieved_chunk_ids)
    assert "Decision basis: VR-006 3. Decision outcomes" in assessment.executive_summary

    final = await runner.decide(assessment.assessment_id, HumanDecision(approved=True, reviewer="cro@northstar"))
    assert final.status == "completed"
    assert final.assessment.human_approval == "approved"


async def test_tool_outage_degrades_the_assessment():
    runner = _runner(ScriptedModel(messages=iter(_script())), unavailable=["search_vendor_documents"])
    response = await runner.run(REQUEST)
    assert response.status == "awaiting_approval", response.error
    assert response.assessment.degraded_mode


async def test_undelegated_domain_is_reported_missing():
    runner = _runner(ScriptedModel(messages=iter(_script(domains=("security",)))))
    response = await runner.run(REQUEST)
    procurement = next(d for d in response.assessment.domains if d.domain == "procurement")
    assert [f.status for f in procurement.findings] == ["MISSING"]
    assert response.assessment.degraded_mode


async def test_model_failure_returns_failed_instead_of_raising():
    response = await _runner(BrokenModel(messages=iter([]))).run(REQUEST)
    assert response.status == "failed"
    assert "model endpoint unreachable" in response.error
