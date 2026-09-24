"""The agents: end-to-end wiring with a scripted model (no LLM, no network), no answers in the
prompts, and tool failures that never crash a run.

The scripted model replays AI messages in call order. It is shared by the orchestrator and its
subagents, so the script reads like the real run: plan -> delegate -> specialist searches and
reports -> next delegation -> final decision. What is tested is OUR wiring (delegation, tool
instrumentation, report collection, gate, human review), not the model's judgement -- that is
what the evaluation suite measures with the real model.
"""

import re
from itertools import count

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from hackathon2.agents import prompts
from hackathon2.agents.config import AgentSettings
from hackathon2.agents.context import RunContext, instrument_tool
from hackathon2.agents.orchestrator import FinalDecision
from hackathon2.agents.runner import AssessmentRunner
from hackathon2.agents.specialists import SPECIALISTS
from hackathon2.agents.tools import stub_provider
from hackathon2.schemas import AssessmentRequest, HumanDecision, ToolResult

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
            call(
                "task",
                {
                    "subagent_type": "security-risk-agent",
                    "description": "Assess security for Asteria AI "
                    "Systems (vendor_id asteria-ai-systems), 2000 users, confidential data, 3 years.",
                },
            ),
            call("search_vendor_documents", {"query": "encryption at rest", "vendor_id": "asteria-ai-systems"}),
            call("DomainReport", SECURITY_REPORT),
        ]
    if "procurement" in domains:
        script += [
            call(
                "task",
                {
                    "subagent_type": "procurement-finance-agent",
                    "description": "Assess procurement for "
                    "Asteria AI Systems (vendor_id asteria-ai-systems), 2000 users, 3 years.",
                },
            ),
            call("DomainReport", PROCUREMENT_REPORT),
        ]
    script.append(call("FinalDecision", FINAL_DECISION))
    return script


def _runner(model, **provider_kwargs) -> AssessmentRunner:
    return AssessmentRunner(
        model=model,
        tools_provider=stub_provider(**provider_kwargs),
        domains=("security", "procurement"),
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
    assert procurement.findings[0].control_id.endswith("-00")  # "domain assessment not completed"
    assert {f.status for f in procurement.findings} == {"MISSING"}  # + each mandatory control, added by the gate
    assert len(procurement.findings) > 1
    assert response.assessment.degraded_mode


def test_instructions_contain_no_vendor_names_or_corpus_answers():
    # Handout section 6: do not hard-code expected answers -- a hidden vendor is assessed on the day.
    texts = [
        prompts.orchestrator_prompt("   - specialist", prompts.PHASE2_NONE),
        str(FinalDecision.model_json_schema()),
    ]
    for domain, s in SPECIALISTS.items():
        texts += [
            prompts.specialist_prompt(title=s.title, domain=domain, prefix=s.control_prefix, focus=s.focus),
            s.description,
        ]
    hints = [r"asteria", r"corvid", r"vendor-[xy]", r"\b\d+\s*(hours?|days?)\b", r"soc ?2", r"enterprise plus"]
    leaks = [h for h in hints for text in texts if re.search(h, text, re.IGNORECASE)]
    assert not leaks


async def test_tool_failures_are_contained():
    def search_policy(query: str) -> str:
        raise ConnectionError("MCP server went away")

    ctx = RunContext(request=REQUEST)
    tool = instrument_tool(StructuredTool.from_function(search_policy, description="search"), ctx)
    assert ToolResult.model_validate_json(await tool.ainvoke({"query": "x"})).status == "unavailable"
    assert ctx.degraded  # an unavailable source degrades the run (FR14) ...
    calm = RunContext(request=REQUEST)
    calm.record_tool_result("calculate_tco", ToolResult.fail("error", "no verified pricing -- use explicit mode"))
    assert calm.tool_failures and not calm.degraded  # ... an invalid call does not
