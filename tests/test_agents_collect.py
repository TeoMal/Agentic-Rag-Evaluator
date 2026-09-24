"""Reading specialist reports back, unassessed domains, the provisional gate and the report."""

from langchain_core.messages import AIMessage, ToolMessage

from hackathon2.agents.collect import collect_domain_reports, missing_domain_report, subagents_called
from hackathon2.agents.gate_fallback import provisional_gate
from hackathon2.agents.orchestrator import with_decision_basis
from hackathon2.agents.report import render_markdown
from hackathon2.agents.specialists import SUBAGENT_DOMAINS
from hackathon2.schemas import AssessmentDraft, AssessmentRequest, DomainReport, Evidence, Finding

REQUEST = AssessmentRequest(
    vendor_name="Asteria AI Systems",
    use_case="Enterprise Generative AI platform",
    user_count=2000,
    data_classification="confidential",
)


def _report(domain: str = "security") -> DomainReport:
    return DomainReport(
        domain=domain,
        risk_rating="high",
        summary="Certification evidence is missing.",
        findings=[
            Finding(
                domain=domain,
                control_id="SEC-01",
                title="Encryption",
                status="SUPPORTED",
                severity="low",
                claim="Data is encrypted at rest with AES-256.",
                citations=[
                    Evidence(
                        chunk_id="vendor-x-security-questionnaire#sB#c1",
                        source="vendor-x-security-questionnaire.pdf",
                        doc_type="vendor_claim",
                        quote="B2 Encryption at rest: YES, AES-256.",
                    )
                ],
            ),
            Finding(
                domain=domain,
                control_id="SEC-02",
                title="Certification",
                status="MISSING",
                severity="high",
                claim="No SOC 2 Type II or ISO 27001 evidence was found.",
                remediation="Provide a current SOC 2 Type II report.",
            ),
        ],
    )


def _delegation(agent: str, call_id: str, content: str) -> list:
    return [
        AIMessage(content="", tool_calls=[{"name": "task", "args": {"subagent_type": agent, "description": "x"},
                                           "id": call_id}]),
        ToolMessage(content=content, tool_call_id=call_id, name="task"),
    ]


def test_valid_report_is_collected_by_domain():
    messages = _delegation("security-risk-agent", "c1", _report().model_dump_json())
    reports, problems = collect_domain_reports(messages, SUBAGENT_DOMAINS)
    assert problems == []
    assert reports["security"].findings[1].status == "MISSING"
    assert subagents_called(messages) == ["security-risk-agent"]


def test_report_for_the_wrong_domain_is_rejected():
    # A security agent must not be able to fill in the legal domain.
    wrong = _report("security").model_dump_json().replace('"security"', '"legal"')
    reports, problems = collect_domain_reports(_delegation("security-risk-agent", "c1", wrong), SUBAGENT_DOMAINS)
    assert reports == {}
    assert "instead of 'security'" in problems[0]


def test_unparseable_report_is_a_problem_not_a_crash():
    reports, problems = collect_domain_reports(
        _delegation("security-risk-agent", "c1", "I could not finish the assessment."), SUBAGENT_DOMAINS
    )
    assert reports == {} and "did not validate" in problems[0]


def test_unassessed_domain_is_missing_never_passed():
    report = missing_domain_report("ai_governance", "tool timeout")
    assert report.risk_rating == "high"
    assert [f.status for f in report.findings] == ["MISSING"]
    assert report.findings[0].control_id == "AIG-00"


def _draft(recommendation="CONDITIONAL_APPROVAL", risk="high") -> AssessmentDraft:
    return AssessmentDraft(
        recommendation=recommendation, risk_rating=risk, domains=[_report()], executive_summary="Summary."
    )


def test_provisional_gate_requires_human_review_for_high_risk():
    assessment = provisional_gate(_draft(), REQUEST, set(), degraded=False)
    assert assessment.human_approval == "pending"
    assert assessment.vendor_id == "asteria-ai-systems"


def test_provisional_gate_lets_low_risk_rejection_through():
    assessment = provisional_gate(_draft("REJECT", "low"), REQUEST, set(), degraded=False)
    assert assessment.human_approval == "not_required"


def test_provisional_gate_requires_review_when_degraded():
    assessment = provisional_gate(_draft("REJECT", "low"), REQUEST, set(), degraded=True)
    assert assessment.human_approval == "pending" and assessment.degraded_mode


def test_markdown_report_names_gaps_and_citations():
    text = render_markdown(provisional_gate(_draft(), REQUEST, set(), degraded=False))
    assert "CONDITIONAL APPROVAL" in text
    gaps = text.split("## Unknown (missing) or contradictory evidence")[1].split("##")[0]
    assert "**SEC-02** (UNKNOWN, high)" in gaps  # MISSING is shown in NFS vocabulary
    assert "vendor-x-security-questionnaire#sB#c1" in text


def _basis() -> Evidence:
    return Evidence(
        chunk_id="vendor-risk-policy#s3#c1",
        source="vendor-risk-policy.pdf",
        doc_type="policy",
        section="VR-006 3. Decision outcomes",
        quote="REJECT: One or more mandatory controls cannot be met",
    )


def test_decision_basis_is_appended_to_the_summary():
    text = with_decision_basis("Conditional approval.", [_basis()])
    assert text.endswith("Decision basis: VR-006 3. Decision outcomes (vendor-risk-policy#s3#c1)")


def test_decision_basis_never_breaks_the_summary_length_limit():
    text = with_decision_basis("x" * 2400, [_basis()] * 40)
    assert len(text) <= 3000
    AssessmentDraft(recommendation="REJECT", risk_rating="high", domains=[], executive_summary=text)
