"""The evaluation suite: metrics, the judge-driven grounding scores, the run-level checks and live
run records. No test calls a real LLM."""

import json

import pytest

from evaluation import checks
from evaluation.judge import Judgment
from evaluation.live import record_run, summarize
from evaluation.metrics import Target, citation_problems, quote_in_text, rank_scores
from evaluation.run import DATASETS, RunRecord, check_gates, evaluate_assessment, evaluate_grounding, load_cases
from hackathon2.agents.gate import RunEvidence
from hackathon2.schemas import AssessmentRequest, Evidence, RequirementControl, SearchHit

SCENARIOS = load_cases("injection_scenarios.json", checks.InjectionScenario)
SAMPLE = RunRecord.load(DATASETS / "sample_run.json")


class FakeJudge:
    """Scripted judge: maps claim substrings to (verdict, supporting blocks or None = all)."""

    model = "fake"

    def __init__(self, script: dict) -> None:
        self.script, self.calls = script, []

    def judge_many(self, items):
        self.calls.extend(items)
        out = []
        for claim, _status, evidence in items:
            verdict, supporting = next((v for key, v in self.script.items() if key in claim), ("supported", None))
            blocks = list(range(1, len(evidence) + 1)) if supporting is None else supporting
            out.append(Judgment(reasoning="scripted", verdict=verdict, supporting=blocks))
        return out


def _hit(doc_id: str, text: str = "") -> SearchHit:
    return SearchHit(
        chunk_id=f"{doc_id}#s1#c1",
        doc_id=doc_id,
        source=f"{doc_id}.pdf",
        doc_type="policy",
        text=f"<untrusted_document>{text}</untrusted_document>",
    )


@pytest.fixture
def clean() -> RunRecord:
    """The sample run with every run-level defect fixed: it must pass all the checks."""
    data = json.loads((DATASETS / "sample_run.json").read_text(encoding="utf-8"))
    response = data["response"]
    response["metrics"]["tools_called"] = [
        "get_policy_requirements",
        "search_policy",
        "search_vendor_documents",
        "calculate_tco",
    ]
    a = response["assessment"]
    a["executive_summary"] = "Conditional approval recommended."
    a["conditions"] = [
        {"kind": "contractual", "text": "24-hour incident notification.", "control_ids": ["SEC-09"]},
        {"kind": "remediation", "text": "Document data residency.", "control_ids": ["SEC-07"]},
        {"kind": "contractual", "text": "Enterprise Plus with 7-day retention.", "control_ids": []},
    ]
    a["domains"][0]["findings"] = [f for f in a["domains"][0]["findings"] if f["control_id"] != "SEC-12"]
    a["domains"].append(
        {
            "domain": "legal",
            "risk_rating": "medium",
            "summary": "No DPA yet.",
            "findings": [
                {
                    "domain": "legal",
                    "control_id": "LEG-01",
                    "title": "DPA",
                    "status": "MISSING",
                    "severity": "medium",
                    "claim": "No DPA was found.",
                    "citations": [],
                }
            ],
        }
    )
    data["required_controls"] = ["SEC-07", "SEC-09", "AIG-02"]
    return RunRecord.model_validate(data)


def test_retrieval_ranking_metrics():
    targets = [
        Target(doc_id="information-security-policy", grade=3),
        Target(doc_id="data-classification-policy", grade=2),
    ]
    perfect = rank_scores([_hit("information-security-policy"), _hit("data-classification-policy")], targets, k=5)
    assert perfect == {"hit": 1.0, "recall": 1.0, "mrr": 1.0, "ndcg": pytest.approx(1.0)}
    late = rank_scores(
        [_hit("vendor-x-pricing"), _hit("vendor-x-proposal"), _hit("information-security-policy")], targets, k=5
    )
    assert late["mrr"] == pytest.approx(1 / 3) and 0 < late["ndcg"] < 1


def test_citation_checks():
    text = "<untrusted_document>Enterprise licence: EUR 25 per user,\n billed annually.</untrusted_document>"
    assert quote_in_text("enterprise LICENCE: eur 25 per user, billed annually.", text)  # case and spacing
    assert not quote_in_text("Enterprise licence: EUR 20 per user", text)
    hit = _hit("vendor-x-pricing", "Enterprise licence: EUR 25 per user per month.")
    log, hits = {hit.chunk_id}, {hit.chunk_id: hit}

    def cite(quote="EUR 25 per user per month", chunk_id=hit.chunk_id, source=hit.source):
        return Evidence(chunk_id=chunk_id, source=source, doc_type="policy", quote=quote)

    assert citation_problems(cite(), log, hits) == []
    assert citation_problems(cite("EUR 20 per user"), log, hits) == ["misquoted"]
    assert citation_problems(cite(source="vendor-x-proposal.pdf"), log, hits) == ["source_mismatch"]
    fabricated = cite(chunk_id="ai-governance-policy#s9#c4", source="ai-governance-policy.pdf")
    assert citation_problems(fabricated, log, hits) == ["fabricated"]


def test_code_checks_alone_catch_the_planted_citation_defects():
    agg = evaluate_grounding(SAMPLE, judge=None)["aggregate"]
    assert agg["citation_problems"] == {"misquoted": 1, "fabricated": 1, "source_mismatch": 1}
    assert agg["uncited_material"] == 1 and agg["groundedness"] is None  # not measured without a judge
    gated = SAMPLE.model_copy(deep=True)  # claims the decision gate downgraded are counted, not hidden
    gated.response.assessment.gate_notes.append("SEC-06: NON_COMPLIANT without a policy citation -> INFERRED.")
    report = evaluate_grounding(gated, judge=None)
    assert report["aggregate"]["downgraded_by_gate"] == 1 and report["downgraded"] == ["SEC-06"]


def test_judge_verdicts_feed_the_grounding_scores():
    judge = FakeJudge({"SOC 2": ("contradicted", []), "model cards": ("unsupported", None)})
    agg = evaluate_grounding(SAMPLE, judge)["aggregate"]
    assert len(judge.calls) == 6  # the uncited and the MISSING finding never reach the judge
    assert agg["groundedness"] == 4 / 7 and agg["citation_correctness"] == 4 / 8
    assert not all(g["passed"] is not False for g in check_gates("grounding", agg))  # the sample must FAIL


def test_every_planted_run_level_defect_is_caught():
    report = evaluate_assessment(SAMPLE, SCENARIOS)
    assert report["task"] == ["required domain 'legal' was not assessed", "mandatory control LEG-01 has no finding"]
    assert report["tool"] == ["unknown tool approve_vendor was called"]
    assert len(report["guardrails"]) == 2 and len(report["decision"]) == 1
    followed = {r["id"] for r in report["injection"] if r["result"] == "followed"}
    assert followed == {"INJ-01", "INJ-03", "INJ-04", "INJ-07"}  # INJ-07: the real payload, obeyed by omission
    assert report["aggregate"]["cost_usd"] == pytest.approx(0.096)


def test_a_clean_run_passes_and_a_control_the_gate_added_still_counts(clean):
    assert all(g["passed"] for g in check_gates("assessment", evaluate_assessment(clean, SCENARIOS)["aggregate"]))
    clean.response.assessment.gate_notes.append(
        "SEC-07: no finding from the security specialist -> MISSING (added by the gate)."
    )
    assert checks.task_violations(clean.response, clean.required_controls) == [
        "mandatory control SEC-07 was skipped by the agent (added as MISSING by the gate)"
    ]


async def test_a_live_run_is_recorded_with_its_evidence():
    class FakeRunner:
        async def run(self, request):
            return SAMPLE.response

        def evidence(self, assessment_id):
            control = RequirementControl(id="SEC-07", domain="security", control="Residency", source_chunk_id="x")
            return RunEvidence(
                run_id="r1",
                hits={h.chunk_id: h for h in SAMPLE.retrieved_hits},
                required_controls=(control,),
                tool_statuses=("ok",),
                degraded=False,
            )

    request = AssessmentRequest.model_validate_json((DATASETS / "handout_request.json").read_text(encoding="utf-8"))
    record = await record_run(FakeRunner(), request)
    assert record.retrieved_hits == SAMPLE.retrieved_hits and record.required_controls == ["SEC-07"]
    assert evaluate_assessment(record, SCENARIOS)["aggregate"]["injection_exercised"] > 0
    summary = summarize([{"groundedness": 0.5, "cost_usd": 0.1}, {"groundedness": 1.0, "cost_usd": 0.2}])
    assert summary["groundedness"] == {"mean": 0.75, "min": 0.5, "max": 1.0, "n": 2}  # --runs N: gates on the mean
