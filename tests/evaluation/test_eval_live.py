from evaluation import checks
from evaluation.live import record_run
from evaluation.run import DATASETS, RunRecord, evaluate_assessment, load_cases
from hackathon2.agents.gate import RunEvidence
from hackathon2.schemas import AssessmentRequest, RequirementControl

SAMPLE = RunRecord.load(DATASETS / "sample_run.json")
REQUEST = AssessmentRequest.model_validate_json((DATASETS / "handout_request.json").read_text(encoding="utf-8"))


class FakeRunner:
    """Stands in for AssessmentRunner: the sample run's response and evidence."""

    def __init__(self, with_evidence: bool = True) -> None:
        self.with_evidence = with_evidence

    async def run(self, request):
        return SAMPLE.response

    def evidence(self, assessment_id):
        if not self.with_evidence:
            return None
        control = RequirementControl(id="SEC-07", domain="security", control="Residency", source_chunk_id="x")
        hits = {h.chunk_id: h for h in SAMPLE.retrieved_hits}
        return RunEvidence(run_id="r1", hits=hits, required_controls=(control,), tool_statuses=("ok",), degraded=False)


async def test_a_live_run_is_recorded_with_its_evidence_and_can_be_scored():
    record = await record_run(FakeRunner(), REQUEST)
    assert record.retrieved_hits == SAMPLE.retrieved_hits
    assert record.required_controls == ["SEC-07"]
    assert "Asteria AI Systems" in record.description
    report = evaluate_assessment(record, load_cases("injection_scenarios.json", checks.InjectionScenario))
    assert report["aggregate"]["injection_exercised"] > 0  # the retrieved payloads reach the injection check


async def test_a_run_without_evidence_is_still_recorded():
    record = await record_run(FakeRunner(with_evidence=False), REQUEST)
    assert record.retrieved_hits == [] and record.required_controls is None
