"""One live assessment, recorded and evaluated -- the whole pipeline scored end to end:

    uv run python -m evaluation.live                          # the handout's request (datasets/handout_request.json)
    uv run python -m evaluation.live --request hidden.json    # any AssessmentRequest, e.g. the hidden vendor case
    uv run python -m evaluation.live --no-llm                 # grounding without the LLM judge (code checks only)

Runs the AssessmentRunner (real model, MCP server, RAG, decision gate), saves the run record --
response, every retrieved chunk with its text, the mandatory controls -- to
evaluation-results/run-<assessment_id>.json, then runs the `assessment` and `grounding` suites on it.
With Langfuse configured, the run is traced and both suites' scores land on that same trace.
Exit code: 0 both suites passed, 1 a gate failed, 2 the run could not start.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from evaluation import checks
from evaluation.judge import Judge
from evaluation.run import DATASETS, RESULTS, RunRecord, evaluate_assessment, evaluate_grounding, finish, load_cases
from hackathon2 import observability
from hackathon2.agents.runner import AssessmentRunner
from hackathon2.llm import LLMNotConfiguredError
from hackathon2.schemas import AssessmentRequest


async def record_run(runner: AssessmentRunner, request: AssessmentRequest) -> RunRecord:
    """Run one assessment and keep what the evaluation needs to check it."""
    response = await runner.run(request)
    evidence = runner.evidence(response.assessment_id)
    return RunRecord(
        description=f"Live run: {request.vendor_name} ({request.use_case}), status {response.status}.",
        response=response,
        retrieved_hits=list(evidence.hits.values()) if evidence else [],
        required_controls=[c.id for c in evidence.required_controls] if evidence else None,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.live", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--request", type=Path, default=DATASETS / "handout_request.json", help="AssessmentRequest JSON"
    )
    parser.add_argument("--no-llm", action="store_true", help="grounding: code checks only")
    args = parser.parse_args(argv)

    try:
        request = AssessmentRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
        judge = None if args.no_llm else Judge.from_settings()
    except (ValueError, OSError, LLMNotConfiguredError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    record = asyncio.run(record_run(AssessmentRunner(), request))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"run-{record.response.assessment_id}.json"
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"run record: {path} (status {record.response.status})\n")

    scenarios = load_cases("injection_scenarios.json", checks.InjectionScenario)
    passed = finish("assessment", evaluate_assessment(record, scenarios), trace_id=record.trace_id)
    passed = finish("grounding", evaluate_grounding(record, judge), trace_id=record.trace_id) and passed
    observability.get_tracer().flush()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
