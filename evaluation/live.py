"""Live assessments, recorded and evaluated -- the whole pipeline scored end to end:

    uv run python -m evaluation.live                          # the handout's request (datasets/handout_request.json)
    uv run python -m evaluation.live --runs 3                 # three runs, each saved, plus their mean
    uv run python -m evaluation.live --request hidden.json    # any AssessmentRequest, e.g. the hidden vendor case
    uv run python -m evaluation.live --no-llm                 # grounding without the LLM judge (code checks only)

Each run goes through the AssessmentRunner (real model, MCP server, RAG, decision gate); its record --
response, every retrieved chunk with its text, the mandatory controls -- is saved to
evaluation-results/run-<assessment_id>.json and scored by the `assessment` and `grounding` suites.
With Langfuse configured, each run is traced and both suites' scores land on that run's trace.
One run has only a dozen or so material findings, so a single finding moves groundedness by ~7 points:
with --runs N the gates are judged on the mean, saved as live-summary-<time>.json.
Exit code: 0 all gates passed, 1 a gate failed, 2 the run could not start.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from statistics import mean

from evaluation import checks
from evaluation.judge import Judge
from evaluation.run import (
    DATASETS,
    RESULTS,
    RunRecord,
    check_gates,
    evaluate_assessment,
    evaluate_grounding,
    finish,
    load_cases,
    save,
)
from hackathon2 import observability
from hackathon2.agents.runner import AssessmentRunner
from hackathon2.llm import LLMNotConfiguredError
from hackathon2.schemas import AssessmentRequest

SUMMARY_METRICS = (
    "task_violations",
    "tool_violations",
    "guardrail_violations",
    "injection_followed",
    "decision_violations",
    "duration_seconds",
    "llm_calls",
    "cost_usd",
    "material_findings",
    "downgraded_by_gate",
    "groundedness",
    "citation_correctness",
    "uncited_material",
)


async def record_run(runner: AssessmentRunner, request: AssessmentRequest) -> RunRecord:
    """Run one assessment and keep what the evaluation needs to check it."""
    response = await runner.run(request)
    evidence = runner.evidence(response.assessment_id)
    return RunRecord(
        description=f"Live run: {request.vendor_name} ({request.use_case}), status {response.status}.",
        response=response,
        retrieved_hits=list(evidence.hits.values()) if evidence else [],
        required_controls=[c.id for c in evidence.required_controls if c.mandatory] if evidence else None,
    )


def summarize(runs: list[dict]) -> dict:
    """Mean, min and max of each metric over the runs; a metric a run did not measure is left out of it."""
    summary = {}
    for metric in SUMMARY_METRICS:
        values = [run[metric] for run in runs if run.get(metric) is not None]
        if values:
            summary[metric] = {"mean": round(mean(values), 4), "min": min(values), "max": max(values), "n": len(values)}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.live", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--request", type=Path, default=DATASETS / "handout_request.json", help="AssessmentRequest JSON"
    )
    parser.add_argument("--runs", type=int, default=1, help="repeat the run and judge the gates on the mean")
    parser.add_argument("--no-llm", action="store_true", help="grounding: code checks only")
    args = parser.parse_args(argv)

    try:
        request = AssessmentRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
        judge = None if args.no_llm else Judge.from_settings()
    except (ValueError, OSError, LLMNotConfiguredError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    scenarios = load_cases("injection_scenarios.json", checks.InjectionScenario)
    RESULTS.mkdir(exist_ok=True)
    runs, passed = [], True
    for n in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n######## live run {n} of {args.runs} ########")
        record = asyncio.run(record_run(AssessmentRunner(), request))
        path = RESULTS / f"run-{record.response.assessment_id}.json"
        path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"run record: {path} (status {record.response.status})\n")
        assessment, grounding = evaluate_assessment(record, scenarios), evaluate_grounding(record, judge)
        passed = finish("assessment", assessment, trace_id=record.trace_id) and passed
        passed = finish("grounding", grounding, trace_id=record.trace_id) and passed
        runs.append(
            {**assessment["aggregate"], **grounding["aggregate"], "assessment_id": record.response.assessment_id}
        )
    observability.get_tracer().flush()
    if args.runs == 1:
        return 0 if passed else 1

    summary = summarize(runs)
    means = {metric: values["mean"] for metric, values in summary.items()}
    gates = check_gates("assessment", means) + check_gates("grounding", means)
    print(f"\n======== mean of {args.runs} live runs ========")
    for metric, values in summary.items():
        print(f"  {metric:<22} mean {values['mean']:>10.3f}   min {values['min']:>9.3f}   max {values['max']:>9.3f}")
    print()
    for g in gates:
        value = "not measured" if g["value"] is None else f"{g['value']:.3f}"
        mark = {True: "PASS", False: "FAIL", None: "n/a "}[g["passed"]]
        print(f"[{mark}] {g['metric']:<22} {value:>12}   (gate {g['op']} {g['threshold']:g}, on the mean)")
    print(f"results: {save('live-summary', {'aggregate': summary, 'runs': runs}, gates)}")
    return 0 if all(g["passed"] is not False for g in gates) else 1


if __name__ == "__main__":
    sys.exit(main())
