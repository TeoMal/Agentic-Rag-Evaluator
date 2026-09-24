"""Langfuse tracing: what a trace carries, that the runner feeds it, and that tracing can never
break a run (no keys, a broken client, or an unreachable Langfuse server)."""

from contextlib import contextmanager

from langchain_core.callbacks import BaseCallbackHandler
from test_agents import REQUEST, ScriptedModel, _runner, _script

from hackathon2.observability import NULL_TRACE, Trace, Tracer, _LiveTrace, evaluation_scores, run_scores, run_summary
from hackathon2.schemas import AssessmentResponse, HumanDecision

LANGFUSE_KEYS = {
    "langfuse_public_key": "pk-lf-test",
    "langfuse_secret_key": "sk-lf-test",
    "langfuse_host": "http://127.0.0.1:1",  # nothing listens here
}


# --- configuration -------------------------------------------------------------------


def test_tracing_needs_both_keys_and_a_host(make_settings):
    assert make_settings(**LANGFUSE_KEYS).tracing_configured
    assert not make_settings().tracing_configured
    assert not make_settings(**{**LANGFUSE_KEYS, "langfuse_secret_key": ""}).tracing_configured
    assert not make_settings(**{**LANGFUSE_KEYS, "langfuse_host": None}).tracing_configured
    assert not make_settings(**LANGFUSE_KEYS, langfuse_tracing_enabled=False).tracing_configured


def test_without_keys_every_call_is_a_no_op(make_settings):
    tracer = Tracer.from_settings(make_settings())
    assert not tracer.enabled
    with tracer.assessment(REQUEST, "run1") as trace:
        assert trace is NULL_TRACE
        assert trace.trace_id is None and trace.callbacks == ()
        trace.finish(AssessmentResponse(assessment_id="a1", status="failed", error="boom"))
    tracer.score("abc", "human_decision", "approved")
    assert tracer.record_evaluation("assessment", {"llm_calls": 3}, [], True) is None
    assert tracer.trace_url("abc") is None
    tracer.flush()


def test_health_reports_tracing(make_settings):
    from fastapi.testclient import TestClient

    from hackathon2.service import create_app

    body = TestClient(create_app(make_settings(**LANGFUSE_KEYS))).get("/health").json()
    assert body["checks"]["tracing"] == "langfuse"


# --- what goes into a trace ------------------------------------------------------------


async def test_run_summary_and_scores_describe_the_outcome():
    response = await _runner(ScriptedModel(messages=iter(_script()))).run(REQUEST)
    summary = run_summary(response)
    assert summary["status"] == "awaiting_approval"
    assert summary["recommendation"] == "CONDITIONAL_APPROVAL"
    assert summary["human_approval"] == "pending"

    scores = {name: (value, kind) for name, value, kind in run_scores(response)}
    assert scores["run_status"] == ("awaiting_approval", "CATEGORICAL")
    assert scores["recommendation"] == ("CONDITIONAL_APPROVAL", "CATEGORICAL")
    assert scores["awaiting_human_review"] == (1.0, "BOOLEAN")
    assert scores["gate_blocked"] == (0.0, "BOOLEAN")


def test_a_failed_run_is_scored_without_an_assessment():
    scores = run_scores(AssessmentResponse(assessment_id="a1", status="failed", error="model down"))
    assert scores == [("run_status", "failed", "CATEGORICAL"), ("gate_blocked", 0.0, "BOOLEAN")]


def test_evaluation_scores_keep_numbers_and_skip_the_rest():
    aggregate = {"groundedness": 0.58, "uncited_material": 0, "citations": 22, "note": "x", "recall": None}
    scores = evaluation_scores("grounding", aggregate, passed=False)
    assert scores == [
        ("grounding.passed", 0.0, "BOOLEAN"),
        ("grounding.groundedness", 0.58, "NUMERIC"),
        ("grounding.uncited_material", 0.0, "NUMERIC"),
        ("grounding.citations", 22.0, "NUMERIC"),
    ]


# --- the runner feeds the trace --------------------------------------------------------


class LLMCounter(BaseCallbackHandler):
    """Counts LLM calls and remembers each tool run's name and parent, as a tracer would see them."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.tool_runs: dict = {}  # run_id -> (name, parent_run_id)

    def on_chat_model_start(self, *args, **kwargs) -> None:
        self.llm_calls += 1

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs) -> None:
        self.tool_runs[run_id] = ((serialized or {}).get("name") or kwargs.get("name"), parent_run_id)


class FakeTrace(Trace):
    def __init__(self) -> None:
        self.trace_id = "0123456789abcdef0123456789abcdef"
        self.callbacks = [LLMCounter()]
        self.finished: list[AssessmentResponse] = []

    def finish(self, response: AssessmentResponse) -> None:
        self.finished.append(response)


class FakeTracer(Tracer):
    def __init__(self) -> None:
        super().__init__()
        self.trace = FakeTrace()
        self.scores: list[tuple] = []

    @contextmanager
    def assessment(self, request, run_id):
        yield self.trace

    def score(self, trace_id, name, value, *, data_type=None, comment=None) -> None:
        self.scores.append((trace_id, name, value))


async def test_runner_traces_the_run_and_the_human_decision():
    tracer = FakeTracer()
    runner = _runner(ScriptedModel(messages=iter(_script())))
    runner._tracer = tracer

    response = await runner.run(REQUEST)

    assert response.metrics.trace_id == tracer.trace.trace_id
    assert tracer.trace.finished == [response]
    # The trace's handler saw every LLM call of the orchestrator AND its specialists.
    counter = tracer.trace.callbacks[0]
    assert counter.llm_calls == response.metrics.llm_calls > 0
    # One tool observation per call: the instrumentation wrapper is traced, the tool it wraps is not.
    nested = [name for name, parent in counter.tool_runs.values() if parent in counter.tool_runs]
    assert nested == []
    assert "search_vendor_documents" in [name for name, _ in counter.tool_runs.values()]

    await runner.decide(response.assessment_id, HumanDecision(approved=False, reviewer="cro@northstar"))
    assert tracer.scores == [(tracer.trace.trace_id, "human_decision", "rejected")]


# --- tracing can never break a run ------------------------------------------------------


class ExplodingSpan:
    trace_id = "f" * 32

    def update(self, **kwargs):
        raise RuntimeError("langfuse is down")


def test_a_failing_trace_update_is_swallowed():
    _LiveTrace(client=None, span=ExplodingSpan(), callbacks=()).finish(
        AssessmentResponse(assessment_id="a1", status="completed")
    )


class ExplodingClient:
    def start_as_current_observation(self, **kwargs):
        raise RuntimeError("langfuse is down")

    def create_score(self, **kwargs):
        raise RuntimeError("langfuse is down")

    def flush(self):
        raise RuntimeError("langfuse is down")


async def test_a_broken_client_leaves_the_run_untraced():
    tracer = Tracer(ExplodingClient(), "pk-lf-test")
    runner = _runner(ScriptedModel(messages=iter(_script())))
    runner._tracer = tracer

    response = await runner.run(REQUEST)

    assert response.status == "awaiting_approval", response.error
    assert response.metrics.trace_id is None
    await runner.decide(response.assessment_id, HumanDecision(approved=True, reviewer="cro@northstar"))
    assert tracer.record_evaluation("assessment", {"llm_calls": 3}, [], True, trace_id="abc") is None
    tracer.flush()


async def test_an_unreachable_langfuse_server_does_not_affect_the_run(make_settings):
    tracer = Tracer.from_settings(make_settings(**LANGFUSE_KEYS))
    assert tracer.enabled
    runner = _runner(ScriptedModel(messages=iter(_script())))
    runner._tracer = tracer
    try:
        response = await runner.run(REQUEST)
        assert response.status == "awaiting_approval", response.error
        assert len(response.metrics.trace_id) == 32  # a real OpenTelemetry trace id
    finally:
        tracer._client.shutdown()
