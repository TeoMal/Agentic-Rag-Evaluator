"""Langfuse tracing -- the handout's section 11 observability, with Langfuse in place of Azure Monitor.

One assessment run = one Langfuse trace:

    vendor-assessment                      root agent span: the request in, the outcome out (also the
                                           trace's input/output); ERROR level when the run failed
      +- LangChain callback observations   every LLM call (model, tokens, latency), every MCP tool call,
      |                                    and each specialist subagent (the deepagents `task` tool runs
      |                                    them inside the same trace)
      +- scores                            run_status, recommendation, risk_rating, gate_blocked,
                                           degraded, awaiting_human_review -- later human_decision
                                           (decide()) and the evaluation suites' metrics

    tracer = get_tracer()
    with tracer.assessment(request, run_id) as trace:        # trace.callbacks -> the agent's config
        ...
        trace.finish(response)
    tracer.score(trace_id, "human_decision", "approved")      # a later event on the same trace
    tracer.record_evaluation("grounding", aggregate, gates, passed, trace_id=...)
    tracer.flush()                                            # before a CLI process exits

Tracing is optional and never breaks a run: without keys (or with LANGFUSE_TRACING_ENABLED=false)
every call here is a no-op, and a Langfuse error is logged, never raised. If the Langfuse server is
down the SDK drops the spans in the background; the assessment itself is unaffected (FR14).
"""

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from typing import Any

from hackathon2.config import Settings, get_settings
from hackathon2.schemas import AssessmentRequest, AssessmentResponse

logger = logging.getLogger(__name__)

TRACE_NAME = "vendor-assessment"
EVALUATION_TRACE_PREFIX = "evaluation"
# Seconds the SDK waits on the Langfuse API: a slow or missing server must not stall a run.
REQUEST_TIMEOUT = 5


class Trace:
    """One assessment's trace. This base class is the no-op used when tracing is off."""

    trace_id: str | None = None
    callbacks: tuple = ()

    def finish(self, response: AssessmentResponse) -> None:
        """Attach the run's outcome: output, level and scores."""


NULL_TRACE = Trace()


class _LiveTrace(Trace):
    def __init__(self, client, span, callbacks: tuple) -> None:
        self._client = client
        self._span = span
        self.trace_id = span.trace_id
        self.callbacks = callbacks

    def finish(self, response: AssessmentResponse) -> None:
        try:
            output = run_summary(response)
            failed = response.status == "failed"
            self._span.update(
                output=output,
                metadata=_run_metadata(response),
                level="ERROR" if failed else None,
                status_message=response.error if failed else None,
            )
            for name, value, data_type in run_scores(response):
                self._span.score_trace(name=name, value=value, data_type=data_type)
        except Exception:
            logger.warning("langfuse: could not finish the trace", exc_info=True)


class Tracer:
    """Builds traces and scores on one Langfuse client; disabled (all no-ops) when client is None."""

    def __init__(self, client=None, public_key: str | None = None) -> None:
        self._client = client
        self._public_key = public_key

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @classmethod
    def from_settings(cls, settings: Settings) -> "Tracer":
        if not settings.tracing_configured:
            return cls()
        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key.get_secret_value(),
                base_url=settings.langfuse_host,
                environment=_environment(settings.app_env),
                release=settings.image_tag,
                timeout=REQUEST_TIMEOUT,
            )
        except Exception:
            logger.warning("langfuse: tracing disabled, the client could not be created", exc_info=True)
            return cls()
        logger.info("langfuse: tracing to %s", settings.langfuse_host)
        return cls(client, settings.langfuse_public_key)

    @contextmanager
    def assessment(self, request: AssessmentRequest, run_id: str) -> Iterator[Trace]:
        """The trace for one assessment run; its callbacks go into the agent's RunnableConfig."""
        if not self.enabled:
            yield NULL_TRACE
            return
        with ExitStack() as stack:
            try:
                from langfuse import propagate_attributes
                from langfuse.langchain import CallbackHandler

                stack.enter_context(
                    propagate_attributes(
                        trace_name=TRACE_NAME,
                        tags=[TRACE_NAME, request.vendor_id or request.vendor_name],
                        metadata={"run_id": run_id, "data_classification": request.data_classification},
                    )
                )
                span = stack.enter_context(
                    self._client.start_as_current_observation(
                        name=TRACE_NAME,
                        as_type="agent",
                        input=request.model_dump(mode="json"),
                        metadata={"run_id": run_id},
                    )
                )
                trace = _LiveTrace(self._client, span, (CallbackHandler(public_key=self._public_key),))
            except Exception:
                logger.warning("langfuse: could not start the trace, running untraced", exc_info=True)
                trace = NULL_TRACE
            yield trace

    def score(
        self,
        trace_id: str | None,
        name: str,
        value: float | str,
        *,
        data_type: str | None = None,
        comment: str | None = None,
    ) -> None:
        """A score on an existing trace, e.g. the human decision that arrives after the run."""
        if not (self.enabled and trace_id):
            return
        try:
            self._client.create_score(trace_id=trace_id, name=name, value=value, data_type=data_type, comment=comment)
        except Exception:
            logger.warning("langfuse: could not record score %s", name, exc_info=True)

    def record_evaluation(
        self,
        suite: str,
        aggregate: Mapping[str, Any],
        gates: Sequence[Mapping[str, Any]],
        passed: bool,
        *,
        trace_id: str | None = None,
    ) -> str | None:
        """An evaluation suite's results as Langfuse scores.

        With the trace_id of the assessment run it evaluated, the scores land on that run's trace;
        otherwise (retrieval, calibration, recorded sample runs) on a trace of their own,
        "evaluation:<suite>". Returns the id of the trace scored, or None when tracing is off.
        """
        if not self.enabled:
            return None
        scores = evaluation_scores(suite, aggregate, passed)
        try:
            if trace_id:
                for name, value, data_type in scores:
                    self._client.create_score(trace_id=trace_id, name=name, value=value, data_type=data_type)
                return trace_id
            with self._client.start_as_current_observation(
                name=f"{EVALUATION_TRACE_PREFIX}:{suite}",
                as_type="evaluator",
                input={"suite": suite},
                output={"aggregate": dict(aggregate), "gates": list(gates), "passed": passed},
                level=None if passed else "WARNING",
            ) as span:
                for name, value, data_type in scores:
                    span.score_trace(name=name, value=value, data_type=data_type)
                return span.trace_id
        except Exception:
            logger.warning("langfuse: could not record the %s evaluation", suite, exc_info=True)
            return None

    def trace_url(self, trace_id: str | None) -> str | None:
        if not (self.enabled and trace_id):
            return None
        try:
            return self._client.get_trace_url(trace_id=trace_id)
        except Exception:  # noqa: BLE001
            return None

    def flush(self) -> None:
        """Send what is buffered -- call before a short-lived process (CLI, evaluation) exits."""
        if not self.enabled:
            return
        try:
            self._client.flush()
        except Exception:
            logger.warning("langfuse: flush failed", exc_info=True)


@lru_cache
def get_tracer() -> Tracer:
    return Tracer.from_settings(get_settings())


# --------------------------------------------------------------------------------------
# What goes into a trace -- plain functions, so they are testable without Langfuse
# --------------------------------------------------------------------------------------


def run_summary(response: AssessmentResponse) -> dict:
    """The trace output: the outcome, not the whole report (that is in the API response)."""
    assessment = response.assessment
    return {
        "assessment_id": response.assessment_id,
        "status": response.status,
        "recommendation": assessment.recommendation if assessment else None,
        "risk_rating": assessment.risk_rating if assessment else None,
        "human_approval": assessment.human_approval if assessment else None,
        "degraded_mode": assessment.degraded_mode if assessment else None,
        "conditions": len(assessment.conditions) if assessment else 0,
        "gate_notes": list(assessment.gate_notes) if assessment else [],
        "error": response.error,
    }


def run_scores(response: AssessmentResponse) -> list[tuple[str, float | str, str]]:
    """(name, value, data_type) for every run-level score."""
    assessment = response.assessment
    scores: list[tuple[str, float | str, str]] = [
        ("run_status", response.status, "CATEGORICAL"),
        ("gate_blocked", _flag(response.status == "failed" and assessment is not None), "BOOLEAN"),
    ]
    if assessment is not None:
        scores += [
            ("recommendation", assessment.recommendation, "CATEGORICAL"),
            ("risk_rating", assessment.risk_rating, "CATEGORICAL"),
            ("degraded", _flag(assessment.degraded_mode), "BOOLEAN"),
            ("awaiting_human_review", _flag(response.status == "awaiting_approval"), "BOOLEAN"),
        ]
    return scores


def evaluation_scores(suite: str, aggregate: Mapping[str, Any], passed: bool) -> list[tuple[str, float | str, str]]:
    """The suite verdict plus every numeric aggregate metric, named "<suite>.<metric>".
    Unmeasured metrics (None) and non-numeric ones are left out."""
    scores: list[tuple[str, float | str, str]] = [(f"{suite}.passed", _flag(passed), "BOOLEAN")]
    for metric, value in aggregate.items():
        if isinstance(value, bool):
            scores.append((f"{suite}.{metric}", _flag(value), "BOOLEAN"))
        elif isinstance(value, int | float):
            scores.append((f"{suite}.{metric}", float(value), "NUMERIC"))
    return scores


def _run_metadata(response: AssessmentResponse) -> dict:
    metrics = response.metrics
    if metrics is None:
        return {}
    return {
        "duration_seconds": metrics.duration_seconds,
        "llm_calls": metrics.llm_calls,
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "tools_called": metrics.tools_called,
        "subagents_called": metrics.subagents_called,
        "retrieved_chunks": len(metrics.retrieved_chunk_ids),
    }


def _flag(value: bool) -> float:
    """Langfuse BOOLEAN scores are 1 / 0."""
    return 1.0 if value else 0.0


def _environment(app_env: str) -> str:
    """Langfuse environments: lowercase letters, digits, '-' and '_', not starting with 'langfuse'."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in app_env.lower()).strip("-_")
    return cleaned if cleaned and not cleaned.startswith("langfuse") else "default"
