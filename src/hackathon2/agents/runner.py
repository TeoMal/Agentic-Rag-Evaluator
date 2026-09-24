"""One vendor assessment, end to end (the entry point for the API and the evaluation suite).

    runner = AssessmentRunner()
    response = await runner.run(request)                      # FR01 -> FR11
    response = await runner.decide(assessment_id, decision)   # FR12, when status == "awaiting_approval"

Flow of run():
    open the tools for this run (the MCP server by default) -> instrument them -> orchestrator deep
    agent (plans with write_todos, reads the NFS decision rules and precedents, delegates to specialists
    in two phases) -> DomainReports read back from the conversation -> missing domains filled as
    MISSING -> AssessmentDraft (LLM decision + reports verbatim + cited decision basis) -> the mandatory
    controls, fetched by code -> decision gate (gate.py: guardrails.gate_assessment on the run's
    evidence) -> response: "awaiting_approval" (human review), "completed" (allowed by the gate, and
    recorded) or "failed" (blocked by the gate, or the run broke).

decide() applies the human decision; an approval is recorded through recording.py.

run() never raises for agent, tool or model failures: it returns status "failed" with the error (FR14).
Results are kept in memory; persistence is up to the API layer.
"""

import logging
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from hackathon2.agents.collect import collect_domain_reports, missing_domain_report, subagents_called
from hackathon2.agents.config import ALL_DOMAINS, AgentSettings, get_agent_settings
from hackathon2.agents.context import RunContext, UsageCallback, content_to_text, instrument_tool, parse_tool_result
from hackathon2.agents.gate import GateResult, RunEvidence, apply_gate, required_controls_from
from hackathon2.agents.orchestrator import FinalDecision, build_orchestrator, render_request, with_decision_basis
from hackathon2.agents.recording import record
from hackathon2.agents.specialists import SPECIALISTS, SUBAGENT_DOMAINS
from hackathon2.agents.tools import ToolsProvider, default_provider
from hackathon2.llm import get_chat_model
from hackathon2.schemas import (
    Assessment,
    AssessmentDraft,
    AssessmentRequest,
    AssessmentResponse,
    Domain,
    HumanDecision,
    RequirementControl,
)

logger = logging.getLogger(__name__)

GateFn = Callable[[AssessmentDraft, AssessmentRequest, RunEvidence], GateResult]


class AssessmentRunner:
    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        tools_provider: ToolsProvider | None = None,
        domains: Sequence[Domain] = ALL_DOMAINS,
        middleware: Iterable = (),
        subagent_middleware: Iterable = (),
        gate: GateFn = apply_gate,
        settings: AgentSettings | None = None,
    ) -> None:
        self.settings = settings or get_agent_settings()
        self.domains: tuple[Domain, ...] = tuple(domains)
        self._model = model
        self._tools_provider = tools_provider or default_provider(self.settings)
        self._middleware = tuple(middleware)
        self._subagent_middleware = tuple(subagent_middleware)
        self._gate = gate
        self._results: dict[str, AssessmentResponse] = {}
        self._evidence: dict[str, RunEvidence] = {}

    # --- public API ---------------------------------------------------------------------

    async def run(self, request: AssessmentRequest) -> AssessmentResponse:
        ctx = RunContext(request=request)
        started = time.perf_counter()
        try:
            async with self._tools_provider() as raw_tools:
                tools = {tool.name: instrument_tool(tool, ctx) for tool in raw_tools}
                agent = build_orchestrator(
                    model=self._get_model(),
                    tools=tools,
                    domains=self.domains,
                    middleware=self._middleware,
                    subagent_middleware=self._subagent_middleware,
                )
                state = await agent.ainvoke(
                    {"messages": [HumanMessage(render_request(request, self.domains))]},
                    config={"recursion_limit": self.settings.recursion_limit, "callbacks": [UsageCallback(ctx)]},
                )
                controls = await self._required_controls({tool.name: tool for tool in raw_tools}, ctx)
        except Exception as exc:
            logger.exception("assessment run failed")
            return self._failed(ctx, started, [], f"{type(exc).__name__}: {exc}")

        messages = state.get("messages", [])
        delegated = subagents_called(messages)
        decision = state.get("structured_response")
        if not isinstance(decision, FinalDecision):
            return self._failed(ctx, started, delegated, "the orchestrator finished without a final decision")

        reports, problems = collect_domain_reports(messages, SUBAGENT_DOMAINS)
        for problem in problems:
            logger.warning("domain report problem: %s", problem)
        domains = []
        for domain in self.domains:
            report = reports.get(domain)
            if report is None:
                agent_name = SPECIALISTS[domain].name
                reason = "; ".join(p for p in problems if agent_name in p) or "the specialist returned no valid report"
                report = missing_domain_report(domain, reason)
            domains.append(report)

        draft = AssessmentDraft(
            recommendation=decision.recommendation,
            risk_rating=decision.risk_rating,
            domains=domains,
            conditions=decision.conditions,
            executive_summary=with_decision_basis(decision.executive_summary, decision.decision_basis),
        )
        evidence = RunEvidence(
            run_id=ctx.run_id,
            hits=dict(ctx.retrieved_hits),
            required_controls=controls,
            tool_statuses=tuple(ctx.tool_statuses),
            degraded=ctx.degraded or len(reports) < len(self.domains),
        )
        gated = self._gate(draft, request, evidence)
        assessment = gated.assessment
        self._evidence[assessment.assessment_id] = evidence
        error = None
        if gated.blocked:
            status = "failed"
            error = "; ".join(note for note in assessment.gate_notes if "BLOCKED" in note)
        elif assessment.human_approval == "pending":
            status = "awaiting_approval"
        else:  # allowed by the gate without review, e.g. an evidenced low-risk rejection
            status = "completed"
            await self._record(assessment, principal="decision-gate")
        response = AssessmentResponse(
            assessment_id=assessment.assessment_id,
            status=status,
            assessment=assessment,
            metrics=ctx.metrics(time.perf_counter() - started, delegated),
            error=error,
        )
        return self._store(response)

    async def decide(self, assessment_id: str, decision: HumanDecision) -> AssessmentResponse:
        """Apply a human reviewer's decision to an assessment that is awaiting approval (FR12)."""
        current = self._results.get(assessment_id)
        if current is None:
            raise KeyError(assessment_id)
        if current.status != "awaiting_approval" or current.assessment is None:
            raise ValueError(f"assessment {assessment_id} is not awaiting approval (status: {current.status})")

        assessment = current.assessment.model_copy(
            update={
                "human_approval": "approved" if decision.approved else "rejected",
                "reviewer": decision.reviewer,
                "reviewer_comment": decision.comment,
            }
        )
        if decision.approved:
            await self._record(assessment, principal=decision.reviewer)
        response = current.model_copy(
            update={"assessment": assessment, "status": "completed" if decision.approved else "rejected_by_reviewer"}
        )
        return self._store(response)

    def get(self, assessment_id: str) -> AssessmentResponse | None:
        return self._results.get(assessment_id)

    def evidence(self, assessment_id: str) -> RunEvidence | None:
        """What the run behind an assessment retrieved and checked -- for the evaluation's run records."""
        return self._evidence.get(assessment_id)

    # --- internals ----------------------------------------------------------------------

    def _get_model(self) -> BaseChatModel:
        if self._model is None:
            kwargs = {} if self.settings.temperature is None else {"temperature": self.settings.temperature}
            self._model = get_chat_model(**kwargs)
        return self._model

    def _store(self, response: AssessmentResponse) -> AssessmentResponse:
        self._results[response.assessment_id] = response
        return response

    def _failed(self, ctx: RunContext, started: float, delegated: list[str], error: str) -> AssessmentResponse:
        return self._store(
            AssessmentResponse(
                assessment_id=uuid.uuid4().hex[:12],
                status="failed",
                metrics=ctx.metrics(time.perf_counter() - started, delegated),
                error=error,
            )
        )

    async def _required_controls(
        self, tools: Mapping[str, BaseTool], ctx: RunContext
    ) -> tuple[RequirementControl, ...]:
        """The mandatory controls of every assessed domain, fetched by code (not by an LLM), so the gate
        checks the report against the whole checklist. A failed fetch counts as a tool failure."""
        tool = tools.get("get_policy_requirements")
        if tool is None:
            ctx.tool_statuses.append("unavailable")
            return ()
        results: list[dict] = []
        for domain in self.domains:
            try:
                parsed = parse_tool_result(content_to_text(await tool.ainvoke({"domain": domain})))
            except Exception:  # noqa: BLE001 -- a failed fetch is recorded, never raised
                parsed = None
            if parsed is None or parsed.status != "ok":
                ctx.tool_statuses.append(parsed.status if parsed is not None else "unavailable")
                continue
            results.extend(r for r in parsed.results if isinstance(r, dict))
        return required_controls_from(results)

    async def _record(self, assessment: Assessment, *, principal: str) -> None:
        """Store a final assessment through the restricted record_assessment tool (recording.py).
        Called by code only, never by an agent. A failure is logged, never raised."""
        try:
            async with self._tools_provider("system") as tools:
                problem = await record(assessment, {tool.name: tool for tool in tools}, principal=principal)
        except Exception as exc:  # noqa: BLE001
            problem = f"{type(exc).__name__}: {exc}"
        if problem:
            logger.warning("assessment %s not recorded: %s", assessment.assessment_id, problem)
        else:
            logger.info("assessment %s recorded by %s", assessment.assessment_id, principal)
