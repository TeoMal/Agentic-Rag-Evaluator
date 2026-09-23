"""One vendor assessment, end to end (the entry point for the API and the evaluation suite).

    runner = AssessmentRunner()
    response = await runner.run(request)                      # FR01 -> FR12
    response = await runner.decide(assessment_id, decision)   # FR13, when status == "awaiting_approval"

Flow of run():
    tools (stub or MCP) -> instrument for this run -> orchestrator deep agent (plans with write_todos,
    delegates to specialists) -> DomainReports read back from the conversation -> missing domains
    filled as MISSING -> AssessmentDraft (LLM decision + reports verbatim) -> decision gate -> response.

run() never raises for agent, tool or model failures: it returns status "failed" with the error (FR15).
Results are kept in memory; persistence is up to the API layer.
"""

import logging
import time
import uuid
from collections.abc import Iterable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from hackathon2.agents.collect import collect_domain_reports, missing_domain_report, subagents_called
from hackathon2.agents.config import ALL_DOMAINS, AgentSettings, get_agent_settings
from hackathon2.agents.context import RunContext, UsageCallback, content_to_text, instrument_tool, parse_tool_result
from hackathon2.agents.gate_fallback import GateFn, resolve_gate
from hackathon2.agents.orchestrator import FinalDecision, build_orchestrator, render_request
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
)

logger = logging.getLogger(__name__)


class AssessmentRunner:
    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        tools_provider: ToolsProvider | None = None,
        domains: Sequence[Domain] = ALL_DOMAINS,
        middleware: Iterable = (),
        subagent_middleware: Iterable = (),
        gate: GateFn | None = None,
        settings: AgentSettings | None = None,
    ) -> None:
        self.settings = settings or get_agent_settings()
        self.domains: tuple[Domain, ...] = tuple(domains)
        self._model = model
        self._tools_provider = tools_provider or default_provider(self.settings)
        self._middleware = tuple(middleware)
        self._subagent_middleware = tuple(subagent_middleware)
        self._gate = gate or resolve_gate()
        self._results: dict[str, AssessmentResponse] = {}

    # --- public API ---------------------------------------------------------------------

    async def run(self, request: AssessmentRequest) -> AssessmentResponse:
        ctx = RunContext(request=request)
        started = time.perf_counter()
        try:
            tools = {tool.name: instrument_tool(tool, ctx) for tool in await self._tools_provider()}
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
            executive_summary=decision.executive_summary,
        )
        degraded = ctx.degraded or len(reports) < len(self.domains)
        assessment = self._gate(draft, request, set(ctx.retrieved_chunk_ids), degraded)
        response = AssessmentResponse(
            assessment_id=assessment.assessment_id,
            status="awaiting_approval" if assessment.human_approval == "pending" else "completed",
            assessment=assessment,
            metrics=ctx.metrics(time.perf_counter() - started, delegated),
        )
        return self._store(response)

    async def decide(self, assessment_id: str, decision: HumanDecision) -> AssessmentResponse:
        """Apply a human reviewer's decision to an assessment that is awaiting approval (FR13)."""
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
            await self._record(assessment, reviewer=decision.reviewer)
        response = current.model_copy(
            update={"assessment": assessment, "status": "completed" if decision.approved else "rejected_by_reviewer"}
        )
        return self._store(response)

    def get(self, assessment_id: str) -> AssessmentResponse | None:
        return self._results.get(assessment_id)

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

    async def _record(self, assessment: Assessment, *, reviewer: str) -> None:
        """Store the approved assessment through the restricted record_assessment tool.

        Called by code only, never by an agent. The token format is a placeholder until the MCP
        engineer defines how approval tokens are issued. A failure is logged, never raised.
        """
        try:
            tools = {tool.name: tool for tool in await self._tools_provider()}
            tool = tools.get("record_assessment")
            if tool is None:
                logger.warning("record_assessment unavailable; assessment %s not recorded", assessment.assessment_id)
                return
            raw = await tool.ainvoke(
                {"assessment": assessment.model_dump(mode="json"), "approval_token": f"human-review:{reviewer}"}
            )
            result = parse_tool_result(content_to_text(raw))
            if result is None or result.status != "ok":
                logger.warning("record_assessment did not succeed for %s: %s", assessment.assessment_id, raw)
        except Exception:
            logger.exception("record_assessment failed for %s", assessment.assessment_id)
