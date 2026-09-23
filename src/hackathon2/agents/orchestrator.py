"""The orchestrator deep agent (FR02 planning, FR07 delegation, FR12 synthesis).

create_deep_agent gives it `write_todos` (the plan), a virtual filesystem for working notes and the
`task` tool for delegating to the specialists. It ends by returning a FinalDecision through
structured output.

FinalDecision deliberately does NOT contain the domain findings. The runner attaches the
specialists' DomainReports verbatim, so the orchestrator LLM cannot restate, soften or lose a
finding -- or alter a citation -- while summarising.
"""

from collections.abc import Iterable, Mapping, Sequence

from deepagents import create_deep_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from hackathon2.agents.prompts import orchestrator_prompt
from hackathon2.agents.specialists import SPECIALISTS, build_specialists
from hackathon2.agents.tools import ORCHESTRATOR_TOOLS, pick_tools
from hackathon2.schemas import AssessmentRequest, Condition, Domain, Recommendation, Severity

ORCHESTRATOR_NAME = "vendor-risk-orchestrator"


class FinalDecision(BaseModel):
    """The orchestrator's synthesis across all domain reports."""

    recommendation: Recommendation
    risk_rating: Severity = Field(description="Overall rating; never lower than the highest domain risk_rating.")
    conditions: list[Condition] = Field(
        default_factory=list,
        description="One entry per gap to close before (or after) go-live, with the control_ids it closes.",
    )
    executive_summary: str = Field(max_length=3000)


def build_orchestrator(
    *,
    model: BaseChatModel,
    tools: Mapping[str, BaseTool],
    domains: Sequence[Domain],
    middleware: Iterable = (),
    subagent_middleware: Iterable = (),
    checkpointer=None,
):
    """A compiled deep agent for one assessment run.

    `tools` are already instrumented for this run (context.instrument_tool). `middleware` and
    `subagent_middleware` are the hooks for the guardrails package.
    """
    specialist_lines = "\n".join(
        f'   - {SPECIALISTS[d].name}: domain "{d}"' for d in domains
    )
    return create_deep_agent(
        model=model,
        tools=pick_tools(tools, ORCHESTRATOR_TOOLS),
        system_prompt=orchestrator_prompt(specialist_lines),
        subagents=build_specialists(tools, domains, subagent_middleware),
        middleware=list(middleware),
        response_format=ToolStrategy(FinalDecision),
        checkpointer=checkpointer,
        name=ORCHESTRATOR_NAME,
    )


def render_request(request: AssessmentRequest, domains: Sequence[Domain]) -> str:
    """The first (and only) user message of a run: the structured request plus the domains to cover."""
    required = ", ".join(f"{d} ({SPECIALISTS[d].name})" for d in domains)
    return (
        "Vendor assessment request:\n"
        f"```json\n{request.model_dump_json(indent=2)}\n```\n"
        f"Required risk domains: {required}.\n"
        "Assess the vendor, identify material risks and recommend APPROVE, CONDITIONAL_APPROVAL or REJECT."
    )
