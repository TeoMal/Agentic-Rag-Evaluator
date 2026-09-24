"""The orchestrator deep agent (FR02 planning, delegation to the specialists, FR11 synthesis).

create_deep_agent gives it `write_todos` (the plan), a virtual filesystem for working notes and the
`task` tool for delegating to the specialists. It reads the NFS decision rules and precedents
itself, runs the specialists in two phases (commercial last, so it can price the compliant
configuration) and ends by returning a FinalDecision through structured output.

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

from hackathon2.agents.prompts import PHASE2_NONE, PHASE2_PROCUREMENT, orchestrator_prompt
from hackathon2.agents.specialists import SPECIALISTS, build_specialists
from hackathon2.agents.tools import ORCHESTRATOR_TOOLS, pick_tools
from hackathon2.schemas import AssessmentRequest, Condition, Domain, Evidence, Recommendation, Severity

ORCHESTRATOR_NAME = "vendor-risk-orchestrator"

# Domains whose conditions can change the price; the commercial assessment runs after them.
COMMERCIAL_DOMAIN: Domain = "procurement"


class FinalDecision(BaseModel):
    """The orchestrator's synthesis across all domain reports."""

    recommendation: Recommendation
    risk_rating: Severity = Field(description="low, medium or high; never lower than the highest domain rating.")
    decision_basis: list[Evidence] = Field(
        min_length=1,
        description="The NFS policy rules (and any precedents) the recommendation rests on, cited from chunks "
        "retrieved in this run: at least the decision-outcome rule and every mandatory rule that drove it.",
    )
    conditions: list[Condition] = Field(
        default_factory=list,
        description="One entry per gap to close before (or shortly after) go-live, with the control_ids it closes.",
    )
    executive_summary: str = Field(max_length=2400)


def _phase_lines(domains: Sequence[Domain]) -> tuple[str, str]:
    phase1 = [d for d in domains if d != COMMERCIAL_DOMAIN]
    lines = "\n".join(f'   - {SPECIALISTS[d].name}: domain "{d}"' for d in phase1) or "   (none)"
    if COMMERCIAL_DOMAIN in domains:
        phase2 = PHASE2_PROCUREMENT.format(name=SPECIALISTS[COMMERCIAL_DOMAIN].name)
    else:
        phase2 = PHASE2_NONE
    return lines, phase2


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
    phase1_lines, phase2_text = _phase_lines(domains)
    return create_deep_agent(
        model=model,
        tools=pick_tools(tools, ORCHESTRATOR_TOOLS),
        system_prompt=orchestrator_prompt(phase1_lines, phase2_text),
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


def with_decision_basis(summary: str, basis: Sequence[Evidence], limit: int = 3000) -> str:
    """The executive summary plus a code-generated 'Decision basis' line listing the cited rules,
    trimmed to the schema's length limit."""
    if not basis:
        return summary[:limit]
    refs = "; ".join(f"{e.section or e.source} ({e.chunk_id})" for e in basis)
    text = f"{summary.rstrip()}\n\nDecision basis: {refs}"
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
