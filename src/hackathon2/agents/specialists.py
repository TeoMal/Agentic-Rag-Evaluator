"""The specialist subagents (FR07 risk domains; handout section 8): one per domain, reached through `task`.

Each specialist returns a DomainReport through structured output (ToolStrategy), so the orchestrator
receives validated JSON instead of free text. ToolStrategy is used rather than the provider's native
json_schema mode because the strict mode rejects the defaults and constraints in schemas.py.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import BaseTool

from hackathon2.agents.prompts import (
    AI_GOVERNANCE_FOCUS,
    LEGAL_FOCUS,
    PROCUREMENT_FOCUS,
    SECURITY_FOCUS,
    specialist_prompt,
)
from hackathon2.agents.tools import SPECIALIST_TOOLS, pick_tools
from hackathon2.schemas import Domain, DomainReport


@dataclass(frozen=True)
class Specialist:
    name: str  # the subagent_type the orchestrator passes to `task`
    domain: Domain
    title: str
    control_prefix: str
    description: str  # what the orchestrator reads when choosing whom to delegate to
    focus: str


SPECIALISTS: dict[Domain, Specialist] = {
    "security": Specialist(
        name="security-risk-agent",
        domain="security",
        title="Security Risk Agent",
        control_prefix="SEC",
        description="Security Risk Agent. Assesses the vendor's information-security controls (encryption, "
        "certifications, incident notification, access control) against NFS policy with cited evidence. "
        "Returns a DomainReport for the 'security' domain.",
        focus=SECURITY_FOCUS,
    ),
    "procurement": Specialist(
        name="procurement-finance-agent",
        domain="procurement",
        title="Procurement / Finance Agent",
        control_prefix="PROC",
        description="Procurement / Finance Agent. Assesses commercial terms: total cost of ownership (computed by "
        "a tool), budget fit and sourcing rules, with cited evidence. Returns a DomainReport for the "
        "'procurement' domain.",
        focus=PROCUREMENT_FOCUS,
    ),
    "legal": Specialist(
        name="legal-compliance-agent",
        domain="legal",
        title="Legal / Compliance Agent",
        control_prefix="LEG",
        description="Legal / Compliance Agent. Assesses data residency, cross-border transfers, sub-processors "
        "and contractual protections against NFS policy with cited evidence. Returns a DomainReport for the "
        "'legal' domain.",
        focus=LEGAL_FOCUS,
    ),
    "ai_governance": Specialist(
        name="ai-governance-agent",
        domain="ai_governance",
        title="AI Governance Agent",
        control_prefix="AIG",
        description="AI Governance Agent. Assesses use of NFS data for model training, model-provider "
        "transparency, audit logging and human oversight against the NFS AI policy with cited evidence. "
        "Returns a DomainReport for the 'ai_governance' domain.",
        focus=AI_GOVERNANCE_FOCUS,
    ),
}

# subagent name -> domain, used to read reports back out of the conversation.
SUBAGENT_DOMAINS: dict[str, Domain] = {s.name: s.domain for s in SPECIALISTS.values()}


def build_specialists(
    tools: Mapping[str, BaseTool],
    domains: Sequence[Domain],
    middleware: Iterable = (),
) -> list[dict]:
    """SubAgent specs for create_deep_agent(subagents=...), one per requested domain."""
    extra_middleware = list(middleware)
    specs: list[dict] = []
    for domain in domains:
        specialist = SPECIALISTS[domain]
        spec: dict = {
            "name": specialist.name,
            "description": specialist.description,
            "system_prompt": specialist_prompt(
                title=specialist.title,
                domain=specialist.domain,
                prefix=specialist.control_prefix,
                focus=specialist.focus,
            ),
            "tools": pick_tools(tools, SPECIALIST_TOOLS[domain]),
            "response_format": ToolStrategy(DomainReport),
        }
        if extra_middleware:
            spec["middleware"] = extra_middleware
        specs.append(spec)
    return specs
