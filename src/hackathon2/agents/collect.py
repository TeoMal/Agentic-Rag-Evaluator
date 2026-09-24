"""Reading the specialists' results back out of the orchestrator's conversation.

Each `task` call leaves an AIMessage tool call (with the subagent_type) and a ToolMessage (with the
subagent's DomainReport as JSON). They are paired by tool_call_id, validated, and checked against
the domain the subagent owns. A domain without a valid report gets a placeholder whose single
finding is MISSING (UNKNOWN in NFS policy) -- an unassessed domain is never a silent pass (FR10, FR14).
"""

from collections.abc import Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import ValidationError

from hackathon2.agents.context import content_to_text
from hackathon2.agents.specialists import SPECIALISTS
from hackathon2.schemas import Domain, DomainReport, Finding


def _task_calls(messages: Sequence[BaseMessage]) -> dict[str, str]:
    """tool_call_id -> subagent_type, for every `task` call in the conversation."""
    calls: dict[str, str] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] == "task" and call.get("id"):
                    calls[call["id"]] = str(call["args"].get("subagent_type", ""))
    return calls


def subagents_called(messages: Sequence[BaseMessage]) -> list[str]:
    """Subagent names in delegation order (RunMetrics.subagents_called, 'agent delegation' metric)."""
    names: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            names.extend(
                str(call["args"].get("subagent_type", "")) for call in message.tool_calls if call["name"] == "task"
            )
    return names


def _json_object(text: str) -> str:
    text = text.strip()
    if text.startswith("{"):
        return text
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


def collect_domain_reports(
    messages: Sequence[BaseMessage],
    expected: Mapping[str, Domain],
) -> tuple[dict[Domain, DomainReport], list[str]]:
    """Valid DomainReports by domain, plus a note for every report that was rejected.

    `expected` maps subagent name -> the domain it owns. If a domain was delegated more than once,
    the last valid report wins.
    """
    calls = _task_calls(messages)
    reports: dict[Domain, DomainReport] = {}
    problems: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.tool_call_id not in calls:
            continue
        agent = calls[message.tool_call_id]
        domain = expected.get(agent)
        if domain is None:
            problems.append(f"ignored output of unexpected subagent '{agent}'")
            continue
        try:
            report = DomainReport.model_validate_json(_json_object(content_to_text(message.content)))
        except (ValidationError, ValueError) as exc:
            problems.append(f"{agent}: report did not validate ({type(exc).__name__})")
            continue
        if report.domain != domain:
            problems.append(f"{agent}: returned a report for '{report.domain}' instead of '{domain}'")
            continue
        reports[domain] = report
    return reports, problems


def missing_domain_report(domain: Domain, reason: str) -> DomainReport:
    """Placeholder for a domain with no valid report: rated high, one MISSING finding."""
    prefix = SPECIALISTS[domain].control_prefix
    reason = reason.strip()[:600] or "the specialist did not return a valid report"
    return DomainReport(
        domain=domain,
        risk_rating="high",
        summary=f"This domain was not assessed: {reason}. It is recorded as UNKNOWN, not as passed.",
        findings=[
            Finding(
                domain=domain,
                control_id=f"{prefix}-00",
                title="Domain assessment not completed",
                status="MISSING",
                severity="high",
                claim=f"No validated {domain} assessment is available for this vendor ({reason}).",
                remediation="Re-run the assessment or complete a manual review of this domain before approval.",
            )
        ],
    )
