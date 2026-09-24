"""Record a final assessment in the NFS register -- the one side-effecting action, done by code only.

    problem = await record(assessment, tools, principal="cro@northstar")   # None = recorded

Three independent checks, so no single bug -- and no prompt-injected LLM -- can store an assessment
nobody approved:
1. our code signs the exact final assessment (mcp_server.auth.issue_approval_token: HMAC with
   MCP_APPROVAL_SECRET, bound to the assessment's id and content, expiring);
2. the guardrails authorize exactly this call (guardrails.authorize_tool_call) with a verifier that
   checks that signature against the payload about to be sent;
3. the MCP server's record_assessment verifies the token again before saving.

Nothing here raises: an assessment that could not be recorded is still the assessment, and the
reason is returned (and logged by the caller).
"""

from collections.abc import Mapping

from langchain_core.tools import BaseTool

from hackathon2.agents.context import content_to_text, parse_tool_result
from hackathon2.guardrails import CallerContext, ToolRule, authorize_tool_call
from hackathon2.mcp_server.auth import ApprovalConfigError, issue_approval_token, verify_approval_token
from hackathon2.schemas import Assessment

TOOL = "record_assessment"
PERMISSION = "record_assessment"


def _is_assessment(arguments: dict[str, object]) -> bool:
    Assessment.model_validate(arguments["assessment"])  # raises -> the guardrails deny (checker failure)
    return True


RULES = {
    TOOL: ToolRule(
        required_permissions=frozenset({PERMISSION}),
        required_arguments=frozenset({"assessment"}),
        allowed_arguments=frozenset({"assessment"}),
        scope_arguments=frozenset(),
        check_arguments=_is_assessment,
        side_effect=True,
    )
}


class SignedApproval:
    """guardrails.ApprovalVerifier: the approval token must be valid for exactly the payload being sent."""

    def __init__(self, token: str) -> None:
        self._token = token

    def verify(self, *, tool_name: str, arguments: dict[str, object], context: CallerContext) -> bool:
        if tool_name != TOOL:
            return False
        return verify_approval_token(self._token, Assessment.model_validate(arguments["assessment"])) is None


async def record(assessment: Assessment, tools: Mapping[str, BaseTool], *, principal: str) -> str | None:
    """Record `assessment` (approved by a human, or allowed by the gate); None on success, else why not."""
    tool = tools.get(TOOL)
    if tool is None:
        return f"{TOOL} is not offered by the current tool source"
    try:
        token = issue_approval_token(assessment)
    except (ApprovalConfigError, ValueError) as exc:
        return str(exc)

    arguments: dict[str, object] = {"assessment": assessment.model_dump(mode="json")}
    caller = CallerContext(principal, assessment.assessment_id, frozenset({PERMISSION}), {})
    decision = authorize_tool_call(TOOL, arguments, context=caller, rules=RULES, verifier=SignedApproval(token))
    if decision.outcome != "allow":
        return f"refused by the guardrails ({', '.join(r.value for r in decision.reasons)})"

    try:
        raw = await tool.ainvoke({**arguments, "approval_token": token})
    except Exception as exc:  # noqa: BLE001 -- a failed side effect is reported, never raised
        return f"{TOOL} failed: {type(exc).__name__}: {exc}"
    result = parse_tool_result(content_to_text(raw))
    if result is None:
        return f"{TOOL} returned an unreadable result"
    if result.status != "ok":
        return f"{TOOL} {result.status}: {result.error}"
    return None
