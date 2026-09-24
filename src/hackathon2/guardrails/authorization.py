"""Tool decisions only. Callers execute only allowed, unchanged arguments.

No authentication, approval token handling, MCP behavior or side effects live here.
Call immediately before execution; the caller must prevent argument/context changes
and make approval consumption/execution atomic where required.
"""

from collections.abc import Mapping
from types import MappingProxyType

from .policy import (
    DEFAULT_LIMITS,
    DEFAULT_PRIVACY_POLICY,
    ApprovalVerifier,
    CallerContext,
    Decision,
    Limits,
    PrivacyPolicy,
    Reason,
    ToolRule,
    _decision,
    _PayloadError,
    _same_data,
    _snapshot,
)
from .privacy import check_privacy


def _caller_snapshot(context: CallerContext, limits: Limits) -> CallerContext:
    if not isinstance(context, CallerContext):
        raise _PayloadError(Reason.MISSING_IDENTITY)
    if any(type(v) is not str or not v.strip() for v in (context.principal_id, context.run_id)):
        raise _PayloadError(Reason.MISSING_IDENTITY)
    if type(context.permissions) is not frozenset or not isinstance(context.allowed_resources, Mapping):
        raise _PayloadError(Reason.INVALID_INPUT)
    if len(context.permissions) > limits.max_nodes or len(context.allowed_resources) > limits.max_nodes:
        raise _PayloadError(Reason.LIMIT_EXCEEDED)
    resources = {}
    resource_count = 0
    for key, values in context.allowed_resources.items():
        if type(values) is not frozenset or len(values) > limits.max_nodes:
            raise _PayloadError(Reason.INVALID_INPUT)
        resource_count += len(values) + 1
        if resource_count > limits.max_nodes:
            raise _PayloadError(Reason.LIMIT_EXCEEDED)
        if type(key) is not str or not key or any(type(v) is not str or not v for v in values):
            raise _PayloadError(Reason.INVALID_INPUT)
        resources[key] = list(values)
    if any(type(p) is not str or not p for p in context.permissions):
        raise _PayloadError(Reason.INVALID_INPUT)
    _snapshot([context.principal_id, context.run_id, list(context.permissions), resources], limits)
    return CallerContext(
        context.principal_id,
        context.run_id,
        context.permissions,
        MappingProxyType({key: frozenset(values) for key, values in resources.items()}),
    )


def _unchanged_arguments(checked: object, original: object, limits: Limits) -> bool:
    # Revalidate bounds and supported types after callbacks (including new cycles).
    try:
        return _same_data(_snapshot(checked, limits, tool_payload=True), original)
    except _PayloadError:
        return False


def authorize_tool_call(
    tool_name: str,
    arguments: dict[str, object],
    *,
    context: CallerContext,
    rules: Mapping[str, ToolRule],
    verifier: ApprovalVerifier | None = None,
    privacy_policy: PrivacyPolicy = DEFAULT_PRIVACY_POLICY,
    limits: Limits = DEFAULT_LIMITS,
) -> Decision:
    """Fail closed. Verifier/checker truthy objects are not accepted as True.

    Verifiers receive a detached argument copy and immutable context snapshot.
    Approval is out of band: never put a token in model-generated tool arguments.
    Arguments must be built-in dicts with string keys, lists, strings, booleans,
    bounded integers, finite floats or None. Tuples, model objects, datetimes and
    subclasses are denied, not converted. Callers must explicitly serialize model
    objects before this boundary and execute the same payload they submit here.
    Callback changes are compared recursively with exact types and float values.
    """
    try:
        if not isinstance(limits, Limits) or type(tool_name) is not str or not tool_name.strip():
            return _decision("deny", Reason.INVALID_INPUT)
        _snapshot(tool_name, limits)
        if not isinstance(rules, Mapping) or tool_name not in rules:
            return _decision("deny", Reason.UNKNOWN_TOOL)
        rule = rules[tool_name]
        if not isinstance(rule, ToolRule):
            return _decision("deny", Reason.INVALID_INPUT)
        caller = _caller_snapshot(context, limits)
        if not rule.required_permissions <= caller.permissions:
            return _decision("deny", Reason.PERMISSION_DENIED)
        if type(arguments) is not dict:
            return _decision("deny", Reason.INVALID_ARGUMENTS)
        args = _snapshot(arguments, limits, tool_payload=True)
        if not rule.required_arguments <= args.keys() or not args.keys() <= rule.allowed_arguments:
            return _decision("deny", Reason.INVALID_ARGUMENTS)
        for key in rule.scope_arguments:
            if type(args[key]) is not str or args[key] not in caller.allowed_resources.get(key, frozenset()):
                return _decision("deny", Reason.SCOPE_MISMATCH)
        privacy = check_privacy(args, policy=privacy_policy, limits=limits)
        if privacy.outcome != "allow":
            return privacy
        checked = _snapshot(args, limits, tool_payload=True)
        try:
            valid = rule.check_arguments(checked)
            if valid is not True or not _unchanged_arguments(checked, args, limits):
                return _decision("deny", Reason.INVALID_ARGUMENTS)
        except Exception:  # noqa: BLE001 - arbitrary checker failure must deny without logging its payload.
            return _decision("deny", Reason.CHECKER_FAILURE)
        if rule.side_effect or tool_name == "record_assessment":
            if verifier is None:
                return _decision("require_review", Reason.APPROVAL_REQUIRED)
            checked = _snapshot(args, limits, tool_payload=True)
            try:
                approved = verifier.verify(tool_name=tool_name, arguments=checked, context=caller)
                if approved is not True or not _unchanged_arguments(checked, args, limits):
                    return _decision("deny", Reason.APPROVAL_INVALID)
            except Exception:  # noqa: BLE001 - arbitrary verifier failure must deny without logging its payload.
                return _decision("deny", Reason.VERIFIER_FAILURE)
        return Decision("allow")
    except _PayloadError as exc:
        return _decision("deny", exc.reason)
    except Exception:  # noqa: BLE001 - trusted adapters/mappings may fail; never turn failure into permission.
        return _decision("deny", Reason.CHECKER_FAILURE)
