"""Minimal configurable sensitive-marker checks. Never redact exact evidence."""

from .policy import (
    DEFAULT_LIMITS,
    DEFAULT_PRIVACY_POLICY,
    Decision,
    Limits,
    PrivacyPolicy,
    Reason,
    _decision,
    _PayloadError,
    _snapshot,
)


def check_privacy(
    value: object, *, policy: PrivacyPolicy = DEFAULT_PRIVACY_POLICY, limits: Limits = DEFAULT_LIMITS
) -> Decision:
    """Inspect nested ordinary data, including keys. Return no matched values.

    This returns a decision only: input and citation quotes are never rewritten.
    Empty marker/literal tuples explicitly disable detection, not input limits.
    """
    if not isinstance(policy, PrivacyPolicy):
        return _decision("deny", Reason.INVALID_INPUT)
    try:
        payload = _snapshot(value, limits)
    except _PayloadError as exc:
        return _decision("deny", exc.reason)
    except Exception:  # noqa: BLE001 - fail closed without exposing arbitrary payload/exception text.
        return _decision("deny", Reason.INVALID_INPUT)

    def sensitive(item: object) -> bool:
        if type(item) is str:
            return any(marker in item for marker in (*policy.markers, *policy.literals))
        if type(item) is dict:
            return any(sensitive(key) or sensitive(child) for key, child in item.items())
        if type(item) is list:
            return any(sensitive(child) for child in item)
        return False

    return _decision("deny", Reason.SENSITIVE_DATA) if sensitive(payload) else Decision("allow")
