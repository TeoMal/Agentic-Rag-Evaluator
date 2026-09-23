"""Screen supplied tool/peer data only; no adapters, execution or authority."""

import json
from dataclasses import dataclass, field
from html import unescape

from .injection import scan_document
from .policy import (
    DEFAULT_LIMITS,
    DEFAULT_PRIVACY_POLICY,
    Decision,
    DocumentScanner,
    Limits,
    PrivacyPolicy,
    Reason,
    _decision,
    _PayloadError,
    _snapshot,
)
from .privacy import check_privacy


@dataclass(frozen=True)
class ContentResult:
    decision: Decision
    presentation: str | None = field(default=None, repr=False)


def _strings(value: object):
    """Walk a previously bounded, detached JSON value, including dictionary keys."""
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for key, child in value.items():
            yield key
            yield from _strings(child)
    elif type(value) is list:
        for child in value:
            yield from _strings(child)


def prepare_untrusted_content(
    value: object,
    *,
    limits: Limits = DEFAULT_LIMITS,
    privacy_policy: PrivacyPolicy = DEFAULT_PRIVACY_POLICY,
    scanner: DocumentScanner = scan_document,
) -> ContentResult:
    """Screen supplied MCP/peer data without trusting any field as authority.

    Accept exact built-in dict (string keys), list, str, bool, int with bit_length
    <= 64, finite float, and None. Reject subclasses, tuples, models, datetimes,
    bytes and cycles; do not coerce or parse serialized/executable objects.
    Callers explicitly convert ToolResult models to plain data before calling.

    Scan every string/key with the shared scanner (one HTML-entity decode for
    screening only). Privacy checks cover raw data. On non-allow, presentation
    is None. Scanner errors/malformed responses deny without exception details.

    On allow, serialize the detached snapshot as sorted-key compact JSON with
    ensure_ascii=True and allow_nan=False. Escape literal <, >, & using JSON
    Unicode escapes, then enclose in <untrusted_content format="json"> tags.
    Input limits apply before serialization; the complete presentation must also
    fit max_payload_chars. Nothing is truncated. Preserve originals separately
    for provenance; this presentation is neither an exact quote nor an authority.
    Wrapping is a presentation aid, NOT an authorization boundary. Never promote
    role/approved/permissions/trusted fields into system or execution context.
    """
    try:
        payload = _snapshot(value, limits, tool_payload=True)
    except _PayloadError as exc:
        reason = Reason.INVALID_INPUT if exc.reason == Reason.INVALID_ARGUMENTS else exc.reason
        return ContentResult(_decision("deny", reason))
    except Exception:  # noqa: BLE001 - no caller data or exception details may escape this boundary.
        return ContentResult(_decision("deny", Reason.INVALID_INPUT))

    try:
        privacy = check_privacy(payload, policy=privacy_policy, limits=limits)
        if type(privacy) is not Decision:
            raise TypeError("invalid privacy result")
        privacy = Decision(privacy.outcome, privacy.reasons)
    except Exception:  # noqa: BLE001 - a failed checker cannot release content.
        return ContentResult(_decision("deny", Reason.CHECKER_FAILURE))
    if privacy.outcome != "allow":
        return ContentResult(privacy)

    try:
        if not callable(scanner):
            raise TypeError("invalid scanner")
        for text in _strings(payload):
            decision = scanner(unescape(text), limits=limits)
            if type(decision) is not Decision:
                raise TypeError("invalid scanner result")
            decision = Decision(decision.outcome, decision.reasons)
            if decision.outcome != "allow":
                return ContentResult(decision)
    except Exception:  # noqa: BLE001 - injected scanners can fail with arbitrary exceptions.
        return ContentResult(_decision("deny", Reason.SCANNER_FAILURE))

    try:
        body = json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        body = body.replace("&", r"\u0026").replace("<", r"\u003c").replace(">", r"\u003e")
        presentation = '<untrusted_content format="json">' + body + "</untrusted_content>"
        if len(presentation) > limits.max_payload_chars:
            return ContentResult(_decision("deny", Reason.LIMIT_EXCEEDED))
    except Exception:  # noqa: BLE001 - fail closed on serialization without exposing source content.
        return ContentResult(_decision("deny", Reason.CHECKER_FAILURE))
    return ContentResult(Decision("allow"), presentation)
