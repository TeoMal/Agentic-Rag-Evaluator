"""Guardrails-owned policy and interfaces. All context must come from trusted code.

Never construct caller permissions, provenance, or approval state from model text.
Reason values are stable public codes; diagnostics deliberately carry no payloads.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Literal, Protocol

from pydantic import BaseModel

from hackathon2.schemas import RequirementControl, SearchHit, ToolStatus


class Reason(StrEnum):
    INVALID_INPUT = "invalid_input"
    LIMIT_EXCEEDED = "limit_exceeded"
    INSTRUCTION_OVERRIDE = "instruction_override"
    EVIDENCE_OVERRIDE = "evidence_override"
    ASSESSMENT_MANIPULATION = "assessment_manipulation"
    ROLE_SPOOFING = "role_spoofing"
    SECRET_EXFILTRATION = "secret_exfiltration"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    FORGED_APPROVAL = "forged_approval"
    SCANNER_FAILURE = "scanner_failure"
    SUSPICIOUS_HIT = "suspicious_hit"
    RETRIEVAL_QUARANTINED = "retrieval_quarantined"
    CONFLICTING_CHUNK_ID = "conflicting_chunk_id"
    EVIDENCE_GAP = "evidence_gap"
    SENSITIVE_DATA = "sensitive_data"
    UNKNOWN_TOOL = "unknown_tool"
    MISSING_IDENTITY = "missing_identity"
    PERMISSION_DENIED = "permission_denied"
    INVALID_ARGUMENTS = "invalid_arguments"
    SCOPE_MISMATCH = "scope_mismatch"
    CHECKER_FAILURE = "checker_failure"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_INVALID = "approval_invalid"
    VERIFIER_FAILURE = "verifier_failure"
    INVALID_OUTPUT = "invalid_output"
    INCONSISTENT_ASSESSMENT = "inconsistent_assessment"
    INVALID_PROVENANCE = "invalid_provenance"
    CITATION_REQUIRED = "citation_required"
    UNKNOWN_CITATION = "unknown_citation"
    QUARANTINED_CITATION = "quarantined_citation"
    CITATION_MISMATCH = "citation_mismatch"
    HIGH_RISK = "high_risk"
    FINAL_APPROVAL = "final_approval"
    TOOL_FAILURE = "tool_failure"
    DEGRADED_MODE = "degraded_mode"
    REVIEW_REJECTED = "review_rejected"


@dataclass(frozen=True)
class Decision:
    outcome: Literal["allow", "deny", "require_review"]
    reasons: tuple[Reason, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in ("allow", "deny", "require_review"):
            raise ValueError("invalid decision")
        if type(self.reasons) is not tuple or any(not isinstance(r, Reason) for r in self.reasons):
            raise ValueError("invalid decision reasons")
        if (self.outcome == "allow") != (not self.reasons):
            raise ValueError("invalid decision reasons")


def _decision(outcome: Literal["deny", "require_review"], *reasons: Reason) -> Decision:
    return Decision(outcome, tuple(dict.fromkeys(reasons)))


@dataclass(frozen=True)
class Limits:
    max_text_chars: int = 16_384
    max_payload_chars: int = 262_144
    max_nodes: int = 10_000
    max_depth: int = 16
    max_hits: int = 100

    def __post_init__(self) -> None:
        ceilings = (1_000_000, 4_000_000, 100_000, 32, 1_000)
        for value, ceiling in zip(vars(self).values(), ceilings, strict=True):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError("invalid guardrail limit")


@dataclass(frozen=True)
class PrivacyPolicy:
    """Minimal marker/literal detection, not a general PII or secret classifier.

    Defaults identify synthetic fixtures only. Additional markers/literals are
    supplied explicitly by trusted callers; this library never reads credentials.
    """

    markers: tuple[str, ...] = field(default=("SYNTHETIC_SECRET_", "SYNTHETIC_API_KEY_"), repr=False)
    literals: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        for values in (self.markers, self.literals):
            if type(values) is not tuple or len(values) > 100:
                raise ValueError("invalid privacy policy")
            if any(type(v) is not str or not 1 <= len(v) <= 256 for v in values):
                raise ValueError("invalid privacy policy")


DEFAULT_LIMITS = Limits()
DEFAULT_PRIVACY_POLICY = PrivacyPolicy()


@dataclass(frozen=True)
class CallerContext:
    principal_id: str
    run_id: str
    permissions: frozenset[str]
    allowed_resources: Mapping[str, frozenset[str]] = field(repr=False)


@dataclass(frozen=True)
class ToolRule:
    """A trusted registry entry, never supplied by a model or retrieved document.

    Every argument name must be explicitly allowed. Scope arguments are required
    strings checked against CallerContext.allowed_resources under the same key.
    check_arguments must return exactly True and validate types/business bounds.
    Side effects require external approval verification; record_assessment always
    does, even if a rule accidentally labels it read-only.
    """

    required_permissions: frozenset[str]
    required_arguments: frozenset[str]
    allowed_arguments: frozenset[str]
    scope_arguments: frozenset[str]
    check_arguments: Callable[[dict[str, object]], bool] = field(repr=False)
    side_effect: bool = True

    def __post_init__(self) -> None:
        for values in (
            self.required_permissions,
            self.required_arguments,
            self.allowed_arguments,
            self.scope_arguments,
        ):
            if type(values) is not frozenset or any(type(v) is not str or not v.strip() for v in values):
                raise ValueError("invalid tool rule")
        if not self.required_arguments <= self.allowed_arguments or not self.scope_arguments <= self.required_arguments:
            raise ValueError("invalid tool rule")
        if not callable(self.check_arguments) or type(self.side_effect) is not bool:
            raise ValueError("invalid tool rule")


class ApprovalVerifier(Protocol):
    """Implemented by MCP/orchestration, with approval supplied out of band.

    Must check authenticated approval, exact action/arguments, run/resource and
    assessment version binding, expiration and replay protection. Return exactly
    True only for valid approval. Issuance, storage and atomic consumption with
    execution are the caller's responsibility, not implemented here.
    """

    def verify(self, *, tool_name: str, arguments: dict[str, object], context: CallerContext) -> bool: ...


class DocumentScanner(Protocol):
    def __call__(self, text: str, *, limits: Limits) -> Decision: ...


@dataclass(frozen=True)
class GateContext:
    """Run-scoped provenance snapshot supplied by RAG/orchestration.

    Ledger entries contain ORIGINAL text for exact quotes, not escaped presentation
    text. The owner must attest these are the run's actual retrieved sources.
    Required controls come from the reviewed checklist, not the draft. Explicitly
    passing an empty checklist means the caller has no required controls.
    Failure flags and tool statuses must include the whole run's relevant history.
    """

    run_id: str
    ledger_run_id: str
    ledger: Mapping[str, SearchHit] = field(repr=False)
    required_controls: tuple[RequirementControl, ...] = field(repr=False)
    quarantined_chunk_ids: frozenset[str] = field(default=frozenset(), repr=False)
    tool_statuses: tuple[ToolStatus, ...] = ()
    evidence_gap: bool = False
    scanner_failed: bool = False
    verifier_failed: bool = False


class _PayloadError(Exception):
    def __init__(self, reason: Reason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def _snapshot(value: object, limits: Limits, *, tool_payload: bool = False) -> object:
    """Bounded detached copy before serialization, including keys and cycles.

    Tool payloads accept exact built-in dict/list/str/bool/int/finite-float/None
    types only, with string keys. Tuples, models, datetimes and subclasses are
    rejected rather than normalized. The general mode is for inspection only,
    never for deciding whether unchanged execution arguments are authorized.
    """
    nodes = chars = 0

    def visit(item: object, depth: int) -> object:
        nonlocal nodes, chars
        nodes += 1
        if nodes > limits.max_nodes or depth > limits.max_depth:
            raise _PayloadError(Reason.LIMIT_EXCEEDED)
        if type(item) is str:
            chars += len(item)
            if len(item) > limits.max_text_chars or chars > limits.max_payload_chars:
                raise _PayloadError(Reason.LIMIT_EXCEEDED)
            return item
        if item is None or type(item) in (bool, int):
            if type(item) is int and item.bit_length() > 64:
                raise _PayloadError(Reason.LIMIT_EXCEEDED)
            return item
        if type(item) is float and isfinite(item):
            return item
        if type(item) is datetime and not tool_payload:
            return item
        if not tool_payload and isinstance(item, BaseModel):
            return visit(vars(item), depth + 1)
        if type(item) is dict:
            if len(item) > limits.max_nodes:
                raise _PayloadError(Reason.LIMIT_EXCEEDED)
            result = {}
            for key, child in item.items():
                if type(key) is not str:
                    raise _PayloadError(Reason.INVALID_ARGUMENTS if tool_payload else Reason.INVALID_INPUT)
                result[visit(key, depth + 1)] = visit(child, depth + 1)
            return result
        if type(item) is list or (type(item) is tuple and not tool_payload):
            if len(item) > limits.max_nodes:
                raise _PayloadError(Reason.LIMIT_EXCEEDED)
            return [visit(child, depth + 1) for child in item]
        raise _PayloadError(Reason.INVALID_ARGUMENTS if tool_payload else Reason.INVALID_INPUT)

    if not isinstance(limits, Limits):
        raise _PayloadError(Reason.INVALID_INPUT)
    return visit(value, 0)


def _same_data(left: object, right: object) -> bool:
    """Exact comparison of already bounded plain data, including scalar types.

    Snapshot untrusted callback output before calling this helper. Dictionary
    insertion order is not significant; list order and float sign are significant.
    """
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_same_data(value, right[key]) for key, value in left.items())
    if type(left) is list:
        return len(left) == len(right) and all(_same_data(a, b) for a, b in zip(left, right, strict=True))
    if type(left) is float:
        return left.hex() == right.hex()
    return left == right
