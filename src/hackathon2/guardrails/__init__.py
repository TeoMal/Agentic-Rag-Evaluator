"""Deterministic safety decisions; no I/O, tool execution, or workflow state.

Callers must enforce decisions at every applicable boundary. An ``allow`` is
permission for that boundary only, never proof of factual correctness or approval.
"""

from .authorization import authorize_tool_call
from .gate import gate_assessment
from .injection import scan_document
from .policy import (
    DEFAULT_LIMITS,
    DEFAULT_PRIVACY_POLICY,
    ApprovalVerifier,
    CallerContext,
    Decision,
    DocumentScanner,
    GateContext,
    Limits,
    PrivacyPolicy,
    Reason,
    ToolRule,
)
from .privacy import check_privacy
from .retrieval import UNTRUSTED_CONTENT_INSTRUCTION, QuarantinedHit, RetrievalResult, prepare_retrieved_hits

__all__ = [
    "DEFAULT_LIMITS",
    "DEFAULT_PRIVACY_POLICY",
    "UNTRUSTED_CONTENT_INSTRUCTION",
    "ApprovalVerifier",
    "CallerContext",
    "Decision",
    "DocumentScanner",
    "GateContext",
    "Limits",
    "PrivacyPolicy",
    "QuarantinedHit",
    "Reason",
    "RetrievalResult",
    "ToolRule",
    "authorize_tool_call",
    "check_privacy",
    "gate_assessment",
    "prepare_retrieved_hits",
    "scan_document",
]
