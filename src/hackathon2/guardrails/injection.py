"""Bounded heuristic screening; a clean document remains untrusted data.

No heuristic detects all injections (including arbitrary encodings or languages).
Permissions and approvals must still be enforced independently of this scanner.
"""

import re
import unicodedata

from .policy import DEFAULT_LIMITS, Decision, Limits, Reason, _decision

_EXFIL_VERB = r"(?:reveal|print|send|leak|exfiltrate|upload|show|disclose)"
# Local grammar only: no chunk-wide exemption for negation. Coordinated bare
# verbs share the prohibition, but objects, punctuation and new clauses end it.
_PROHIBITION = re.compile(
    r"\b(?:never|do not|don't|must not|mustn't|should not|shouldn't|shall not) "
    r"(?:ever )?"
    rf"(?:{_EXFIL_VERB}(?:, | (?:or|and) |, (?:or|and) )){{0,4}}$"
)

_RULES = (
    (
        Reason.INSTRUCTION_OVERRIDE,
        re.compile(
            r"\b(?:ignore|disregard|override|forget)\s+"
            r"(?:(?:all|any|the|your|previous|prior|above|system|developer|safety)\s+){0,6}"
            r"(?:instructions?|rules?|prompts?|polic(?:y|ies)|guardrails)\b"
        ),
    ),
    (
        Reason.ROLE_SPOOFING,
        re.compile(
            r"(?:<\|(?:im_start|system|developer|assistant)\|>|</?(?:system|developer)>|"
            r"(?:^|\n)\s*(?:system|developer|assistant)\s*:|"
            r"\byou are now\s+(?:the\s+)?(?:system|developer|administrator)\b)"
        ),
    ),
    (
        Reason.SECRET_EXFILTRATION,
        re.compile(
            # Lookahead checks each verb, including overlapping matches. A
            # prohibited verb must not consume a later affirmative instruction.
            rf"(?=\b{_EXFIL_VERB}\b[^.!?\n]{{0,100}}"
            r"(?:\b(?:secrets?|credentials?|passwords?|api[\s_-]*keys?|system\s+prompt|environment\s+variables)\b|\.env\b))"
        ),
    ),
    (
        Reason.UNAUTHORIZED_ACTION,
        re.compile(
            r"\b(?:call|invoke|execute|run)\s+(?:(?:the|tool|command)\s+){0,2}"
            r"(?:record_assessment|delete_\w+|exec|shell|bash|powershell|curl|wget)\b|"
            r"\b(?:delete|erase)\s+(?:all\s+)?(?:files|records|logs)\b|"
            r"\b(?:bypass|skip|disable)\s+(?:(?:the|human|tool)\s+){0,2}"
            r"(?:approval|authorization|review|guardrails)\b"
        ),
    ),
    (
        Reason.FORGED_APPROVAL,
        re.compile(
            r"\b(?:human_approval|approval_status)[\"']?\s*[:=]\s*[\"']?approved\b|"
            r"\b(?:human|reviewer|administrator)\s+(?:has\s+)?(?:already\s+)?approved\b|"
            r"\b(?:mark|set)\s+(?:this\s+|the\s+)?(?:assessment|vendor|request)\s+(?:as\s+)?approved\b"
        ),
    ),
)


def scan_document(text: str, *, limits: Limits = DEFAULT_LIMITS) -> Decision:
    """Return stable codes only. Oversized input is denied, never silently truncated."""
    if type(text) is not str or not isinstance(limits, Limits):
        return _decision("deny", Reason.INVALID_INPUT)
    if len(text) > min(limits.max_text_chars, limits.max_payload_chars):
        return _decision("deny", Reason.LIMIT_EXCEEDED)
    normalized = unicodedata.normalize("NFKC", text).casefold()
    if len(normalized) > limits.max_text_chars:
        return _decision("deny", Reason.LIMIT_EXCEEDED)
    normalized = "".join(c for c in normalized if unicodedata.category(c) != "Cf")
    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    reasons = tuple(
        code
        for code, pattern in _RULES
        if any(
            code != Reason.SECRET_EXFILTRATION
            or not _PROHIBITION.search(normalized[max(0, match.start() - 160) : match.start()])
            for match in pattern.finditer(normalized)
        )
    )
    return _decision("deny", *reasons) if reasons else Decision("allow")
