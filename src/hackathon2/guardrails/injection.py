"""Bounded heuristic screening; a clean document remains untrusted data.

No heuristic detects all injections (including arbitrary encodings or languages).
Permissions and approvals must still be enforced independently of this scanner.
Targeted English/Greek assessment rules cover selected imperative phrases, not
general language understanding, every Greek inflection, or transliterated Greek.
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
# Suppress only the override-family match immediately governed by a prohibition.
# Whitespace may include PDF line breaks; punctuation or intervening words end
# the scope. A later affirmative match is still inspected independently.
_OVERRIDE_PROHIBITION = re.compile(
    r"\b(?:(?:must|should|shall|may|can)\s+(?:not|never)|do\s+not|don['’]t|"
    r"never|mustn['’]t|shouldn['’]t|cannot|can['’]t)\s+(?:ever\s+)?[\"'“‘]?$"
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


# Bounded phrase windows: require directives/evidence-disregard/forced outcomes,
# never isolated words such as APPROVE, compliant, vendor or Greek equivalents.
# Greek patterns use accent-insensitive casefolded text (final sigma becomes σ).
_GR_IGNORE = r"(?:αγνοησε|αγνοηστε|παραβλεψε|παραβλεψτε|παρακαμψε|παρακαμψτε)"
_GR_EVIDENCE = r"(?:στοιχεια|τεκμηρια|αποδειξεισ|τεκμηριωση)"
_EVIDENCE_CONDITION = re.compile(r"^\s+(?:only\s+)?(?:if|when)\b[^.!?;\n]{0,120}\bevidence\b")
_ASSESSMENT_RULES = (
    (Reason.INSTRUCTION_OVERRIDE, re.compile(rf"\b{_GR_IGNORE}\b[^.!?;\n]{{0,60}}\b(?:οδηγιεσ|εντολεσ|κανονεσ)\b")),
    (
        Reason.EVIDENCE_OVERRIDE,
        re.compile(
            r"\b(?:ignore|disregard|dismiss)\s+(?:(?:all|any|the|available|mandatory|supporting|contradictory)\s+){0,4}"
            r"(?:evidence|citations|findings)\b|"
            r"\b(?:regardless|irrespective)\s+of\s+(?:(?:all|any|the|available|supporting)\s+){0,4}evidence\b|"
            rf"\b{_GR_IGNORE}\b[^.!?;\n]{{0,60}}\b{_GR_EVIDENCE}\b|"
            rf"\bανεξαρτητα\s+απο\s+(?:(?:τα|την|τη|ολα|διαθεσιμα)\s+){{0,4}}{_GR_EVIDENCE}\b"
        ),
    ),
    (
        Reason.ASSESSMENT_MANIPULATION,
        re.compile(
            r"\b(?:force|guarantee)\b[^.!?;\n]{0,60}\b(?:assessment|outcome|recommendation|verdict)\b"
            r"[^.!?;\n]{0,60}\b(?:approve|reject|conditional_approval|compliant)\b|"
            r"\b(?:always|unconditionally)\s+(?:approve|reject)\s+(?:(?:the|this|that)\s+)?vendor\b|"
            r"\b(?:treat|mark|declare|classify)\b[^.!?;\n]{0,40}\bvendor\b[^.!?;\n]{0,40}\bcompliant\b"
            r"[^.!?;\n]{0,40}\b(?:regardless|unconditionally|no matter)\b|"
            r"\b(?:return|output)\s+[\"']?(?:approve|reject|conditional_approval)\b"
            r"[^.!?;\n]{0,60}\b(?:regardless|unconditionally|no matter)\b|"
            r"\b(?:παντα|οπωσδηποτε|ανευ\s+ορων)\s+(?:εγκρινε|εγκρινετε|απορριψε|απορριψτε)\b"
            r"[^.!?;\n]{0,40}\b(?:προμηθευτη|προμηθευτεσ|αξιολογηση)\b|"
            r"\b(?:εγκρινε|εγκρινετε|απορριψε|απορριψτε)\b[^.!?;\n]{0,60}\bπρομηθευτη\b"
            r"[^.!?;\n]{0,60}\b(?:ανεξαρτητα|ανευ\s+ορων|οπωσδηποτε)\b|"
            r"\b(?:θεωρησε|χαρακτηρισε)\b[^.!?;\n]{0,40}\bπρομηθευτη\b"
            r"[^.!?;\n]{0,40}\bσυμμορφωμενο\b[^.!?;\n]{0,40}\bανεξαρτητα\b"
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
            not (
                code == Reason.SECRET_EXFILTRATION
                and _PROHIBITION.search(normalized[max(0, match.start() - 160) : match.start()])
            )
            and not (
                code == Reason.INSTRUCTION_OVERRIDE
                and _OVERRIDE_PROHIBITION.search(normalized[max(0, match.start() - 160) : match.start()])
            )
            for match in pattern.finditer(normalized)
        )
    )
    folded = "".join(c for c in unicodedata.normalize("NFD", normalized) if not unicodedata.combining(c))
    if len(folded) > limits.max_text_chars:
        return _decision("deny", Reason.LIMIT_EXCEEDED)
    reasons += tuple(
        code
        for code, pattern in _ASSESSMENT_RULES
        if any(
            not (
                code == Reason.EVIDENCE_OVERRIDE
                and match.group().startswith(("ignore ", "disregard ", "dismiss "))
                and _PROHIBITION.search(folded[max(0, match.start() - 160) : match.start()])
            )
            and not (
                code == Reason.ASSESSMENT_MANIPULATION
                and match.group().startswith("always ")
                and _EVIDENCE_CONDITION.search(folded[match.end() : match.end() + 160])
            )
            for match in pattern.finditer(folded)
        )
    )
    return _decision("deny", *reasons) if reasons else Decision("allow")
