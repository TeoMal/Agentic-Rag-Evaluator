"""Deterministic classification of knowledge-pack files -- metadata only, never conclusions.

The corpus layout already says what each file is, so no LLM guesses it:

    knowledge/
    ├── <name>-policy.pdf                          policy             per POLICY_DOMAINS
    ├── vendor-<vendor>-proposal.pdf               vendor_claim       general
    ├── vendor-<vendor>-security-questionnaire.pdf vendor_claim       security
    ├── vendor-<vendor>-pricing.pdf                vendor_claim       procurement
    └── historical-vendor-assessments/
        └── vendor-<vendor>-assessment.pdf         enterprise_record  general

Rules instead of a fixed file list, so a different vendor (e.g. the hidden vendor
case: vendor-y-*.pdf) is classified without code changes. Any other file raises
UnknownDocumentError instead of being indexed without metadata.
"""

import re
from pathlib import PurePath, PurePosixPath

from hackathon2.rag.data_models import DocumentInfo, KnowledgeDomain
from hackathon2.rag.errors import UnknownDocumentError
from hackathon2.schemas import DocType

HISTORICAL_DIR = "historical-vendor-assessments"

# Primary domain of each NFS policy. A policy file not listed here raises
# UnknownDocumentError: its domain is not guessed, it has to be added here.
POLICY_DOMAINS: dict[str, KnowledgeDomain] = {
    "procurement-policy": "procurement",
    "information-security-policy": "security",
    "ai-governance-policy": "ai_governance",
    "data-classification-policy": "data",
    # Defines the risk domains, overall rating and decision outcomes for every domain.
    "vendor-risk-policy": "general",
}

# Vendor document kind (file-name suffix) -> primary domain.
VENDOR_CLAIM_DOMAINS: dict[str, KnowledgeDomain] = {
    "proposal": "general",
    "security-questionnaire": "security",
    "pricing": "procurement",
}

_POLICY = re.compile(r"^(?P<doc_id>[a-z0-9-]+-policy)\.pdf$")
_VENDOR_CLAIM = re.compile(
    rf"^(?P<doc_id>vendor-(?P<vendor>[a-z0-9]+(?:-[a-z0-9]+)*?)-(?P<kind>{'|'.join(VENDOR_CLAIM_DOMAINS)}))\.pdf$"
)
_ASSESSMENT = re.compile(r"^(?P<doc_id>vendor-(?P<vendor>[a-z0-9]+(?:-[a-z0-9]+)*)-assessment)\.pdf$")
_ANY_PDF = re.compile(r"^(?P<doc_id>[a-z0-9-]+)\.pdf$")


def classify(relative_path: str | PurePath) -> DocumentInfo:
    """Metadata for a file, given its path relative to the knowledge directory.

    Matching is case-insensitive; `source` keeps the original file name for citations.
    """
    path = PurePosixPath(PurePath(relative_path).as_posix())
    name, folder = path.name.lower(), path.parent.as_posix().lower()

    if folder == HISTORICAL_DIR:
        if match := _ASSESSMENT.match(name):
            return _info(path, match["doc_id"], "enterprise_record", "general", match["vendor"])
        if match := _ANY_PDF.match(name):  # a record in the historical folder, vendor not in its name
            return _info(path, match["doc_id"], "enterprise_record", "general", None)
    elif folder == ".":
        if match := _POLICY.match(name):
            if match["doc_id"] not in POLICY_DOMAINS:
                raise UnknownDocumentError(
                    f"Policy '{path}' has no domain in POLICY_DOMAINS (rag/document_registry.py); add it there."
                )
            return _info(path, match["doc_id"], "policy", POLICY_DOMAINS[match["doc_id"]], None)
        if match := _VENDOR_CLAIM.match(name):
            return _info(path, match["doc_id"], "vendor_claim", VENDOR_CLAIM_DOMAINS[match["kind"]], match["vendor"])

    raise UnknownDocumentError(
        f"No document registry rule matches '{path}'. Expected <name>-policy.pdf, "
        f"vendor-<vendor>-{{{','.join(VENDOR_CLAIM_DOMAINS)}}}.pdf or "
        f"{HISTORICAL_DIR}/vendor-<vendor>-assessment.pdf (see rag/document_registry.py)."
    )


def _info(
    path: PurePosixPath, doc_id: str, doc_type: DocType, domain: KnowledgeDomain, vendor: str | None
) -> DocumentInfo:
    return DocumentInfo(doc_id=doc_id, source=path.name, doc_type=doc_type, domain=domain, vendor=vendor)
