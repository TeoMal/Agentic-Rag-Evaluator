"""Deterministic classification of knowledge-pack files -- metadata only, never conclusions.

The corpus layout already says what each file is, so no LLM guesses it. Same order as
the MCP server (mcp_server/knowledge.py::_default_doc_type):

    1. an NFS policy listed in POLICY_DOMAINS          policy             per POLICY_DOMAINS
    2. historical-vendor-assessments/*.pdf             enterprise_record  general
    3. vendor-<vendor>-*.pdf                           vendor_claim       security for -security-questionnaire,
                                                                          procurement for -pricing, else general
    4. anything else                                   UnknownDocumentError

Rules instead of a fixed file list, so a different vendor (e.g. the hidden vendor
case: vendor-y-*.pdf) is classified without code changes. Only the name and folder
decide, never the content, which the vendor controls: a vendor's own
'vendor-y-privacy-policy.pdf' is a vendor claim, never an NFS requirement. Step 4
differs from the MCP server's fallback ("policy") on purpose: a stray file must not
become an NFS rule.

The vendor tag of vendor-<vendor>-<kind>.pdf is <vendor> ('blue-sky' in
vendor-blue-sky-pricing.pdf). For a kind other than proposal / security-questionnaire /
pricing it is the longest tag known from those files that the name starts with, else
the first word after 'vendor-'.
"""

import re
from collections.abc import Collection, Iterable
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

# Vendor document kind (file-name suffix) -> primary domain. Any other vendor document is "general".
VENDOR_CLAIM_DOMAINS: dict[str, KnowledgeDomain] = {
    "proposal": "general",
    "security-questionnaire": "security",
    "pricing": "procurement",
}

_VENDOR_CLAIM = re.compile(
    rf"^(?P<doc_id>vendor-(?P<vendor>[a-z0-9]+(?:-[a-z0-9]+)*?)-(?P<kind>{'|'.join(VENDOR_CLAIM_DOMAINS)}))\.pdf$"
)
_VENDOR_ANY = re.compile(r"^(?P<doc_id>vendor-(?P<rest>[a-z0-9]+(?:-[a-z0-9]+)*))\.pdf$")
_ASSESSMENT = re.compile(r"^(?P<doc_id>vendor-(?P<vendor>[a-z0-9]+(?:-[a-z0-9]+)*)-assessment)\.pdf$")
_ANY_PDF = re.compile(r"^(?P<doc_id>[a-z0-9-]+)\.pdf$")


def vendor_tags(relative_paths: Iterable[str | PurePath]) -> set[str]:
    """Vendor tags named by proposal / security-questionnaire / pricing files in the top folder."""
    tags = set()
    for relative_path in relative_paths:
        path = PurePosixPath(PurePath(relative_path).as_posix())
        if path.parent.as_posix() == "." and (match := _VENDOR_CLAIM.match(path.name.lower())):
            tags.add(match["vendor"])
    return tags


def classify(relative_path: str | PurePath, known_vendors: Collection[str] = ()) -> DocumentInfo:
    """Metadata for a file, given its path relative to the knowledge directory.

    `known_vendors` (see vendor_tags) lets a vendor document of another kind find its vendor.
    Matching is case-insensitive; `source` keeps the original file name for citations.
    """
    path = PurePosixPath(PurePath(relative_path).as_posix())
    name, folder = path.name.lower(), path.parent.as_posix().lower()

    if folder == ".":
        doc_id = name.removesuffix(".pdf")
        if doc_id in POLICY_DOMAINS:  # 1. NFS policy
            return _info(path, doc_id, "policy", POLICY_DOMAINS[doc_id], None)
        if match := _VENDOR_CLAIM.match(name):  # 3. vendor document of a known kind
            return _info(path, match["doc_id"], "vendor_claim", VENDOR_CLAIM_DOMAINS[match["kind"]], match["vendor"])
        if match := _VENDOR_ANY.match(name):  # 3. any other vendor document
            return _info(path, match["doc_id"], "vendor_claim", "general", _vendor_of(match["rest"], known_vendors))
    elif folder == HISTORICAL_DIR:  # 2. historical assessment
        if match := _ASSESSMENT.match(name):
            return _info(path, match["doc_id"], "enterprise_record", "general", match["vendor"])
        if match := _ANY_PDF.match(name):  # a record in the historical folder, vendor not in its name
            return _info(path, match["doc_id"], "enterprise_record", "general", None)

    raise UnknownDocumentError(  # 4.
        f"No document registry rule matches '{path}'. Expected an NFS policy listed in POLICY_DOMAINS, "
        f"vendor-<vendor>-<kind>.pdf or {HISTORICAL_DIR}/<name>.pdf "
        "(lower-case letters, digits and '-' only; see rag/document_registry.py)."
    )


def _vendor_of(rest: str, known_vendors: Collection[str]) -> str:
    """The longest known tag `rest` starts with ('blue-sky-soc2-report' -> 'blue-sky'), else its first word."""
    matching = [tag for tag in known_vendors if rest.startswith(f"{tag}-")]
    return max(matching, key=len) if matching else rest.split("-", 1)[0]


def _info(
    path: PurePosixPath, doc_id: str, doc_type: DocType, domain: KnowledgeDomain, vendor: str | None
) -> DocumentInfo:
    return DocumentInfo(doc_id=doc_id, source=path.name, doc_type=doc_type, domain=domain, vendor=vendor)
