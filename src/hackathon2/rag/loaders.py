"""Pipeline A, step 1: load the PDFs in Settings.knowledge_dir into page-level documents.

Walks the knowledge directory including subfolders (historical-vendor-assessments/).
Each page keeps its source file name and 1-based page number, and is enriched with
the file's classification from document_registry.classify (doc_id, doc_type, domain,
vendor). Page text is kept exactly as pypdf extracts it -- it is later quoted as
evidence. No chunking or embedding happens here.

Two kinds of failure, handled differently:
- The directory itself is wrong (a non-PDF file, a PDF no registry rule matches, two
  files with the same name): raise, before anything is loaded, so nothing is
  indexed without metadata.
- A PDF or page yields no text (corrupt, encrypted, scanned, extraction
  error): record an ExtractionIssue and keep going, so one bad file does not stop
  ingestion (FR14) and the gap is still visible to the caller.
"""

import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from hackathon2.config import Settings, get_settings
from hackathon2.rag.data_models import DocumentInfo, DocumentPage, ExtractionIssue, KnowledgeLoad
from hackathon2.rag.document_registry import classify
from hackathon2.rag.errors import DocumentLoadError, UnknownDocumentError, UnsupportedDocumentError

__all__ = ["DocumentLoadError", "UnknownDocumentError", "UnsupportedDocumentError", "load_knowledge", "load_pdf"]

logger = logging.getLogger(__name__)


def load_knowledge(settings: Settings | None = None) -> KnowledgeLoad:
    """Every page of every PDF in the knowledge directory (path then page order), plus extraction issues."""
    knowledge_dir = (settings or get_settings()).knowledge_dir
    if not knowledge_dir.is_dir():
        raise DocumentLoadError(f"Knowledge directory not found: {knowledge_dir}")

    files = _classified_files(knowledge_dir)
    result = KnowledgeLoad(documents=[info.source for _, info in files], pages=[])
    for path, info in files:
        pages, issues = load_pdf(path, info)
        result.pages.extend(pages)
        result.issues.extend(issues)

    for issue in result.issues:
        where = f"page {issue.page}" if issue.page else "whole file"
        logger.warning("No text extracted: %s (%s): %s", issue.source, where, issue.reason)
    return result


def load_pdf(path: Path, info: DocumentInfo) -> tuple[list[DocumentPage], list[ExtractionIssue]]:
    """The pages of one PDF, each carrying `info` plus its page number, and any extraction issues."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            return [], [ExtractionIssue(source=info.source, reason="encrypted PDF")]
        raw_pages = list(reader.pages)
    except (PdfReadError, OSError, ValueError) as exc:
        return [], [ExtractionIssue(source=info.source, reason=f"unreadable PDF: {exc}")]

    if not raw_pages:
        return [], [ExtractionIssue(source=info.source, reason="PDF has no pages")]

    pages: list[DocumentPage] = []
    issues: list[ExtractionIssue] = []
    for number, raw in enumerate(raw_pages, start=1):
        try:
            text = raw.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 -- pypdf raises many error types on malformed content streams
            text = ""
            issues.append(ExtractionIssue(source=info.source, page=number, reason=f"text extraction failed: {exc}"))
        else:
            if not text.strip():
                issues.append(ExtractionIssue(source=info.source, page=number, reason="no text on page"))
        pages.append(DocumentPage(**info.model_dump(), page=number, text=text))
    return pages, issues


def _classified_files(knowledge_dir: Path) -> list[tuple[Path, DocumentInfo]]:
    """Every file to load with its classification, after checking the whole directory first.

    Dotfiles and dot-folders (.gitkeep) are ignored.
    """
    files = sorted(
        p
        for p in knowledge_dir.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(knowledge_dir).parts)
    )

    unsupported = [p.relative_to(knowledge_dir).as_posix() for p in files if p.suffix.lower() != ".pdf"]
    if unsupported:
        raise UnsupportedDocumentError(f"Only PDF files are supported; found: {', '.join(unsupported)}")

    classified = [(p, classify(p.relative_to(knowledge_dir))) for p in files]

    seen: dict[str, Path] = {}
    for path, info in classified:
        if info.source in seen:
            raise DocumentLoadError(f"Two files are named {info.source}: {seen[info.source]} and {path}")
        seen[info.source] = path
    return classified
