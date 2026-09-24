"""Pipeline A, step 2: split pages into chunks that inherit the page's metadata.

Every chunk carries doc_id, source, doc_type, domain, vendor and page from its
DocumentPage, plus the section it belongs to and a stable chunk_id:

    '<doc_id>#s<section>#c<n>'   e.g. 'information-security-policy#s3#c1', 'vendor-x-security-questionnaire#sB#c1'
    '<doc_id>#p<page>#c<n>'      text with no detected section (document header, unnumbered documents)

Design choices:
- Chunks never cross a page boundary, so every citation points to exactly one page.
- Pages are first cut into section blocks at headings, then split by size, so a
  chunk belongs to exactly one section (retrieve_document(expand_section=True) can
  gather a whole section). A section that continues onto the next page keeps its
  label there.
- chunk_ids are deterministic: re-indexing the same PDFs gives the same ids, so
  citations and evaluation cases stay valid across re-ingestion.
- Chunk text is the page text as extracted; only NUL characters are removed, because
  Postgres rejects them in text columns.
- `suspicious` is left False here; the injection scan (security.py) sets it in ingest.

Section headings are recognised only in the two forms the knowledge pack uses:
numbered ('3. Encryption', '2.1 Up to EUR 25,000 annual value') and lettered
('B. Encryption'). Unnumbered headings ('Summary', 'Key findings' in the historical
assessments) cannot be told apart from body text reliably, so those chunks get
section=None instead of a guessed one.
"""

import re
from collections import Counter
from collections.abc import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hackathon2.rag.data_models import DocumentChunk, DocumentPage

# Baseline defaults, overridable per call:
# - CHUNK_SIZE = schemas.Evidence.quote's max_length, so any chunk can be cited whole.
# - CHUNK_OVERLAP keeps the 20% ratio of the LangChain RAG tutorial (1000 / 200), so a
#   sentence cut at a chunk boundary also appears in the neighbouring chunk.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# '3. Encryption', '2.1 Up to EUR 25,000 annual value', 'B. Encryption': a section
# number (trailing dot optional) or a single capital letter with a dot, then a
# capitalised title.
_HEADING = re.compile(r"^(?:(?P<number>\d+(?:\.\d+)*)\.?|(?P<letter>[A-Z])\.)\s+[A-Z]")

Section = tuple[str, str]  # (label, heading line), e.g. ('3', '3. Encryption'), ('B', 'B. Encryption')


def chunk_pages(
    pages: Iterable[DocumentPage], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[DocumentChunk]:
    """Split pages (in document then page order, as load_knowledge returns them) into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    counters: Counter[str] = Counter()
    open_section: dict[str, Section | None] = {}  # doc_id -> section still open at the end of the last page

    chunks: list[DocumentChunk] = []
    for page in pages:
        blocks, open_section[page.doc_id] = _section_blocks(
            page.text.replace("\x00", ""), open_section.get(page.doc_id)
        )
        for section, block in blocks:
            key = f"{page.doc_id}#s{section[0]}" if section else f"{page.doc_id}#p{page.page}"
            for text in splitter.split_text(block):
                counters[key] += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{key}#c{counters[key]}",
                        doc_id=page.doc_id,
                        source=page.source,
                        doc_type=page.doc_type,
                        domain=page.domain,
                        vendor=page.vendor,
                        section=section[1] if section else None,
                        page=page.page,
                        text=text,
                    )
                )
    return chunks


def _section_blocks(text: str, section: Section | None) -> tuple[list[tuple[Section | None, str]], Section | None]:
    """Cut a page into (section, text) blocks at headings; also return the section open at the end.

    Text before the first heading on the page belongs to `section`, the one carried
    over from the previous page. Heading-only blocks (no body) are dropped.
    """
    blocks: list[tuple[Section | None, str]] = []
    lines: list[str] = []
    has_body = False
    for line in text.splitlines():
        heading = _heading(line)
        if heading:
            if has_body:
                blocks.append((section, "\n".join(lines)))
            section, lines, has_body = heading, [], False
        elif line.strip():
            has_body = True
        lines.append(line)
    if has_body:
        blocks.append((section, "\n".join(lines)))
    return blocks, section


def _heading(line: str) -> Section | None:
    line = line.strip()
    match = _HEADING.match(line)
    return (match["number"] or match["letter"], line) if match else None
