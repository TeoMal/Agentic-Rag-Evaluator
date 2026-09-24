"""Errors raised by the RAG ingestion pipeline."""


class DocumentLoadError(RuntimeError):
    """Raised when the knowledge pack cannot be loaded as-is."""


class UnsupportedDocumentError(DocumentLoadError):
    """A file in the knowledge directory is not a PDF."""


class UnknownDocumentError(DocumentLoadError):
    """A PDF in the knowledge directory matches no document registry rule."""


class VectorStoreUnavailableError(RuntimeError):
    """The configured vector database cannot be reached or initialised."""


class RetrievalUnavailableError(RuntimeError):
    """A search could not be run (backend failure). Not the same as "no evidence found"."""


class EvidenceChannelError(RuntimeError):
    """A retrieval channel returned a document of the wrong type -- a filter bug, never silently accepted."""


class IngestionError(RuntimeError):
    """Indexing could not complete; the message says what state the index was left in."""
