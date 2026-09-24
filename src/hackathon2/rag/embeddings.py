"""Embedding model used by both pipelines (A: indexing chunks, B: embedding queries).

The rest of the RAG package depends only on langchain_core's `Embeddings` interface
(`embed_documents` for chunks, `embed_query` for questions), which is also what the
vector store expects. The provider is chosen here and nowhere else: today Azure
OpenAI, built by hackathon2.llm.get_embeddings from Settings, i.e. from the
environment / .env (AZURE_OPENAI_EMBEDDING_DEPLOYMENT plus the AZURE_OPENAI_* and
OPENAI_API_VERSION settings). No credentials live in code.

Indexing and querying must use the same model, otherwise query and chunk vectors are
not comparable -- so both pipelines get their embedder from get_embedder().
"""

from langchain_core.embeddings import Embeddings

from hackathon2.config import Settings, get_settings
from hackathon2.llm import LLMNotConfiguredError, get_embeddings

__all__ = ["Embeddings", "EmbeddingsNotConfiguredError", "get_embedder"]

# Raised by get_embedder when the embedding settings are missing; re-exported so RAG
# callers do not import the provider module.
EmbeddingsNotConfiguredError = LLMNotConfiguredError


def get_embedder(settings: Settings | None = None) -> Embeddings:
    """The embedding model for both indexing and querying.

    Raises EmbeddingsNotConfiguredError, naming the missing variables, when the
    embedding deployment or the Azure OpenAI settings are not set.
    """
    return get_embeddings(settings or get_settings())
