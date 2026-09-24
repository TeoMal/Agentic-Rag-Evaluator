"""Azure OpenAI model factories (course units 47 and 35/step9).

Built lazily, on first use, so the service still starts -- and /health still
answers -- when credentials are missing, instead of crashing at import time.
"""

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from hackathon2.config import Settings, get_settings


class LLMNotConfiguredError(RuntimeError):
    """Raised when an Azure OpenAI setting the caller needs is missing."""


def get_chat_model(settings: Settings | None = None, **kwargs) -> AzureChatOpenAI:
    settings = settings or get_settings()
    if not settings.llm_configured:
        raise LLMNotConfiguredError(
            "Azure OpenAI is not configured: set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
            "OPENAI_API_VERSION and AZURE_OPENAI_DEPLOYMENT_NAME (see .env.example)."
        )
    return AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment_name,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.openai_api_version,
        api_key=settings.azure_openai_api_key,
        **kwargs,
    )


def get_embeddings(settings: Settings | None = None, **kwargs) -> AzureOpenAIEmbeddings:
    settings = settings or get_settings()
    target = settings.embedding_target
    if target is None:
        raise LLMNotConfiguredError(
            "Azure OpenAI embeddings are not configured: set AZURE_OPENAI_EMBEDDING_DEPLOYMENT "
            "(an embedding model deployment, e.g. text-embedding-3-small) and either the chat settings or "
            "AZURE_EMBEDDING_ENDPOINT + AZURE_EMBEDDING_API_KEY for a separate resource."
        )
    return AzureOpenAIEmbeddings(
        azure_deployment=target.deployment,
        azure_endpoint=target.endpoint,
        api_version=target.api_version,
        api_key=target.api_key,
        **kwargs,
    )
