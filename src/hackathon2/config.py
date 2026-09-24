"""Runtime settings, read from the environment (and `.env` for local runs).

One place for every knob, so the container and a local `uv run` configure the
service the same way. Names match the course `.env`
(AZURE_OPENAI_* / OPENAI_API_VERSION) so existing keys work unchanged.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, quote_plus, urlsplit

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Chat API version when OPENAI_API_VERSION is not set: tool calling and structured output (what
# the agents run on) need 2024 versions -- the older GA 2023-05-15 has neither.
DEFAULT_CHAT_API_VERSION = "2024-12-01-preview"


class EmbeddingTarget(NamedTuple):
    """Where embeddings are computed: resource endpoint, key, deployment and API version."""

    endpoint: str
    api_key: SecretStr
    deployment: str
    api_version: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    # Baked into the image at build time (Dockerfile ARG) -- /health echoes it so a
    # deploy can prove the NEW image is the one answering.
    image_tag: str = "dev"

    # --- Azure OpenAI (chat + embeddings) ---
    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    openai_api_version: str = DEFAULT_CHAT_API_VERSION
    azure_openai_deployment_name: str | None = None
    azure_openai_embedding_deployment: str | None = None
    # Optional: embeddings on ANOTHER Azure OpenAI resource. AZURE_EMBEDDING_ENDPOINT may be the
    # resource URL or a full deployment URL (.../openai/deployments/<name>/embeddings?api-version=...);
    # whatever is not given comes from the chat settings above.
    azure_embedding_endpoint: str | None = None
    azure_embedding_api_key: SecretStr | None = None

    # --- Postgres + pgvector (vector store and durable HITL checkpoints) ---
    # Optional: without a host the app runs on in-memory state.
    postgres_host: str | None = None
    postgres_port: int = 5432
    postgres_user: str = "hackathon2"
    postgres_password: SecretStr | None = None
    postgres_db: str = "hackathon2"

    # --- RAG corpus (the NFS knowledge pack) ---
    knowledge_dir: Path = Path("knowledge")

    # --- Langfuse tracing (observability.py) ---
    # Optional: without both keys and a host nothing is traced and nothing is sent anywhere.
    # LANGFUSE_HOST is the local stack (docker-compose.langfuse.yml) or Langfuse Cloud.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None
    langfuse_tracing_enabled: bool = True

    @property
    def llm_configured(self) -> bool:
        return all(
            (
                self.azure_openai_api_key and self.azure_openai_api_key.get_secret_value(),
                self.azure_openai_endpoint,
                self.openai_api_version,
                self.azure_openai_deployment_name,
            )
        )

    @property
    def tracing_configured(self) -> bool:
        return bool(
            self.langfuse_tracing_enabled
            and self.langfuse_public_key
            and self.langfuse_secret_key
            and self.langfuse_secret_key.get_secret_value()
            and self.langfuse_host
        )

    @property
    def embedding_target(self) -> EmbeddingTarget | None:
        """The embedding resource, or None when a piece is missing (no deployment, no key...)."""
        endpoint, deployment, version = self.azure_openai_endpoint, None, self.openai_api_version
        if self.azure_embedding_endpoint:
            url = urlsplit(self.azure_embedding_endpoint)
            endpoint = f"{url.scheme}://{url.netloc}/"
            if match := re.search(r"/deployments/([^/]+)", url.path):
                deployment = match.group(1)
            version = parse_qs(url.query).get("api-version", [version])[0]
        deployment = self.azure_openai_embedding_deployment or deployment
        key = self.azure_embedding_api_key or self.azure_openai_api_key
        if not (endpoint and deployment and key and key.get_secret_value()):
            return None
        return EmbeddingTarget(endpoint, key, deployment, version)

    @property
    def database_url(self) -> str | None:
        """libpq URL -- for psycopg and langgraph-checkpoint-postgres."""
        if not self.postgres_host or self.postgres_password is None:
            return None
        password = quote_plus(self.postgres_password.get_secret_value())
        # docker-compose publishes Postgres on 127.0.0.1 only, and on Windows "localhost" tries IPv6
        # (::1) first -- about 10 s per connection, minutes for PGVector, which connects often.
        host = "127.0.0.1" if self.postgres_host == "localhost" else self.postgres_host
        return (
            f"postgresql://{quote_plus(self.postgres_user)}:{password}"
            f"@{host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sqlalchemy_database_url(self) -> str | None:
        """SQLAlchemy URL -- for langchain-postgres (PGVector)."""
        url = self.database_url
        return url.replace("postgresql://", "postgresql+psycopg://", 1) if url else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
