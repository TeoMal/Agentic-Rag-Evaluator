"""Runtime settings, read from the environment (and `.env` for local runs).

One place for every knob, so the container and a local `uv run` configure the
service the same way. Names match the course `.env`
(AZURE_OPENAI_* / OPENAI_API_VERSION) so existing keys work unchanged.
"""

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    openai_api_version: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_embedding_deployment: str | None = None

    # --- Postgres + pgvector (vector store and durable HITL checkpoints) ---
    # Optional: without a host the app runs on in-memory state.
    postgres_host: str | None = None
    postgres_port: int = 5432
    postgres_user: str = "hackathon2"
    postgres_password: SecretStr | None = None
    postgres_db: str = "hackathon2"

    # --- RAG corpus (the NFS knowledge pack) ---
    knowledge_dir: Path = Path("knowledge")

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
    def database_url(self) -> str | None:
        """libpq URL -- for psycopg and langgraph-checkpoint-postgres."""
        if not self.postgres_host or self.postgres_password is None:
            return None
        password = quote_plus(self.postgres_password.get_secret_value())
        return (
            f"postgresql://{quote_plus(self.postgres_user)}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sqlalchemy_database_url(self) -> str | None:
        """SQLAlchemy URL -- for langchain-postgres (PGVector)."""
        url = self.database_url
        return url.replace("postgresql://", "postgresql+psycopg://", 1) if url else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
