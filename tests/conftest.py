import os

import pytest

from hackathon2 import observability
from hackathon2.config import Settings

# Anything a developer's shell could leak into Settings.
_ISOLATED_PREFIXES = ("AZURE_", "OPENAI_", "POSTGRES_", "IMAGE_TAG", "KNOWLEDGE_DIR", "APP_ENV", "LANGFUSE_")


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    """Tests never send traces, even when the developer's .env has Langfuse keys."""
    monkeypatch.setattr(observability, "get_tracer", lambda: observability.Tracer())


@pytest.fixture
def make_settings(monkeypatch, tmp_path):
    """Settings built only from what the test passes -- never the real .env or shell."""
    for key in list(os.environ):
        if key.upper().startswith(_ISOLATED_PREFIXES):
            monkeypatch.delenv(key)

    def _make(**overrides) -> Settings:
        overrides.setdefault("knowledge_dir", tmp_path)
        return Settings(_env_file=None, **overrides)

    return _make


@pytest.fixture
def llm_settings() -> dict:
    return {
        "azure_openai_api_key": "test-key",
        "azure_openai_endpoint": "https://example.openai.azure.com/",
        "openai_api_version": "2024-12-01-preview",
        "azure_openai_deployment_name": "gpt-4.1-mini",
    }
