"""Settings, /health and the deploy script."""

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from hackathon2.llm import get_embeddings
from hackathon2.service import create_app

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy.py"
_spec = importlib.util.spec_from_file_location("deploy_script", SCRIPT)
deploy = importlib.util.module_from_spec(_spec)
sys.modules["deploy_script"] = deploy  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(deploy)


def test_llm_needs_key_endpoint_and_deployment_and_defaults_the_api_version(make_settings, llm_settings):
    assert make_settings(**llm_settings).llm_configured
    for missing in ("azure_openai_api_key", "azure_openai_endpoint", "azure_openai_deployment_name"):
        assert not make_settings(**{k: v for k, v in llm_settings.items() if k != missing}).llm_configured
    no_version = make_settings(**{k: v for k, v in llm_settings.items() if k != "openai_api_version"})
    assert no_version.openai_api_version == "2024-12-01-preview"  # tool calling needs a 2024 version


def test_embeddings_can_live_on_a_separate_resource(make_settings, llm_settings):
    url = "https://embed.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
    settings = make_settings(**llm_settings, azure_embedding_endpoint=url, azure_embedding_api_key="embed-key")
    target = settings.embedding_target
    assert (target.endpoint, target.deployment, target.api_version) == (
        "https://embed.openai.azure.com/",
        "text-embedding-3-small",
        "2023-05-15",
    )
    assert get_embeddings(settings).deployment == "text-embedding-3-small"


def test_database_url_escapes_credentials_and_reaches_localhost_over_ipv4(make_settings):
    settings = make_settings(postgres_host="localhost", postgres_port=5446, postgres_password="p@ss/word")
    assert settings.database_url == "postgresql://hackathon2:p%40ss%2Fword@127.0.0.1:5446/hackathon2"
    assert make_settings(postgres_host="db").database_url is None  # no password -> in-memory state


def test_health_reports_the_image_tag_and_every_subsystem(make_settings, tmp_path):
    (tmp_path / "historical-vendor-assessments").mkdir()
    (tmp_path / "historical-vendor-assessments" / "vendor-alpha-assessment.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "procurement-policy.pdf").write_bytes(b"%PDF-1.4")
    client = TestClient(create_app(make_settings(image_tag="abc123")))
    assert client.get("/").json() == {"status": "ok"}
    body = client.get("/health").json()
    assert body["image_tag"] == "abc123"
    assert body["checks"] == {"llm": "not_configured", "database": "not_configured", "knowledge_documents": 2,
                              "tracing": "not_configured"}


def test_deploy_script_env_file_and_image_tags():
    text = '# comment\nAZURE_OPENAI_API_KEY="abc=123"\nexport APP_PORT=8020 # port\nEMPTY=\n'
    assert deploy.parse_env(text) == {"AZURE_OPENAI_API_KEY": "abc=123", "APP_PORT": "8020", "EMPTY": ""}
    assert deploy.parse_env(deploy.set_env_value(text, "EMPTY", "v"))["EMPTY"] == "v"
    now = datetime(2026, 9, 23, 14, 5, 9, tzinfo=UTC)
    assert deploy.make_image_tag("0123456789ab", dirty=False, now=now) == "0123456789ab"
    assert deploy.make_image_tag("0123456789ab", dirty=True, now=now) == "0123456789ab-dirty-20260923140509"
