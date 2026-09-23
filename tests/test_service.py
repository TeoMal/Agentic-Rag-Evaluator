from fastapi.testclient import TestClient

from hackathon2.service import create_app


def test_liveness(make_settings):
    client = TestClient(create_app(make_settings()))
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_reports_image_tag_and_subsystems(make_settings):
    body = TestClient(create_app(make_settings(image_tag="abc123"))).get("/health").json()
    assert body["image_tag"] == "abc123"
    assert body["checks"] == {
        "llm": "not_configured",
        "database": "not_configured",
        "telemetry": "disabled",
        "knowledge_documents": 0,
    }


def test_health_reports_configured_llm(make_settings, llm_settings):
    body = TestClient(create_app(make_settings(**llm_settings))).get("/health").json()
    assert body["checks"]["llm"] == "configured"


def test_health_counts_only_corpus_documents(make_settings, tmp_path):
    (tmp_path / "procurement-policy.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "vendor-x-pricing.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / ".gitkeep").write_text("")
    body = TestClient(create_app(make_settings())).get("/health").json()
    assert body["checks"]["knowledge_documents"] == 2


def test_unreachable_database_degrades_instead_of_failing(make_settings):
    # FR15 in miniature: a dead dependency is reported, the service keeps answering.
    settings = make_settings(postgres_host="127.0.0.1", postgres_port=1, postgres_password="x")
    response = TestClient(create_app(settings)).get("/health")
    assert response.status_code == 200
    assert response.json()["checks"]["database"] == "unavailable"
