import pytest

from hackathon2.llm import LLMNotConfiguredError, get_chat_model, get_embeddings


def test_llm_configured_needs_key_endpoint_and_deployment(make_settings, llm_settings):
    assert make_settings(**llm_settings).llm_configured
    for missing in ("azure_openai_api_key", "azure_openai_endpoint", "azure_openai_deployment_name"):
        partial = {k: v for k, v in llm_settings.items() if k != missing}
        assert not make_settings(**partial).llm_configured, missing


def test_chat_api_version_defaults_to_one_with_tool_calling(make_settings, llm_settings):
    settings = make_settings(**{k: v for k, v in llm_settings.items() if k != "openai_api_version"})
    assert settings.llm_configured and settings.openai_api_version == "2024-12-01-preview"


def test_database_url_escapes_credentials(make_settings):
    settings = make_settings(postgres_host="db", postgres_password="p@ss/word")
    assert settings.database_url == "postgresql://hackathon2:p%40ss%2Fword@db:5432/hackathon2"
    assert settings.sqlalchemy_database_url == "postgresql+psycopg://hackathon2:p%40ss%2Fword@db:5432/hackathon2"


def test_localhost_database_is_reached_over_ipv4(make_settings):
    # Windows resolves localhost to ::1 first; the compose port is published on 127.0.0.1 only.
    settings = make_settings(postgres_host="localhost", postgres_port=5446, postgres_password="pw")
    assert settings.database_url == "postgresql://hackathon2:pw@127.0.0.1:5446/hackathon2"


def test_database_is_optional(make_settings):
    assert make_settings().database_url is None
    assert make_settings(postgres_host="db").database_url is None  # no password -> not configured


def test_chat_model_fails_clearly_when_not_configured(make_settings):
    with pytest.raises(LLMNotConfiguredError, match="AZURE_OPENAI_API_KEY"):
        get_chat_model(make_settings())


def test_chat_model_targets_the_configured_deployment(make_settings, llm_settings):
    llm = get_chat_model(make_settings(**llm_settings))
    assert llm.deployment_name == "gpt-4.1-mini"


def test_embeddings_need_their_own_deployment(make_settings, llm_settings):
    with pytest.raises(LLMNotConfiguredError, match="EMBEDDING"):
        get_embeddings(make_settings(**llm_settings))
    embeddings = get_embeddings(
        make_settings(**llm_settings, azure_openai_embedding_deployment="text-embedding-3-small")
    )
    assert embeddings.deployment == "text-embedding-3-small"


def test_embeddings_can_live_on_a_separate_resource(make_settings, llm_settings):
    url = "https://embed.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
    settings = make_settings(**llm_settings, azure_embedding_endpoint=url, azure_embedding_api_key="embed-key")
    target = settings.embedding_target
    assert (target.endpoint, target.deployment, target.api_version) == (
        "https://embed.openai.azure.com/",
        "text-embedding-3-small",
        "2023-05-15",
    )
    assert target.api_key.get_secret_value() == "embed-key"
    assert settings.openai_api_version == "2024-12-01-preview"  # the chat model keeps its own version
    assert make_settings(azure_embedding_endpoint=url, azure_embedding_api_key="k").embedding_target  # no chat needed
