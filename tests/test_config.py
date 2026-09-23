import pytest

from hackathon2.llm import LLMNotConfiguredError, get_chat_model, get_embeddings


def test_llm_configured_needs_all_four_azure_settings(make_settings, llm_settings):
    assert make_settings(**llm_settings).llm_configured
    for missing in llm_settings:
        partial = {k: v for k, v in llm_settings.items() if k != missing}
        assert not make_settings(**partial).llm_configured, missing


def test_database_url_escapes_credentials(make_settings):
    settings = make_settings(postgres_host="db", postgres_password="p@ss/word")
    assert settings.database_url == "postgresql://hackathon2:p%40ss%2Fword@db:5432/hackathon2"
    assert settings.sqlalchemy_database_url == "postgresql+psycopg://hackathon2:p%40ss%2Fword@db:5432/hackathon2"


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
    embeddings = get_embeddings(make_settings(**llm_settings, azure_openai_embedding_deployment="text-embedding-3-small"))
    assert embeddings.deployment == "text-embedding-3-small"
