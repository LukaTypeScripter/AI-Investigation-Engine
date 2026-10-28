from pathlib import Path

import pytest
from pydantic import ValidationError

from investigation_engine.settings import Settings

SETTING_VARIABLES = [
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "DATABASE_URL",
    "GEMINI_MODEL",
    "GROQ_MODEL",
    "EMBEDDING_MODEL",
    "RETRIEVAL_CANDIDATES",
    "LLM_CACHE_PATH",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in SETTING_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def test_defaults_leave_every_secret_unset() -> None:
    settings = Settings(_env_file=None)

    assert settings.gemini_api_key is None
    assert settings.groq_api_key is None
    assert settings.database_url is None


def test_defaults_describe_the_local_free_setup() -> None:
    settings = Settings(_env_file=None)

    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dimension == 384
    assert settings.retrieval_candidates == 50
    assert settings.llm_cache_path == Path("data/llm_cache.sqlite")


def test_values_are_read_from_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", "20")

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "gemini-secret"
    assert settings.retrieval_candidates == 20


def test_secrets_never_appear_in_the_text_form_of_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:db-password@host/db")

    settings = Settings(_env_file=None)

    shown = repr(settings) + str(settings)
    assert "gemini-secret" not in shown
    assert "db-password" not in shown


def test_values_are_read_from_a_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=groq-secret\nUNRELATED_SETTING=1\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "groq-secret"


def test_environment_variables_win_over_the_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("GROQ_MODEL", "from-environment")

    settings = Settings(_env_file=env_file)

    assert settings.groq_model == "from-environment"


@pytest.mark.parametrize("value", ["0", "-3", "many"])
def test_invalid_numbers_are_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_CANDIDATES", value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
