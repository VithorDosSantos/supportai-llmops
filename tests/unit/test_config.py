from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from supportai.config import (
    DatabaseSettings,
    LLMProvider,
    LLMSettings,
    Settings,
    get_settings,
)

_ENV_VARS = (
    "ENV",
    "SEED",
    "DB_HOST",
    "DB_PORT",
    "DB_PASSWORD",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Isola os testes do ambiente e de um `.env` local do desenvolvedor."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_are_local_and_free() -> None:
    settings = Settings()
    assert settings.llm.provider is LLMProvider.OLLAMA
    assert settings.seed == 42
    assert settings.db.port == 5432


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "db.internal")
    monkeypatch.setenv("DB_PORT", "6543")
    monkeypatch.setenv("SEED", "7")
    settings = Settings()
    assert settings.db.host == "db.internal"
    assert settings.db.port == 6543
    assert settings.seed == 7


def test_database_dsn() -> None:
    dsn = DatabaseSettings(host="h", port=1, user="u", password=SecretStr("p"), name="n").dsn
    assert dsn == "postgresql://u:p@h:1/n"


def test_invalid_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValidationError):
        LLMSettings()


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_paid_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    monkeypatch.setenv("LLM_PROVIDER", provider)
    with pytest.raises(ValidationError, match="chave de API"):
        LLMSettings()


def test_empty_api_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(ValidationError, match="chave de API"):
        LLMSettings()


def test_paid_provider_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    settings = LLMSettings()
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-test"


def test_secrets_do_not_leak_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_PASSWORD", "super-secreta")
    assert "super-secreta" not in repr(Settings())


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
