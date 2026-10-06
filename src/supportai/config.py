"""Configuração central da aplicação via variáveis de ambiente.

Por que pydantic-settings: valida tipos na inicialização (falha cedo, não no
meio de uma requisição), lê `.env` em desenvolvimento e variáveis de ambiente
em produção sem mudar código, e `SecretStr` evita que chaves vazem em logs/reprs.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMProvider(StrEnum):
    """Provedores suportados; o código nunca deve depender de um só."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class Environment(StrEnum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class DatabaseSettings(BaseSettings):
    """Postgres + pgvector. Prefixo `DB_` nas variáveis de ambiente."""

    model_config = SettingsConfigDict(env_prefix="DB_", env_file=".env", extra="ignore")

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = "supportai"
    password: SecretStr = SecretStr("supportai")
    name: str = "supportai"

    @property
    def dsn(self) -> str:
        """URI libpq, aceita diretamente pelo psycopg 3."""
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"


class LLMSettings(BaseSettings):
    """Provedor e modelos de LLM. Prefixo `LLM_` nas variáveis de ambiente."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = "llama3.1:8b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_s: float = Field(default=30.0, gt=0)
    ollama_base_url: str = "http://localhost:11434"
    # Chaves sem prefixo: são os nomes que os SDKs oficiais já esperam.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    @field_validator("anthropic_api_key", "openai_api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, value: object) -> object:
        # `ANTHROPIC_API_KEY=` no .env chega como "", que não é None e burlaria
        # a checagem abaixo.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _require_key_for_paid_provider(self) -> "LLMSettings":
        # Falhar na inicialização é melhor do que na primeira chamada ao LLM.
        required = {
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
        }
        if self.provider in required and required[self.provider] is None:
            raise ValueError(f"provedor '{self.provider}' exige a chave de API correspondente")
        return self


class Settings(BaseSettings):
    """Configuração raiz, agregando as seções."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Environment = Environment.DEV
    log_level: str = "INFO"
    seed: int = 42
    data_dir: Path = PROJECT_ROOT / "data"
    configs_dir: Path = PROJECT_ROOT / "configs"
    mlflow_tracking_uri: str = "file:./mlruns"

    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


@lru_cache
def get_settings() -> Settings:
    """Singleton barato; em testes, use `get_settings.cache_clear()`."""
    return Settings()
