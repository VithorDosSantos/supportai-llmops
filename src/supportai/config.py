"""Configuração central da aplicação via variáveis de ambiente.

Por que pydantic-settings: valida tipos na inicialização (falha cedo, não no
meio de uma requisição), lê `.env` em desenvolvimento e variáveis de ambiente
em produção sem mudar código, e `SecretStr` evita que chaves vazem em logs/reprs.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ConfigT = TypeVar("ConfigT", bound=BaseModel)


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


class APISettings(BaseSettings):
    """Serviço de triagem. Prefixo `API_` nas variáveis de ambiente."""

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")

    # URI do MLflow: `models:/<nome>@champion` (registry) ou um diretório local
    # exportado (`supportai.classifier.export`), usado no container e no deploy.
    model_uri_category: str = "models:/supportai-triage-category@champion"
    model_uri_urgency: str = "models:/supportai-triage-urgency@champion"
    # Limite de tamanho: protege a latência (TF-IDF de caracteres é linear no
    # texto) e evita abuso; tickets reais têm < 2 mil caracteres.
    max_text_chars: int = Field(default=5000, ge=100)
    log_json: bool = True
    # O MLflow 3 recusa desserializar modelos sklearn (inclusive skops) sem
    # MLFLOW_ALLOW_PICKLE_DESERIALIZATION=true, porque carregar um artefato pode
    # executar código. Opt-in explícito: só ligue para modelos do seu registry.
    allow_model_deserialization: bool = False


class Settings(BaseSettings):
    """Configuração raiz, agregando as seções."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Environment = Environment.DEV
    log_level: str = "INFO"
    seed: int = 42
    data_dir: Path = PROJECT_ROOT / "data"
    configs_dir: Path = PROJECT_ROOT / "configs"
    # SQLite em vez do file store: o Model Registry (aliases como "champion")
    # exige um backend de banco, e o file store está obsoleto no MLflow 3.
    mlflow_tracking_uri: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    mlflow_artifact_root: Path = PROJECT_ROOT / "mlartifacts"

    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)


@lru_cache
def get_settings() -> Settings:
    """Singleton barato; em testes, use `get_settings.cache_clear()`."""
    return Settings()


def load_yaml_config(path: Path, model: type[ConfigT]) -> ConfigT:
    """Lê um YAML de `configs/` e valida com um modelo Pydantic.

    Validar na leitura transforma um erro de digitação no YAML em uma mensagem
    clara no início do treino, em vez de um KeyError depois de minutos.
    """
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return model.model_validate(raw)
