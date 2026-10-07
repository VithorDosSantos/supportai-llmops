"""Modelos tipados de `configs/data.yaml`."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportai.config import PROJECT_ROOT, load_yaml_config

DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"

URGENCY_LEVELS = ("low", "medium", "high")


class _Strict(BaseModel):
    # extra="forbid": uma chave com erro de digitação no YAML falha alto.
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceConfig(_Strict):
    repo_id: str
    revision: str
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str


class PathsConfig(_Strict):
    raw: Path
    processed_dir: Path

    def resolve(self, root: Path = PROJECT_ROOT) -> "PathsConfig":
        """Caminhos relativos no YAML são relativos à raiz do projeto."""
        return PathsConfig(raw=root / self.raw, processed_dir=root / self.processed_dir)


class FilterConfig(_Strict):
    language: str
    queues: dict[str, str] = Field(min_length=2)
    priority_map: dict[str, str]
    min_text_chars: int = Field(ge=1)

    @model_validator(mode="after")
    def _priorities_map_to_known_levels(self) -> "FilterConfig":
        unknown = set(self.priority_map.values()) - set(URGENCY_LEVELS)
        if unknown:
            raise ValueError(f"níveis de urgência desconhecidos: {sorted(unknown)}")
        return self

    @property
    def categories(self) -> list[str]:
        return sorted(set(self.queues.values()))


class SplitConfig(_Strict):
    near_duplicate_threshold: float = Field(gt=0, le=1)
    test_size: float = Field(gt=0, lt=0.5)
    val_size: float = Field(gt=0, lt=0.5)
    seed: int


class DataConfig(_Strict):
    source: SourceConfig
    paths: PathsConfig
    filter: FilterConfig
    split: SplitConfig


def load_data_config(path: Path = DEFAULT_DATA_CONFIG) -> DataConfig:
    return load_yaml_config(path, DataConfig)
