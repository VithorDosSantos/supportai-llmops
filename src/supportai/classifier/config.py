"""Modelos tipados de `configs/train.yaml`."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from supportai.config import PROJECT_ROOT, load_yaml_config

DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"

Target = Literal["category", "urgency"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MlflowConfig(_Strict):
    experiment: str
    registered_model_prefix: str
    champion_alias: str
    selection_metric: str

    def registered_model_name(self, target: str) -> str:
        return f"{self.registered_model_prefix}-{target}"


class LogRegConfig(_Strict):
    c_grid: list[float] = Field(min_length=1)
    class_weight: Literal["balanced"] | None = "balanced"
    max_iter: int = Field(default=3000, ge=100)


class TfidfConfig(LogRegConfig):
    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]
    min_df: int = Field(ge=1)
    max_features_word: int | None
    max_features_char: int | None
    sublinear_tf: bool


class EncoderConfig(_Strict):
    name: str
    prefix: str = ""


class EmbeddingsConfig(LogRegConfig):
    encoders: list[EncoderConfig] = Field(min_length=1)
    batch_size: int = Field(ge=1)

    def encoder(self, name: str) -> EncoderConfig:
        for enc in self.encoders:
            if enc.name == name:
                return enc
        raise KeyError(f"encoder '{name}' não está em configs/train.yaml")


class ModelsConfig(_Strict):
    tfidf: TfidfConfig
    embeddings: EmbeddingsConfig


class TrainConfig(_Strict):
    seed: int
    targets: list[Target] = Field(min_length=1)
    mlflow: MlflowConfig
    models: ModelsConfig


def load_train_config(path: Path = DEFAULT_TRAIN_CONFIG) -> TrainConfig:
    return load_yaml_config(path, TrainConfig)
