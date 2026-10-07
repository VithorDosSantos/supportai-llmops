"""Carregamento dos classificadores de triagem para o serviço.

A API conhece só a interface `TriageModel` (texto -> rótulo + confiança). Se o
artefato é TF-IDF ou embeddings, ou se veio do registry ou de um diretório
exportado, é detalhe do loader. Nos testes, um modelo falso implementa a mesma
interface e nada do MLflow é carregado.
"""

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from supportai.classifier.export import EXPORT_METADATA_FILE

_REGISTRY_URI = re.compile(r"^models:/(?P<name>[^@/]+)@(?P<alias>[^/]+)$")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class ModelInfo:
    uri: str
    name: str
    version: str
    run_id: str | None


class TriageModel(Protocol):
    info: ModelInfo

    def predict(self, texts: Sequence[str]) -> list[Prediction]: ...


class SklearnTriageModel:
    """Pipeline sklearn (texto cru -> rótulo) com predict_proba."""

    def __init__(self, pipeline: Any, info: ModelInfo) -> None:
        self._pipeline = pipeline
        self.info = info

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        frame = pd.DataFrame({"text": list(texts)})
        proba = np.asarray(self._pipeline.predict_proba(frame))
        classes = self._pipeline.classes_
        best = proba.argmax(axis=1)
        return [
            Prediction(label=str(classes[i]), confidence=float(proba[row, i]))
            for row, i in enumerate(best)
        ]


def describe_model_uri(uri: str, client: MlflowClient | None = None) -> ModelInfo:
    """Resolve nome/versão para logs, /health e métricas.

    `models:/nome@alias` consulta o registry; um diretório exportado traz um
    arquivo de metadados escrito no momento da exportação.
    """
    match = _REGISTRY_URI.match(uri)
    if match:
        mv = (client or MlflowClient()).get_model_version_by_alias(match["name"], match["alias"])
        return ModelInfo(uri=uri, name=match["name"], version=str(mv.version), run_id=mv.run_id)

    metadata = Path(uri) / EXPORT_METADATA_FILE
    if metadata.exists():
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        return ModelInfo(
            uri=uri, name=meta["name"], version=str(meta["version"]), run_id=meta.get("run_id")
        )
    return ModelInfo(uri=uri, name=Path(uri).name, version="unknown", run_id=None)


def load_triage_model(uri: str, allow_deserialization: bool) -> SklearnTriageModel:
    if allow_deserialization:
        os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] = "true"
    info = describe_model_uri(uri)
    pipeline = mlflow.sklearn.load_model(uri)
    return SklearnTriageModel(pipeline, info)
