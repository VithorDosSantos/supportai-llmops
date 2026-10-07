"""API ponta a ponta com um modelo real: treina um TF-IDF minúsculo, registra no
MLflow (SQLite temporário), exporta e sobe a API pelos dois caminhos de carga:
registry (`models:/...@champion`) e diretório exportado. Sem rede e sem Postgres."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from supportai.api.main import create_app
from supportai.classifier import train
from supportai.classifier.config import load_train_config
from supportai.classifier.export import EXPORT_METADATA_FILE, export_champions
from supportai.classifier.registry import promote_champions
from supportai.config import APISettings, Settings

CFG = load_train_config()
TINY = CFG.model_copy(
    update={
        "models": CFG.models.model_copy(
            update={"tfidf": CFG.models.tfidf.model_copy(update={"min_df": 1, "c_grid": [1.0]})}
        ),
        "mlflow": CFG.mlflow.model_copy(
            update={"experiment": "it-triage", "registered_model_prefix": "it-triage"}
        ),
    }
)


def _df(n: int, offset: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "ticket_id": f"{offset + i:016x}",
                "text": f"Invoice charged twice, refund payment {offset + i}",
                "category": "billing_payments",
                "urgency": "high",
            }
        )
        rows.append(
            {
                "ticket_id": f"{offset + i + 10_000:016x}",
                "text": f"Router crashes, server error on the VPN {offset + i}",
                "category": "technical_support",
                "urgency": "low",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def registry(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Settings]:
    tmp = tmp_path_factory.mktemp("mlflow")
    settings = Settings(
        mlflow_tracking_uri=f"sqlite:///{tmp / 'mlflow.db'}",
        mlflow_artifact_root=tmp / "artifacts",
        api=APISettings(
            model_uri_category="models:/it-triage-category@champion",
            model_uri_urgency="models:/it-triage-urgency@champion",
            allow_model_deserialization=True,
            log_json=False,
        ),
    )
    train.setup_mlflow(settings, "it-triage")
    splits = {"train": _df(12, 0), "val": _df(4, 100), "test": _df(4, 200)}
    for target in TINY.targets:
        train.train_tfidf(TINY, splits, target)
    promote_champions(TINY)
    yield settings
    mlflow.set_tracking_uri(None)


def _assert_classifies(settings: Settings, expected_version: str) -> None:
    with TestClient(create_app(settings)) as client:
        health = client.get("/health").json()
        assert health["models"]["category"]["version"] == expected_version
        body = client.post("/classify", json={"text": "I was charged twice on my invoice"}).json()
        assert body["category"]["label"] == "billing_payments"
        assert body["urgency"]["label"] == "high"
        assert 0.5 < body["category"]["confidence"] <= 1.0


def test_api_loads_champions_from_registry(registry: Settings) -> None:
    _assert_classifies(registry, expected_version="1")


def test_api_loads_exported_models(registry: Settings, tmp_path: Path) -> None:
    exported = export_champions(TINY, tmp_path / "models")
    assert (exported["category"] / EXPORT_METADATA_FILE).exists()
    local = registry.model_copy(
        update={
            "api": registry.api.model_copy(
                update={
                    "model_uri_category": str(exported["category"]),
                    "model_uri_urgency": str(exported["urgency"]),
                }
            )
        }
    )
    _assert_classifies(local, expected_version="1")
