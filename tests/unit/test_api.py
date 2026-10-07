"""Testes da API com modelos falsos: rápidos e sem MLflow."""

import json
import logging
from collections.abc import Iterator, Mapping, Sequence

import pytest
from fastapi.testclient import TestClient

from supportai.api.logging import JsonFormatter, request_id_var
from supportai.api.main import create_app
from supportai.api.models import ModelInfo, Prediction, TriageModel
from supportai.config import APISettings, Settings


class FakeModel:
    def __init__(self, target: str, label: str) -> None:
        self.info = ModelInfo(
            uri=f"fake://{target}", name=f"fake-{target}", version="7", run_id=None
        )
        self.label = label
        self.calls: list[list[str]] = []

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        self.calls.append(list(texts))
        return [Prediction(self.label, 0.9) for _ in texts]


@pytest.fixture
def models() -> dict[str, FakeModel]:
    return {
        "category": FakeModel("category", "billing_payments"),
        "urgency": FakeModel("urgency", "high"),
    }


@pytest.fixture
def client(models: dict[str, FakeModel]) -> Iterator[TestClient]:
    settings = Settings(api=APISettings(max_text_chars=200, log_json=True))

    def loader(_: Settings) -> Mapping[str, TriageModel]:
        return models

    with TestClient(create_app(settings, model_loader=loader)) as c:
        yield c


def test_health_reports_loaded_models(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "models": {
            "category": {"name": "fake-category", "version": "7"},
            "urgency": {"name": "fake-urgency", "version": "7"},
        },
    }


def test_classify_returns_both_targets(client: TestClient, models: dict[str, FakeModel]) -> None:
    response = client.post("/classify", json={"text": "I was charged twice"})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == {"label": "billing_payments", "confidence": 0.9}
    assert body["urgency"] == {"label": "high", "confidence": 0.9}
    assert body["models"]["category"]["version"] == "7"
    assert body["latency_ms"] >= 0
    assert models["category"].calls == [["I was charged twice"]]


def test_request_id_is_propagated_or_generated(client: TestClient) -> None:
    given = client.post("/classify", json={"text": "hello"}, headers={"X-Request-ID": "abc-1"})
    assert given.headers["X-Request-ID"] == "abc-1"
    assert given.json()["request_id"] == "abc-1"

    generated = client.get("/health")
    assert len(generated.headers["X-Request-ID"]) == 32


@pytest.mark.parametrize(
    "payload",
    [{"text": ""}, {}, {"text": "ok", "extra": 1}, {"text": "x" * 201}],
    ids=["empty", "missing", "extra-field", "too-long"],
)
def test_invalid_input_is_rejected(client: TestClient, payload: dict[str, object]) -> None:
    assert client.post("/classify", json=payload).status_code == 422


def test_metrics_count_requests_and_predictions(client: TestClient) -> None:
    client.post("/classify", json={"text": "hello"})
    client.post("/classify", json={"text": ""})
    text = client.get("/metrics").text
    assert 'supportai_http_requests_total{method="POST",route="/classify",status="200"} 1.0' in text
    assert 'supportai_http_requests_total{method="POST",route="/classify",status="422"} 1.0' in text
    assert 'supportai_predictions_total{label="billing_payments",target="category"} 1.0' in text
    assert 'supportai_model_info{name="fake-urgency",target="urgency",version="7"} 1.0' in text


def test_unknown_route_uses_bounded_label(client: TestClient) -> None:
    client.get("/does-not-exist/123")
    assert 'route="unmatched",status="404"' in client.get("/metrics").text


def test_startup_fails_when_a_model_is_missing() -> None:
    app = create_app(Settings(), model_loader=lambda _: {})
    with pytest.raises(RuntimeError, match="modelos ausentes"), TestClient(app):
        pass


def test_json_formatter_includes_request_id_and_extra() -> None:
    token = request_id_var.set("rid-9")
    try:
        record = logging.makeLogRecord(
            {"msg": "hi", "levelname": "INFO", "name": "t", "status": 200}
        )
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "rid-9"
    assert payload["status"] == 200
    assert payload["message"] == "hi"
