"""Testes do classificador sem rede: TF-IDF real em dados minúsculos e um
encoder falso no lugar do SentenceTransformer (nada é baixado)."""

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from supportai.classifier import models, train
from supportai.classifier.config import load_train_config
from supportai.classifier.metrics import evaluate, plot_confusion_matrix
from supportai.classifier.models import (
    SentenceEncoder,
    build_tfidf_pipeline,
    compact_tfidf_vocabulary,
    to_text_list,
)
from supportai.classifier.registry import promote_champions
from supportai.config import Settings

CFG = load_train_config()
TINY_TFIDF = CFG.models.tfidf.model_copy(update={"min_df": 1, "c_grid": [0.5, 2.0]})
TINY_CFG = CFG.model_copy(update={"models": CFG.models.model_copy(update={"tfidf": TINY_TFIDF})})

_VOCAB = {
    "billing_payments": ["invoice", "charged twice", "payment failed", "credit card"],
    "technical_support": ["router down", "app crashes", "server error", "vpn broken"],
}


def _split_df(n: int, offset: int) -> pd.DataFrame:
    rows = []
    for category, phrases in _VOCAB.items():
        for i in range(n):
            phrase = phrases[i % len(phrases)]
            rows.append(
                {
                    "ticket_id": hashlib.sha1(f"{category}{i}{offset}".encode()).hexdigest()[:16],
                    "text": f"Hello, {phrase} on order {offset + i}. Please help.",
                    "category": category,
                    "urgency": ["low", "medium", "high"][i % 3],
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def splits() -> dict[str, pd.DataFrame]:
    return {"train": _split_df(24, 0), "val": _split_df(6, 1000), "test": _split_df(6, 2000)}


class FakeEncoder:
    """Embedding determinístico por hash de palavras: separável o bastante para os testes."""

    calls = 0

    def encode(self, sentences: list[str], **_: Any) -> np.ndarray:
        FakeEncoder.calls += 1
        out = np.zeros((len(sentences), 32), dtype=np.float32)
        for row, sentence in enumerate(sentences):
            for word in sentence.lower().split():
                out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 32] += 1.0
        return out


@pytest.fixture
def fake_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "load_sentence_transformer", lambda name: FakeEncoder())


@pytest.fixture
def mlflow_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")
    settings = Settings(
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_artifact_root=tmp_path / "artifacts",
    )
    monkeypatch.setattr(train, "REPORTS_DIR", tmp_path / "reports")
    train.setup_mlflow(settings, "test-triage")
    yield settings
    mlflow.set_tracking_uri(None)


def test_to_text_list_accepts_dataframe_series_and_list() -> None:
    expected = ["a", "b"]
    assert to_text_list(pd.DataFrame({"text": expected, "other": [1, 2]})) == expected
    assert to_text_list(pd.Series(expected)) == expected
    assert to_text_list(expected) == expected


def test_evaluate_metrics_on_known_predictions() -> None:
    result = evaluate(["a", "a", "b", "b"], ["a", "b", "b", "b"], labels=["a", "b"])
    assert result.accuracy == pytest.approx(0.75)
    # F1(a)=2/3, F1(b)=0.8 -> macro = 0.7333
    assert result.f1_macro == pytest.approx((2 / 3 + 0.8) / 2)
    assert result.confusion.tolist() == [[1, 1], [0, 2]]
    assert set(result.as_metrics("test")) == {
        "test_f1_macro",
        "test_accuracy",
        "test_f1_a",
        "test_f1_b",
    }
    plot_confusion_matrix(result, "t")


def test_tfidf_pipeline_learns_tiny_task(splits: dict[str, pd.DataFrame]) -> None:
    pipe = build_tfidf_pipeline(TINY_TFIDF, c=1.0, seed=42)
    pipe.fit(splits["train"][["text"]], splits["train"]["category"])
    pred = pipe.predict(splits["test"][["text"]])
    assert (pred == splits["test"]["category"]).mean() == 1.0

    # Vocabulário com int nativo (serialização rápida no skops), mesmas predições.
    compact_tfidf_vocabulary(pipe)
    for _, vec in pipe.named_steps["features"].transformer_list:
        assert all(type(v) is int for v in vec.vocabulary_.values())
    assert (pipe.predict(splits["test"][["text"]]) == pred).all()


def test_grid_search_picks_best_c_on_validation() -> None:
    class Const:
        def __init__(self, label: str) -> None:
            self.label = label

        def predict(self, x: list[str]) -> list[str]:
            return [self.label] * len(x)

    y_val = ["a", "a", "a", "b"]
    outcome = train.grid_search_c(
        [1.0, 2.0], lambda c: Const("a" if c == 2.0 else "b"), ["x"] * 4, y_val, ["a", "b"]
    )
    assert outcome.best_c == 2.0
    assert set(outcome.val_f1_by_c) == {1.0, 2.0}


def test_sentence_encoder_is_lazy_and_not_serialized(fake_encoder: None) -> None:
    enc = SentenceEncoder("fake/model", prefix="query: ")
    assert "_encoder" not in enc.__dict__
    vectors = enc.transform(["router down", "invoice"])
    assert vectors.shape == (2, 32)
    assert "_encoder" in enc.__dict__
    assert "_encoder" not in enc.__getstate__()


def test_train_register_and_load_champion(
    splits: dict[str, pd.DataFrame], mlflow_tmp: Settings, fake_encoder: None
) -> None:
    """Ponta a ponta: treina TF-IDF e embeddings, promove o campeão, carrega pelo alias."""
    tfidf_runs = [train.train_tfidf(TINY_CFG, splits, t) for t in TINY_CFG.targets]
    encoder = TINY_CFG.models.embeddings.encoders[0]
    embeddings = train.encode_splits(encoder, 8, splits)
    emb_runs = [
        train.train_embeddings(TINY_CFG, splits, t, encoder, embeddings) for t in TINY_CFG.targets
    ]
    train.write_summary(tfidf_runs + emb_runs, "test")

    client = MlflowClient()
    promotions = promote_champions(TINY_CFG.model_copy(update={"mlflow": _mlflow_cfg()}), client)
    assert {p.target for p in promotions} == {"category", "urgency"}
    assert all(p.changed for p in promotions)

    # Escolha pela validação, não pelo teste.
    for p in promotions:
        candidates = [r for r in tfidf_runs + emb_runs if r.target == p.target]
        assert p.selection_metric == max(r.val_f1_macro for r in candidates)

    # Idempotente: promover de novo não cria versão nova.
    again = promote_champions(TINY_CFG.model_copy(update={"mlflow": _mlflow_cfg()}), client)
    assert not any(p.changed for p in again)

    model = mlflow.pyfunc.load_model("models:/test-triage-category@champion")
    pred = model.predict(pd.DataFrame({"text": ["I was charged twice on my invoice"]}))
    assert list(pred) == ["billing_payments"]


def test_embedding_pipeline_roundtrips_through_mlflow(
    splits: dict[str, pd.DataFrame], mlflow_tmp: Settings, fake_encoder: None
) -> None:
    encoder = TINY_CFG.models.embeddings.encoders[0]
    embeddings = train.encode_splits(encoder, 8, splits)
    summary = train.train_embeddings(TINY_CFG, splits, "category", encoder, embeddings)

    uri = MlflowClient().get_run(summary.run_id).data.tags["model_uri"]
    loaded = mlflow.sklearn.load_model(uri)
    assert "_encoder" not in loaded.named_steps["encoder"].__dict__
    pred = loaded.predict(pd.DataFrame({"text": ["the app crashes, server error"]}))
    assert list(pred) == ["technical_support"]


def _mlflow_cfg() -> Any:
    return TINY_CFG.mlflow.model_copy(
        update={"experiment": "test-triage", "registered_model_prefix": "test-triage"}
    )
