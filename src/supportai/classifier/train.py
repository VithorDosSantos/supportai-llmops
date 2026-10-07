"""Treino dos classificadores de triagem com rastreamento no MLflow.

Fluxo por (família de modelo, alvo):
1. grid de C avaliado na **validação**;
2. o melhor C é avaliado **uma vez** no teste;
3. params, métricas, matrizes de confusão e o pipeline (texto -> rótulo) vão
   para um run do MLflow, com tags de commit do git e hash dos dados (dvc.lock).

Uso:
    python -m supportai.classifier.train --model tfidf
    python -m supportai.classifier.train --model embeddings [--encoder NOME]
"""

import argparse
import json
import logging
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from supportai.classifier.config import (
    EncoderConfig,
    Target,
    TrainConfig,
    load_train_config,
)
from supportai.classifier.metrics import EvalResult, evaluate, plot_confusion_matrix
from supportai.classifier.models import (
    SKOPS_TRUSTED_TYPES,
    SentenceEncoder,
    build_embedding_pipeline,
    build_logreg,
    build_tfidf_pipeline,
    compact_tfidf_vocabulary,
)
from supportai.config import PROJECT_ROOT, Settings, get_settings
from supportai.data.config import load_data_config

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")
REPORTS_DIR = PROJECT_ROOT / "reports"


class Predictor(Protocol):
    def predict(self, x: Any) -> Any: ...


@dataclass(frozen=True)
class GridOutcome:
    best_c: float
    best_model: Predictor
    val_result: EvalResult
    val_f1_by_c: dict[float, float]


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    model_family: str
    target: str
    best_c: float
    val_f1_macro: float
    test_f1_macro: float
    test_accuracy: float


def load_splits(processed_dir: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(processed_dir / f"{name}.parquet") for name in SPLIT_NAMES}


def grid_search_c(
    c_grid: Sequence[float],
    fit: Callable[[float], Predictor],
    x_val: Any,
    y_val: Sequence[str],
    labels: Sequence[str],
) -> GridOutcome:
    """Escolhe C pelo F1 macro na validação. Empate: o menor C (mais regularizado)."""
    best: tuple[float, Predictor, EvalResult] | None = None
    scores: dict[float, float] = {}
    for c in sorted(c_grid):
        model = fit(c)
        result = evaluate(y_val, model.predict(x_val), labels)
        scores[c] = result.f1_macro
        logger.info("  C=%-6g val_f1_macro=%.4f", c, result.f1_macro)
        if best is None or result.f1_macro > best[2].f1_macro:
            best = (c, model, result)
    assert best is not None  # c_grid tem min_length=1 na config
    return GridOutcome(best[0], best[1], best[2], scores)


# --- rastreabilidade -------------------------------------------------------


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def data_hashes(lock_path: Path = PROJECT_ROOT / "dvc.lock") -> dict[str, str]:
    """md5 de cada split segundo o dvc.lock: liga o run do MLflow à versão exata dos dados."""
    if not lock_path.exists():
        return {}
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    outs = lock.get("stages", {}).get("prepare", {}).get("outs", [])
    return {f"data_md5_{Path(o['path']).stem}": o["md5"] for o in outs if "md5" in o}


def setup_mlflow(settings: Settings, experiment: str) -> str:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        artifact_root = settings.mlflow_artifact_root
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id: str = mlflow.create_experiment(
            experiment, artifact_location=artifact_root.resolve().as_uri()
        )
    else:
        experiment_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def _log_eval_artifacts(result: EvalResult, split: str, title: str, tmp: Path) -> None:
    fig = plot_confusion_matrix(result, f"{title} ({split})")
    fig.savefig(tmp / f"confusion_{split}.png", dpi=120)
    plt.close(fig)
    report = {
        "f1_macro": result.f1_macro,
        "accuracy": result.accuracy,
        "per_class": result.per_class,
    }
    (tmp / f"report_{split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cm = pd.DataFrame(result.confusion, index=result.labels, columns=result.labels)
    cm.to_csv(tmp / f"confusion_{split}.csv")


def log_run(
    *,
    model_family: str,
    target: Target,
    params: dict[str, Any],
    grid: GridOutcome,
    test_result: EvalResult,
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    test_pred: Sequence[str],
) -> RunSummary:
    run_name = f"{model_family}-{target}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {"target": target, "model_family": model_family, "git_commit": git_commit()}
            | data_hashes()
        )
        mlflow.log_params(params | {"best_C": grid.best_c})
        mlflow.log_metrics(grid.val_result.as_metrics("val") | test_result.as_metrics("test"))
        for c, f1 in grid.val_f1_by_c.items():
            mlflow.log_metric(f"grid_val_f1_macro_C_{c:g}", f1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            _log_eval_artifacts(grid.val_result, "val", run_name, tmp)
            _log_eval_artifacts(test_result, "test", run_name, tmp)
            # Erros do teste para análise qualitativa (só os errados, para ficar pequeno).
            errors = test_df.assign(predicted=list(test_pred))
            errors = errors[errors[target] != errors["predicted"]]
            errors[["ticket_id", "text", target, "predicted"]].to_csv(
                tmp / "test_errors.csv", index=False
            )
            mlflow.log_artifacts(str(tmp), artifact_path="evaluation")

        example = pd.DataFrame({"text": test_df["text"].head(3).tolist()})
        signature = infer_signature(example, pipeline.predict(example))
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=example,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        # O registro de campeão (registry.py) usa esta URI do modelo logado.
        mlflow.set_tag("model_uri", info.model_uri)

    return RunSummary(
        run_id=run.info.run_id,
        model_family=model_family,
        target=target,
        best_c=grid.best_c,
        val_f1_macro=grid.val_result.f1_macro,
        test_f1_macro=test_result.f1_macro,
        test_accuracy=test_result.accuracy,
    )


# --- famílias de modelo ----------------------------------------------------


def train_tfidf(cfg: TrainConfig, splits: dict[str, pd.DataFrame], target: Target) -> RunSummary:
    mcfg = cfg.models.tfidf
    labels = sorted(splits["train"][target].unique())
    train, val, test = (splits[s] for s in SPLIT_NAMES)

    def fit(c: float) -> Predictor:
        pipeline = build_tfidf_pipeline(mcfg, c, cfg.seed).fit(train[["text"]], train[target])
        model: Predictor = compact_tfidf_vocabulary(pipeline)
        return model

    grid = grid_search_c(mcfg.c_grid, fit, val[["text"]], val[target].tolist(), labels)
    pipeline: Pipeline = grid.best_model
    test_pred = list(pipeline.predict(test[["text"]]))
    return log_run(
        model_family="tfidf",
        target=target,
        params={"seed": cfg.seed} | mcfg.model_dump(exclude={"c_grid"}) | {"c_grid": mcfg.c_grid},
        grid=grid,
        test_result=evaluate(test[target].tolist(), test_pred, labels),
        pipeline=pipeline,
        test_df=test,
        test_pred=test_pred,
    )


def encode_splits(
    encoder: EncoderConfig, batch_size: int, splits: dict[str, pd.DataFrame]
) -> dict[str, npt.NDArray[np.float32]]:
    """Encoda cada split uma única vez; reutilizado pelos dois alvos e por todo o grid."""
    transformer = SentenceEncoder(encoder.name, encoder.prefix, batch_size)
    return {name: transformer.transform(df["text"].tolist()) for name, df in splits.items()}


def train_embeddings(
    cfg: TrainConfig,
    splits: dict[str, pd.DataFrame],
    target: Target,
    encoder: EncoderConfig,
    embeddings: dict[str, npt.NDArray[np.float32]],
) -> RunSummary:
    mcfg = cfg.models.embeddings
    labels = sorted(splits["train"][target].unique())
    y_train = splits["train"][target]

    def fit(c: float) -> Predictor:
        model: Predictor = build_logreg(mcfg, c, cfg.seed).fit(embeddings["train"], y_train)
        return model

    grid = grid_search_c(
        mcfg.c_grid, fit, embeddings["val"], splits["val"][target].tolist(), labels
    )
    test_pred = list(grid.best_model.predict(embeddings["test"]))
    pipeline = build_embedding_pipeline(encoder, mcfg.batch_size, grid.best_model)
    return log_run(
        model_family=f"emb-{encoder.name.split('/')[-1]}",
        target=target,
        params={
            "seed": cfg.seed,
            "encoder": encoder.name,
            "encoder_prefix": encoder.prefix,
            "class_weight": mcfg.class_weight,
            "max_iter": mcfg.max_iter,
            "c_grid": mcfg.c_grid,
        },
        grid=grid,
        test_result=evaluate(splits["test"][target].tolist(), test_pred, labels),
        pipeline=pipeline,
        test_df=splits["test"],
        test_pred=test_pred,
    )


def write_summary(summaries: list[RunSummary], name: str) -> Path:
    """Resumo versionado no git (via DVC metrics) para diffs entre experimentos."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"train_{name}.json"
    payload = {
        f"{s.model_family}/{s.target}": {
            "run_id": s.run_id,
            "best_C": s.best_c,
            "val_f1_macro": round(s.val_f1_macro, 4),
            "test_f1_macro": round(s.test_f1_macro, 4),
            "test_accuracy": round(s.test_accuracy, 4),
        }
        for s in summaries
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--model", choices=["tfidf", "embeddings"], required=True)
    parser.add_argument("--encoder", help="nome do encoder (padrão: todos de train.yaml)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_train_config()
    settings = get_settings()
    splits = load_splits(load_data_config().paths.resolve().processed_dir)
    setup_mlflow(settings, cfg.mlflow.experiment)

    summaries: list[RunSummary] = []
    if args.model == "tfidf":
        for target in cfg.targets:
            logger.info("tfidf / %s", target)
            summaries.append(train_tfidf(cfg, splits, target))
        name = "tfidf"
    else:
        ecfg = cfg.models.embeddings
        encoders = [ecfg.encoder(args.encoder)] if args.encoder else ecfg.encoders
        for encoder in encoders:
            logger.info("encodando splits com %s", encoder.name)
            embeddings = encode_splits(encoder, ecfg.batch_size, splits)
            for target in cfg.targets:
                logger.info("%s / %s", encoder.name, target)
                summaries.append(train_embeddings(cfg, splits, target, encoder, embeddings))
        name = "embeddings" if not args.encoder else f"emb_{args.encoder.split('/')[-1]}"

    path = write_summary(summaries, name)
    logger.info("resumo em %s", path)


if __name__ == "__main__":
    main()
