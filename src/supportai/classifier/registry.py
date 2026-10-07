"""Promoção do campeão no MLflow Model Registry.

Para cada alvo, o run com maior métrica de **validação** (nunca a de teste)
vira uma versão do modelo registrado e recebe o alias `champion`. A API
(Fase 2) carrega `models:/supportai-triage-<alvo>@champion`: trocar de modelo
em produção é mover o alias, sem mudar código nem redeploy.

Uso: `python -m supportai.classifier.registry`
"""

import logging
from dataclasses import dataclass

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from supportai.classifier.config import TrainConfig, load_train_config
from supportai.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Promotion:
    target: str
    model_name: str
    version: str
    run_id: str
    selection_metric: float
    changed: bool


def best_run_id(client: MlflowClient, experiment_id: str, target: str, metric: str) -> str:
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.target = '{target}' and attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise LookupError(f"nenhum run finalizado para o alvo '{target}'")
    run_id: str = runs[0].info.run_id
    return run_id


def _current_champion_run(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        run_id: str = client.get_model_version_by_alias(name, alias).run_id
    except MlflowException:
        return None
    return run_id


def promote_champions(cfg: TrainConfig, client: MlflowClient | None = None) -> list[Promotion]:
    client = client or MlflowClient()
    experiment = client.get_experiment_by_name(cfg.mlflow.experiment)
    if experiment is None:
        raise LookupError(f"experimento '{cfg.mlflow.experiment}' não existe; rode o treino antes")

    promotions = []
    alias = cfg.mlflow.champion_alias
    metric = cfg.mlflow.selection_metric
    for target in cfg.targets:
        name = cfg.mlflow.registered_model_name(target)
        run_id = best_run_id(client, experiment.experiment_id, target, metric)
        run = client.get_run(run_id)
        score = float(run.data.metrics[metric])

        # Idempotente: não cria versão nova se o campeão já é este run.
        if _current_champion_run(client, name, alias) == run_id:
            version = client.get_model_version_by_alias(name, alias).version
            promotions.append(Promotion(target, name, str(version), run_id, score, changed=False))
            continue

        mv = mlflow.register_model(
            run.data.tags["model_uri"],
            name,
            tags={metric: f"{score:.4f}", "model_family": run.data.tags.get("model_family", "")},
        )
        client.set_registered_model_alias(name, alias, mv.version)
        promotions.append(Promotion(target, name, str(mv.version), run_id, score, changed=True))
    return promotions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    for p in promote_champions(load_train_config()):
        status = "promovido" if p.changed else "já era o campeão"
        logger.info(
            "%s@champion -> v%s (run %s, val=%.4f): %s",
            p.model_name,
            p.version,
            p.run_id,
            p.selection_metric,
            status,
        )


if __name__ == "__main__":
    main()
