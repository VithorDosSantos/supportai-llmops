"""Exporta os campeões do Model Registry para um diretório local.

Por que: o registry é um SQLite local (ADR 0004). Um container ou um deploy no
Render/Railway não enxerga esse arquivo, então copiamos o artefato do campeão
para `models/<alvo>/` e o embutimos na imagem. A versão fica registrada num
arquivo de metadados, para a API reportar exatamente qual modelo está servindo.

Uso: `python -m supportai.classifier.export [--out models]`
"""

import argparse
import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from supportai.classifier.config import TrainConfig, load_train_config
from supportai.config import PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)

EXPORT_METADATA_FILE = "registry.json"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "models"


def export_champions(
    cfg: TrainConfig, out_dir: Path, client: MlflowClient | None = None
) -> dict[str, Path]:
    client = client or MlflowClient()
    exported: dict[str, Path] = {}
    for target in cfg.targets:
        name = cfg.mlflow.registered_model_name(target)
        mv = client.get_model_version_by_alias(name, cfg.mlflow.champion_alias)
        dest = out_dir / target
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{name}/{mv.version}", dst_path=str(dest)
        )
        # download_artifacts pode criar um subdiretório; normalizamos para `dest`.
        local_path = Path(local)
        if local_path != dest:
            for item in local_path.iterdir():
                shutil.move(str(item), dest / item.name)
            local_path.rmdir()
        metadata = {
            "name": name,
            "version": mv.version,
            "alias": cfg.mlflow.champion_alias,
            "run_id": mv.run_id,
        }
        (dest / EXPORT_METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        exported[target] = dest
        logger.info("%s v%s -> %s", name, mv.version, dest)
    return exported


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exporta os campeões do registry")
    parser.add_argument("--out", type=Path, default=DEFAULT_EXPORT_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    export_champions(load_train_config(), args.out)


if __name__ == "__main__":
    main()
