"""Memória (RSS) do processo da API: quanto vem de imports e quanto de cada modelo.

Uso: uv run python scripts/api_memory.py  (lê as URIs de modelo das settings)
"""

import gc
import os
from pathlib import Path


def rss_mb() -> int:
    status = Path("/proc/self/status").read_text()
    return int(status.split("VmRSS:")[1].split()[0]) // 1024


def main() -> None:
    os.environ["MLFLOW_ALLOW_PICKLE_DESERIALIZATION"] = "true"
    print(f"python vazio: {rss_mb()} MB")

    import mlflow

    import supportai.api.main  # noqa: F401  (FastAPI, MLflow, scikit-learn, pandas)
    from supportai.config import get_settings

    print(f"após imports: {rss_mb()} MB")
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    models = []
    for target, uri in (
        ("categoria", settings.api.model_uri_category),
        ("urgência", settings.api.model_uri_urgency),
    ):
        models.append(mlflow.sklearn.load_model(uri))
        gc.collect()
        print(f"+ modelo {target}: {rss_mb()} MB")


if __name__ == "__main__":
    main()
