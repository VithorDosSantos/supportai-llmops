"""Serviço de triagem (FastAPI).

Rodar localmente: `uv run uvicorn --factory supportai.api.main:create_app --port 8000`

App factory (em vez de um `app` global): importar o módulo não lê settings nem
carrega modelos, e cada teste cria um app isolado com um loader falso.
"""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.routing import APIRoute
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from supportai.api.logging import configure_logging, request_id_var
from supportai.api.metrics import APIMetrics
from supportai.api.models import TriageModel, load_triage_model
from supportai.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    LabelPrediction,
    ModelVersion,
)
from supportai.config import Settings, get_settings

logger = logging.getLogger("supportai.api")

REQUEST_ID_HEADER = "X-Request-ID"
TARGETS = ("category", "urgency")

ModelLoader = Callable[[Settings], Mapping[str, TriageModel]]


def get_models(request: Request) -> dict[str, TriageModel]:
    models: dict[str, TriageModel] = request.app.state.models
    return models


Models = Annotated[dict[str, TriageModel], Depends(get_models)]


def load_models_from_settings(settings: Settings) -> dict[str, TriageModel]:
    """Carrega os dois campeões. Falha no startup (não na 1ª requisição) se algo faltar."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uris = {
        "category": settings.api.model_uri_category,
        "urgency": settings.api.model_uri_urgency,
    }
    allow = settings.api.allow_model_deserialization
    return {target: load_triage_model(uri, allow) for target, uri in uris.items()}


def create_app(
    settings: Settings | None = None, model_loader: ModelLoader = load_models_from_settings
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.api.log_json)
    metrics = APIMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        start = time.perf_counter()
        models = dict(model_loader(settings))
        missing = set(TARGETS) - set(models)
        if missing:
            raise RuntimeError(f"modelos ausentes: {sorted(missing)}")
        app.state.models = models
        for target, model in models.items():
            metrics.model_info.labels(target, model.info.name, model.info.version).set(1)
        logger.info(
            "modelos carregados",
            extra={
                "models": {t: f"{m.info.name}:v{m.info.version}" for t, m in models.items()},
                "load_s": round(time.perf_counter() - start, 2),
            },
        )
        yield

    app = FastAPI(
        title="SupportAI - triagem",
        version="0.1.0",
        description="Classifica tickets de suporte por categoria e urgência.",
        lifespan=lifespan,
    )
    app.state.metrics = metrics

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            # Rota (template), não o path cru: evita explosão de cardinalidade nas métricas.
            route = request.scope.get("route")
            route_path = route.path if isinstance(route, APIRoute) else "unmatched"
            metrics.requests.labels(request.method, route_path, str(status)).inc()
            metrics.latency.labels(request.method, route_path).observe(elapsed)
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 2),
                },
            )
            request_id_var.reset(token)

    def versions(models: Mapping[str, TriageModel]) -> dict[str, ModelVersion]:
        return {
            t: ModelVersion(name=m.info.name, version=m.info.version) for t, m in models.items()
        }

    @app.get("/health", response_model=HealthResponse)
    def health(models: Models) -> HealthResponse:
        return HealthResponse(status="ok", models=versions(models))

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # `def` (não `async def`): a inferência é CPU-bound e síncrona; o FastAPI a
    # roda num threadpool, sem bloquear o event loop para as outras requisições.
    @app.post("/classify", response_model=ClassifyResponse)
    def classify(body: ClassifyRequest, models: Models) -> ClassifyResponse:
        if len(body.text) > settings.api.max_text_chars:
            raise HTTPException(
                status_code=422,
                detail=f"text excede {settings.api.max_text_chars} caracteres",
            )
        start = time.perf_counter()
        preds = {t: models[t].predict([body.text])[0] for t in TARGETS}
        for target, pred in preds.items():
            metrics.predictions.labels(target, pred.label).inc()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "classificação",
            extra={
                "category": preds["category"].label,
                "urgency": preds["urgency"].label,
                "inference_ms": latency_ms,
                "text_chars": len(body.text),
            },
        )
        return ClassifyResponse(
            request_id=request_id_var.get() or "",
            category=LabelPrediction(**vars(preds["category"])),
            urgency=LabelPrediction(**vars(preds["urgency"])),
            models=versions(models),
            latency_ms=latency_ms,
        )

    return app
