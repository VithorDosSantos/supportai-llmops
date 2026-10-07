"""Métricas Prometheus do serviço.

Um registry por app (e não o global do prometheus_client) para que cada app
criado nos testes tenha contadores isolados e não haja "Duplicated timeseries".
"""

from dataclasses import dataclass, field

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Buckets em segundos pensados para inferência em CPU (~ms) + overhead HTTP.
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5)


@dataclass
class APIMetrics:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        self.requests = Counter(
            "supportai_http_requests_total",
            "Requisições HTTP",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "supportai_http_request_duration_seconds",
            "Latência das requisições HTTP",
            ["method", "route"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.predictions = Counter(
            "supportai_predictions_total",
            "Predições por alvo e rótulo (base para monitorar drift de predição)",
            ["target", "label"],
            registry=self.registry,
        )
        self.model_info = Gauge(
            "supportai_model_info",
            "Modelo carregado (valor sempre 1; informação nos labels)",
            ["target", "name", "version"],
            registry=self.registry,
        )
