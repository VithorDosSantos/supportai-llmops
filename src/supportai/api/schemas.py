"""Contratos HTTP da API (Pydantic v2): validados na entrada e documentados no OpenAPI."""

from pydantic import BaseModel, ConfigDict, Field


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # O limite máximo é aplicado na rota (vem das settings, não é constante).
    text: str = Field(min_length=1, description="Assunto + corpo do ticket, em inglês.")


class LabelPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1, description="Probabilidade da classe prevista.")


class ModelVersion(BaseModel):
    name: str
    version: str


class ClassifyResponse(BaseModel):
    request_id: str
    category: LabelPrediction
    urgency: LabelPrediction
    models: dict[str, ModelVersion]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    models: dict[str, ModelVersion]
