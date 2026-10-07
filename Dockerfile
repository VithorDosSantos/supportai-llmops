# Imagem do serviço de triagem. Multi-stage:
#   builder             -> resolve dependências com uv (compiladores e cache ficam aqui).
#                          Só `dependencies` do pyproject (serviço); grupos train/dev ficam fora.
#   runtime             -> só o venv + código; modelos montados em /app/models
#   runtime-with-models -> runtime + models/ embutido (deploy em PaaS sem volume)
#
# Build:  docker build --target runtime-with-models -t supportai-api .
#         (antes: uv run python -m supportai.classifier.export  -> gera models/)

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependências antes do código: mudar o código não invalida a camada pesada.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --no-install-project
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable

FROM python:${PYTHON_VERSION}-slim AS runtime
# Usuário sem privilégios: um RCE no processo não vira root no container.
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    API_MODEL_URI_CATEGORY=/app/models/category \
    API_MODEL_URI_URGENCY=/app/models/urgency \
    API_ALLOW_MODEL_DESERIALIZATION=true \
    MLFLOW_DISABLE_TELEMETRY=true \
    PORT=8000
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"
# PORT vem da plataforma (Render/Railway injetam); 1 worker por container e
# escala horizontal: os modelos ocupam memória por processo.
CMD ["sh", "-c", "exec uvicorn --factory supportai.api.main:create_app --host 0.0.0.0 --port ${PORT} --no-access-log"]

FROM runtime AS runtime-with-models
COPY --chown=app:app models /app/models
