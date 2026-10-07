"""Logging estruturado (JSON) com request id.

Uma linha JSON por evento é o formato que agregadores (Loki, CloudWatch,
Datadog) indexam sem regex. O request id vem do header `X-Request-ID` (se um
proxy/gateway já gerou um) ou é criado aqui, volta na resposta e aparece em
todo log da requisição: é o que liga a reclamação do usuário ao log certo.
"""

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Atributos padrão do LogRecord; o resto (passado via `extra=`) vai para o JSON.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        payload.update({k: v for k, v in vars(record).items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # O access log do uvicorn duplicaria o nosso (que já tem latência e request id).
    logging.getLogger("uvicorn.access").disabled = True
