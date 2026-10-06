from collections.abc import Iterator

import psycopg
import pytest

from supportai.config import DatabaseSettings


@pytest.fixture(scope="session")
def db_conn() -> Iterator[psycopg.Connection]:
    """Conexão com o Postgres do docker compose; pula o teste se indisponível.

    Pular (em vez de falhar) mantém `pytest` útil sem Docker; no CI o serviço
    sobe e os testes de integração rodam de verdade.
    """
    try:
        conn = psycopg.connect(DatabaseSettings().dsn, connect_timeout=3, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres indisponível: {exc}")
    yield conn
    conn.close()
