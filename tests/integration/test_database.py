import psycopg
import pytest

pytestmark = pytest.mark.integration


def test_pgvector_extension_is_installed(db_conn: psycopg.Connection) -> None:
    row = db_conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None, "rode `docker compose up -d db` (o init cria a extensão)"


def test_vector_similarity_search(db_conn: psycopg.Connection) -> None:
    """Fumaça do que o RAG vai precisar: ordenar por distância de cosseno."""
    db_conn.execute("CREATE TEMP TABLE t_smoke (id int, emb vector(3))")
    db_conn.execute("INSERT INTO t_smoke VALUES (1, '[1,0,0]'), (2, '[0,1,0]'), (3, '[0.9,0.1,0]')")
    rows = db_conn.execute("SELECT id FROM t_smoke ORDER BY emb <=> '[1,0,0]' LIMIT 2").fetchall()
    assert [r[0] for r in rows] == [1, 3]
