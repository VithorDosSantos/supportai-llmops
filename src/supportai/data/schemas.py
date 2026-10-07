"""Contratos de dados (Pandera) nas fronteiras do pipeline.

Dois pontos de validação:
- **bruto**: o CSV de origem tem as colunas e tipos que a limpeza assume. Se o
  upstream mudar, o erro aparece aqui, com a coluna culpada, e não como um
  `KeyError` no meio da limpeza.
- **limpo/split**: o que o treino consome, com regras de negócio (rótulos
  válidos, texto não vazio, IDs únicos, splits sem vazamento de grupo).

`lazy=True` coleta todas as falhas de uma vez, útil para diagnosticar dados ruins.
"""

from collections.abc import Sequence

import pandas as pd
import pandera.pandas as pa

from supportai.data.config import URGENCY_LEVELS

SPLITS = ("train", "val", "test")


def raw_ticket_schema() -> pa.DataFrameSchema:
    """Somente as colunas usadas; as demais (tags, answer...) são ignoradas."""
    return pa.DataFrameSchema(
        {
            "subject": pa.Column(str, nullable=True),
            "body": pa.Column(str, nullable=True),
            "queue": pa.Column(str),
            "priority": pa.Column(str),
            "language": pa.Column(str),
        },
        strict=False,
        coerce=True,
    )


def _clean_columns(categories: Sequence[str], min_text_chars: int) -> dict[str, pa.Column]:
    return {
        "ticket_id": pa.Column(str, unique=True, checks=pa.Check.str_matches(r"^[0-9a-f]{16}$")),
        "subject": pa.Column(str),
        "body": pa.Column(str),
        "text": pa.Column(
            str,
            checks=[
                pa.Check.str_length(min_value=min_text_chars),
                # Sem espaços nas pontas: a limpeza normaliza espaços.
                pa.Check(lambda s: s == s.str.strip(), error="text não normalizado"),
            ],
        ),
        "category": pa.Column(str, checks=pa.Check.isin(list(categories))),
        "urgency": pa.Column(str, checks=pa.Check.isin(list(URGENCY_LEVELS))),
    }


def clean_ticket_schema(categories: Sequence[str], min_text_chars: int) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        _clean_columns(categories, min_text_chars),
        strict=True,
        coerce=True,
        unique=["text"],
    )


def split_ticket_schema(categories: Sequence[str], min_text_chars: int) -> pa.DataFrameSchema:
    columns = _clean_columns(categories, min_text_chars) | {
        "group_id": pa.Column(int, checks=pa.Check.ge(0)),
        "split": pa.Column(str, checks=pa.Check.isin(list(SPLITS))),
    }
    return pa.DataFrameSchema(
        columns,
        checks=[pa.Check(_groups_do_not_cross_splits, error="grupo em mais de um split")],
        strict=True,
        coerce=True,
        unique=["text"],
    )


def _groups_do_not_cross_splits(df: pd.DataFrame) -> bool:
    return bool(df.groupby("group_id")["split"].nunique().le(1).all())
