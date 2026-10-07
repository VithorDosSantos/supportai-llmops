"""Limpeza: do CSV bruto ao dataset rotulado de triagem.

Cada passo é uma função pura e pequena (fácil de testar) e o relatório guarda
quantas linhas cada filtro removeu: sem isso, uma mudança no upstream que
descarta metade dos dados passaria despercebida.
"""

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from supportai.data.config import FilterConfig
from supportai.data.schemas import clean_ticket_schema, raw_ticket_schema

_WHITESPACE = r"\s+"


@dataclass
class CleanReport:
    """Contagem de linhas após cada etapa, na ordem em que foram aplicadas."""

    steps: dict[str, int] = field(default_factory=dict)

    def record(self, step: str, df: pd.DataFrame) -> pd.DataFrame:
        self.steps[step] = len(df)
        return df


def normalize_whitespace(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(_WHITESPACE, " ", regex=True).str.strip()


def build_text(subject: pd.Series, body: pd.Series) -> pd.Series:
    """Assunto + corpo: o assunto costuma ser o resumo mais informativo.

    ~16% dos tickets não têm assunto; nesses casos o texto é só o corpo.
    """
    subj = normalize_whitespace(subject)
    text = subj.where(subj == "", subj + ". ") + normalize_whitespace(body)
    return text.str.strip()


def map_priority(priority: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Prioridades desconhecidas viram NaN e são descartadas depois (não chutamos)."""
    return priority.str.strip().str.lower().map(mapping)


def ticket_id(text: pd.Series) -> pd.Series:
    """ID estável derivado do conteúdo: o mesmo ticket tem o mesmo ID em toda execução."""
    return text.map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16])


def clean_tickets(raw: pd.DataFrame, cfg: FilterConfig) -> tuple[pd.DataFrame, CleanReport]:
    report = CleanReport()
    df = report.record("raw", raw_ticket_schema().validate(raw, lazy=True))

    df = report.record("language", df[df["language"].str.strip().str.lower() == cfg.language])
    df = report.record("queue", df[df["queue"].isin(cfg.queues.keys())])

    df = df.assign(
        subject=normalize_whitespace(df["subject"]),
        body=normalize_whitespace(df["body"]),
        text=build_text(df["subject"], df["body"]),
        category=df["queue"].map(cfg.queues),
        urgency=map_priority(df["priority"], cfg.priority_map),
    )
    df = report.record("priority", df.dropna(subset=["urgency"]))
    df = report.record("min_text_chars", df[df["text"].str.len() >= cfg.min_text_chars])
    # Duplicatas exatas (ignorando caixa): mantêm a primeira ocorrência.
    df = report.record("exact_duplicates", df[~df["text"].str.lower().duplicated()])

    df = df.assign(ticket_id=ticket_id(df["text"]))
    columns = ["ticket_id", "subject", "body", "text", "category", "urgency"]
    clean = df[columns].reset_index(drop=True)
    schema = clean_ticket_schema(cfg.categories, cfg.min_text_chars)
    return schema.validate(clean, lazy=True), report
