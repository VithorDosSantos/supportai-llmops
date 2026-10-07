import pandas as pd
import pandera.pandas as pa
import pytest

from supportai.data.clean import build_text, clean_tickets, map_priority, ticket_id
from supportai.data.config import FilterConfig, load_data_config

FILTER = FilterConfig(
    language="en",
    queues={"Billing and Payments": "billing_payments", "Technical Support": "technical_support"},
    priority_map={"very_low": "low", "low": "low", "medium": "medium", "critical": "high"},
    min_text_chars=10,
)


def _raw(**overrides: list[object]) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "subject": ["Charge", None, "Login", "Hallo"],
        "body": [
            "I was charged twice  for my order.",
            "  The app crashes\non start.  ",
            "Cannot log in since yesterday.",
            "Ich wurde doppelt belastet.",
        ],
        "queue": [
            "Billing and Payments",
            "Technical Support",
            "Technical Support",
            "Billing and Payments",
        ],
        "priority": ["critical", "very_low", "medium", "low"],
        "language": ["en", "en", "EN", "de"],
        "answer": ["...", "...", "...", "..."],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_build_text_joins_subject_and_normalizes_whitespace() -> None:
    text = build_text(pd.Series(["Charge", None]), pd.Series(["twice  \n here", " only body "]))
    assert text.tolist() == ["Charge. twice here", "only body"]


def test_map_priority_collapses_to_three_levels() -> None:
    mapped = map_priority(pd.Series(["Very_Low", "critical", "unknown"]), FILTER.priority_map)
    assert mapped.tolist()[:2] == ["low", "high"]
    assert pd.isna(mapped.iloc[2])


def test_ticket_id_is_deterministic() -> None:
    ids = ticket_id(pd.Series(["abc", "abc", "abd"]))
    assert ids.iloc[0] == ids.iloc[1] != ids.iloc[2]
    assert len(ids.iloc[0]) == 16


def test_clean_tickets_filters_maps_and_reports() -> None:
    clean, report = clean_tickets(_raw(), FILTER)

    assert clean["category"].tolist() == [
        "billing_payments",
        "technical_support",
        "technical_support",
    ]
    assert clean["urgency"].tolist() == ["high", "low", "medium"]
    assert clean["text"].iloc[1] == "The app crashes on start."
    assert report.steps["raw"] == 4
    assert report.steps["language"] == 3  # o ticket em alemão sai


def test_clean_tickets_drops_unmapped_queue_priority_short_and_duplicates() -> None:
    raw = _raw(
        queue=["Billing and Payments", "Human Resources", "Technical Support", "Technical Support"],
        priority=["critical", "low", "bogus", "low"],
        language=["en", "en", "en", "en"],
        body=["I was charged twice for my order."] * 2
        + ["short", "I WAS CHARGED TWICE FOR MY ORDER."],
        subject=["Charge", "Charge", None, "CHARGE"],
    )
    clean, report = clean_tickets(raw, FILTER)
    assert len(clean) == 1
    assert report.steps == {
        "raw": 4,
        "language": 4,
        "queue": 3,
        "priority": 2,
        "min_text_chars": 2,
        "exact_duplicates": 1,
    }


def test_raw_schema_rejects_missing_column() -> None:
    with pytest.raises(pa.errors.SchemaErrors, match="queue"):
        clean_tickets(_raw().drop(columns="queue"), FILTER)


def test_project_data_config_is_valid() -> None:
    cfg = load_data_config()
    assert len(cfg.filter.categories) == 6
    assert set(cfg.filter.priority_map.values()) == {"low", "medium", "high"}
