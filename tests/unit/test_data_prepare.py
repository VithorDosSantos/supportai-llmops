import json
from pathlib import Path

import pandas as pd

from supportai.data.config import (
    DataConfig,
    FilterConfig,
    PathsConfig,
    SourceConfig,
    SplitConfig,
)
from supportai.data.prepare import prepare, write_outputs

_TOPICS = {
    "Billing and Payments": ["invoice", "refund of charge", "credit card", "payment plan"],
    "Technical Support": ["router", "laptop driver", "VPN client", "database backup"],
}
_WORDS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
]


def _raw(n_per_queue: int = 60) -> pd.DataFrame:
    rows = []
    for queue, topics in _TOPICS.items():
        for i in range(n_per_queue):
            topic = topics[i % len(topics)]
            # Palavras distintas por linha para que os textos não sejam quase-duplicatas.
            noise = " ".join(_WORDS[(i * k) % len(_WORDS)] + str(i * k) for k in range(1, 6))
            rows.append(
                {
                    "subject": f"Issue {i}",
                    "body": f"Problem with my {topic}. Reference {noise}.",
                    "queue": queue,
                    "priority": ["low", "medium", "high"][i % 3],
                    "language": "en",
                }
            )
    return pd.DataFrame(rows)


CFG = DataConfig(
    source=SourceConfig(
        repo_id="x/y", revision="r", filename="f.csv", sha256="0" * 64, license="l"
    ),
    paths=PathsConfig(raw=Path("raw.csv"), processed_dir=Path("processed")),
    filter=FilterConfig(
        language="en",
        queues={
            "Billing and Payments": "billing_payments",
            "Technical Support": "technical_support",
        },
        priority_map={"low": "low", "medium": "medium", "high": "high"},
        min_text_chars=10,
    ),
    split=SplitConfig(near_duplicate_threshold=0.9, test_size=0.15, val_size=0.15, seed=42),
)


def test_prepare_produces_all_splits_and_report(tmp_path: Path) -> None:
    df, report = prepare(_raw(), CFG)

    assert set(df["split"]) == {"train", "val", "test"}
    assert sum(report["rows_per_split"].values()) == len(df) == 120
    assert report["rows_after_step"]["raw"] == 120

    write_outputs(df, report, tmp_path / "processed", tmp_path / "report.json")
    train = pd.read_parquet(tmp_path / "processed" / "train.parquet")
    assert "split" not in train.columns
    assert {"text", "category", "urgency", "group_id"} <= set(train.columns)
    assert json.loads((tmp_path / "report.json").read_text())["rows_per_split"]["train"] == len(
        train
    )
