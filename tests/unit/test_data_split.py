import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from supportai.data.schemas import split_ticket_schema
from supportai.data.split import grouped_stratified_split, near_duplicate_groups


def test_paraphrases_share_a_group_and_distinct_texts_do_not() -> None:
    texts = pd.Series(
        [
            "Inquiry about billing options for the SaaS project management platform",
            "Inquiry regarding billing options for the SaaS project management platform",
            "My laptop screen flickers after the latest driver update",
        ]
    )
    groups = near_duplicate_groups(texts, threshold=0.8)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_threshold_one_only_groups_identical_texts() -> None:
    texts = pd.Series(["same text here", "same text here", "same text there"])
    groups = near_duplicate_groups(texts, threshold=1.0)
    assert groups[0] == groups[1] != groups[2]


def _synthetic(n_groups: int = 300, seed: int = 0) -> tuple[pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 4, size=n_groups)
    groups = np.repeat(np.arange(n_groups), sizes)
    labels_per_group = rng.choice(["a|low", "b|high", "c|medium"], size=n_groups, p=[0.6, 0.3, 0.1])
    return pd.Series(labels_per_group[groups]), groups


def test_split_keeps_groups_together_and_preserves_strata() -> None:
    strata, groups = _synthetic()
    split = grouped_stratified_split(strata, groups, test_size=0.15, val_size=0.15, seed=42)

    per_group = pd.DataFrame({"g": groups, "s": split}).groupby("g")["s"].nunique()
    assert per_group.max() == 1

    fractions = split.value_counts(normalize=True)
    assert fractions["test"] == pytest.approx(1 / 7, abs=0.04)
    assert fractions["val"] == pytest.approx(1 / 7, abs=0.04)
    overall = strata.value_counts(normalize=True)
    for name in ("train", "val", "test"):
        dist = strata[split == name].value_counts(normalize=True)
        assert dist["a|low"] == pytest.approx(overall["a|low"], abs=0.08)


def test_split_is_reproducible_with_seed() -> None:
    strata, groups = _synthetic()
    first = grouped_stratified_split(strata, groups, 0.15, 0.15, seed=7)
    second = grouped_stratified_split(strata, groups, 0.15, 0.15, seed=7)
    pd.testing.assert_series_equal(first, second)


def test_split_schema_rejects_group_leakage() -> None:
    df = pd.DataFrame(
        {
            "ticket_id": ["0123456789abcdef", "fedcba9876543210"],
            "subject": ["", ""],
            "body": ["first ticket body", "second ticket body"],
            "text": ["first ticket body", "second ticket body"],
            "category": ["a", "a"],
            "urgency": ["low", "high"],
            "group_id": [0, 0],
            "split": ["train", "test"],
        }
    )
    schema = split_ticket_schema(categories=["a"], min_text_chars=5)
    with pytest.raises(pa.errors.SchemaErrors, match="grupo em mais de um split"):
        schema.validate(df, lazy=True)
    schema.validate(df.assign(split="train"), lazy=True)
