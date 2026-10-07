"""Etapa `prepare`: CSV bruto -> splits validados em Parquet + relatório.

Uso: `python -m supportai.data.prepare`
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from supportai.config import PROJECT_ROOT
from supportai.data.clean import clean_tickets
from supportai.data.config import DataConfig, load_data_config
from supportai.data.schemas import SPLITS, split_ticket_schema
from supportai.data.split import grouped_stratified_split, near_duplicate_groups

logger = logging.getLogger(__name__)

DEFAULT_REPORT = PROJECT_ROOT / "reports" / "data.json"


def prepare(raw: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean, clean_report = clean_tickets(raw, cfg.filter)

    groups = near_duplicate_groups(clean["text"], cfg.split.near_duplicate_threshold)
    # Estratifica pela combinação categoria x urgência para manter as duas
    # distribuições estáveis entre os splits.
    strata = clean["category"] + "|" + clean["urgency"]
    split = grouped_stratified_split(
        strata, groups, cfg.split.test_size, cfg.split.val_size, cfg.split.seed
    )
    df = clean.assign(group_id=groups, split=split)
    schema = split_ticket_schema(cfg.filter.categories, cfg.filter.min_text_chars)
    df = schema.validate(df, lazy=True)
    return df, build_report(df, clean_report.steps)


def build_report(df: pd.DataFrame, steps: dict[str, int]) -> dict[str, Any]:
    group_sizes = df.groupby("group_id").size()
    return {
        "rows_after_step": steps,
        "rows_per_split": {s: int((df["split"] == s).sum()) for s in SPLITS},
        "fraction_per_split": {s: round(float((df["split"] == s).mean()), 4) for s in SPLITS},
        "near_duplicate_groups": {
            "n_groups": int(group_sizes.size),
            "rows_in_multi_ticket_groups": int(group_sizes[group_sizes > 1].sum()),
            "max_group_size": int(group_sizes.max()),
        },
        "category_distribution": _distribution(df, "category"),
        "urgency_distribution": _distribution(df, "urgency"),
    }


def _distribution(df: pd.DataFrame, column: str) -> dict[str, dict[str, float]]:
    table = pd.crosstab(df[column], df["split"], normalize="columns").round(4)
    return {
        split: {str(label): float(v) for label, v in table[split].items()}
        for split in SPLITS
        if split in table
    }


def write_outputs(
    df: pd.DataFrame, report: dict[str, Any], out_dir: Path, report_path: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        part = df[df["split"] == name].drop(columns="split").reset_index(drop=True)
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    report_path.write_text(report_json, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_data_config()
    paths = cfg.paths.resolve()
    raw = pd.read_csv(paths.raw)
    df, report = prepare(raw, cfg)
    write_outputs(df, report, paths.processed_dir, DEFAULT_REPORT)
    logger.info("splits: %s", report["rows_per_split"])


if __name__ == "__main__":
    main()
