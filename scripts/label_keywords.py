"""Sinal de ruído de rótulo: fração de tickets de cada categoria que menciona
palavras óbvias do tema (ex.: "refund" em returns_exchanges).

Uso: uv run python scripts/label_keywords.py
"""

import pandas as pd

from supportai.classifier.train import SPLIT_NAMES, load_splits
from supportai.data.config import load_data_config

KEYWORDS = {
    "return/exchange/refund": r"\b(?:return|exchange|refund)",
    "billing/invoice/charge/payment": r"\b(?:billing|invoice|charge|payment)",
}


def main() -> None:
    splits = load_splits(load_data_config().paths.resolve().processed_dir)
    df = pd.concat([splits[s] for s in SPLIT_NAMES], ignore_index=True)
    text = df["text"].str.lower()
    table = pd.DataFrame(
        {
            name: text.str.contains(pattern).groupby(df["category"]).mean()
            for name, pattern in KEYWORDS.items()
        }
    )
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
