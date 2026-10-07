"""Quanto um split aleatório infla as métricas neste dataset?

Compara o F1 macro de teste do baseline TF-IDF (mesmos hiperparâmetros do
campeão) em dois splits do mesmo tamanho:
- agrupado: o split oficial (grupos de quase-duplicatas não cruzam splits);
- aleatório: StratifiedShuffleSplit ignorando os grupos.

Uso: uv run python scripts/split_leakage.py
"""

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from supportai.classifier.config import load_train_config
from supportai.classifier.metrics import evaluate
from supportai.classifier.models import build_tfidf_pipeline
from supportai.classifier.train import REPORTS_DIR, load_splits
from supportai.data.config import load_data_config


def main() -> None:
    cfg = load_train_config()
    best_c = json.loads((REPORTS_DIR / "train_tfidf.json").read_text())
    splits = load_splits(load_data_config().paths.resolve().processed_dir)
    train_val = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    full = pd.concat([train_val, splits["test"]], ignore_index=True)

    results = {}
    for target in cfg.targets:
        c = best_c[f"tfidf/{target}"]["best_C"]
        labels = sorted(full[target].unique())
        rand_train, rand_test = train_test_split(
            full,
            test_size=len(splits["test"]),
            stratify=full["category"] + "|" + full["urgency"],
            random_state=cfg.seed,
        )
        row = {}
        for name, (tr, te) in {
            "grouped": (train_val, splits["test"]),
            "random": (rand_train, rand_test),
        }.items():
            model = build_tfidf_pipeline(cfg.models.tfidf, c, cfg.seed).fit(
                tr[["text"]], tr[target]
            )
            row[name] = round(evaluate(te[target], model.predict(te[["text"]]), labels).f1_macro, 4)
        row["inflation"] = round(row["random"] - row["grouped"], 4)
        results[target] = row
        print(target, row, flush=True)

    out = REPORTS_DIR / "split_leakage.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
