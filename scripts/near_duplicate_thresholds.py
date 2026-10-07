"""Sensibilidade dos grupos de quase-duplicatas ao limiar de similaridade.

Gera a tabela do ADR 0003: no subconjunto em inglês do CSV bruto (todas as
filas), quantos grupos, o maior grupo, quantos tickets estão em grupos com
mais de um ticket e quantos grupos têm rótulos conflitantes.

Uso: uv run python scripts/near_duplicate_thresholds.py
"""

import numpy as np
import pandas as pd

from supportai.data.clean import build_text
from supportai.data.config import load_data_config
from supportai.data.split import near_duplicate_groups


def main() -> None:
    cfg = load_data_config()
    raw = pd.read_csv(cfg.paths.resolve().raw)
    en = raw[raw["language"] == cfg.filter.language].reset_index(drop=True)
    text = build_text(en["subject"], en["body"])

    print("limiar | grupos | maior | em grupos >1 | grupos >1 com fila conflitante")
    for threshold in (0.9, 0.85, 0.8, 0.75):
        groups = near_duplicate_groups(text, threshold)
        sizes = np.bincount(groups)
        df = pd.DataFrame({"g": groups, "queue": en["queue"]})
        multi = df[sizes[groups] > 1]
        conflict = multi.groupby("g")["queue"].nunique().gt(1).mean()
        print(
            f"{threshold:.2f} | {sizes.size} | {sizes.max()} | {(sizes[groups] > 1).sum()} "
            f"| {conflict:.1%}"
        )


if __name__ == "__main__":
    main()
