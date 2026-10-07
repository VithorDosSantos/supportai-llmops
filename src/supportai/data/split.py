"""Split treino/validação/teste estratificado e agrupado por quase-duplicatas.

Por que agrupar: o dataset é sintético e cheio de paráfrases do mesmo ticket
("inquiry about billing options" / "inquiry regarding billing options"). Um
split aleatório coloca paráfrases em treino e teste, e a métrica passa a medir
memorização. Agrupamos tickets parecidos e mantemos cada grupo inteiro num só split.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors


def near_duplicate_groups(texts: pd.Series, threshold: float) -> npt.NDArray[np.int64]:
    """Atribui o mesmo `group_id` a textos com cosseno >= `threshold`.

    TF-IDF de n-gramas de caracteres captura paráfrases leves (troca de
    palavras, pontuação) sem depender de modelo neural, e roda em segundos.
    Os grupos são as componentes conexas do grafo de vizinhança, logo a relação
    é transitiva: A~B e B~C põem A, B e C juntos. Por isso o limiar não pode ser
    baixo demais (cadeias longas viram grupos gigantes; com 0,8 o maior tem 14
    tickets neste dataset).
    """
    vectors = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True
    ).fit_transform(texts)
    graph = (
        NearestNeighbors(metric="cosine")
        .fit(vectors)
        .radius_neighbors_graph(vectors, radius=1.0 - threshold, mode="connectivity")
    )
    _, labels = connected_components(graph, directed=False)
    return np.asarray(labels, dtype=np.int64)


def _one_group_fold(
    y: pd.Series, groups: npt.NDArray[np.int64], n_splits: int, seed: int
) -> npt.NDArray[np.bool_]:
    """Máscara do primeiro fold de um StratifiedGroupKFold (~1/n_splits dos dados)."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held_out = next(cv.split(np.zeros(len(y)), y, groups))
    mask = np.zeros(len(y), dtype=bool)
    mask[held_out] = True
    return mask


def grouped_stratified_split(
    strata: pd.Series,
    groups: npt.NDArray[np.int64],
    test_size: float,
    val_size: float,
    seed: int,
) -> pd.Series:
    """Rótulo de split por linha ("train" | "val" | "test").

    StratifiedGroupKFold não aceita uma fração arbitrária, então usamos
    n_splits = round(1/fração) e pegamos um fold. Os tamanhos efetivos são
    aproximados (0,15 vira 1/7 ≈ 0,143) e ficam registrados no relatório.
    """
    split = pd.Series("train", index=strata.index, dtype=object)

    test_mask = _one_group_fold(strata, groups, round(1 / test_size), seed)
    split[test_mask] = "test"

    rest = ~test_mask
    val_fraction_of_rest = val_size / (1 - test_size)
    val_mask_rest = _one_group_fold(
        strata[rest], groups[rest], round(1 / val_fraction_of_rest), seed
    )
    split[strata.index[rest][val_mask_rest]] = "val"
    return split.astype(str)
