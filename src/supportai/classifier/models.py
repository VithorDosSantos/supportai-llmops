"""Pipelines scikit-learn da triagem: texto bruto -> rótulo.

O artefato registrado é sempre um `Pipeline` que recebe texto cru. Assim a API
(Fase 2) não precisa saber se o modelo é TF-IDF ou embeddings, e não há risco
de o pré-processamento do serviço divergir do usado no treino (training/serving skew).
"""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from supportai.classifier.config import EncoderConfig, LogRegConfig, TfidfConfig

TEXT_COLUMN = "text"

# O MLflow serializa sklearn com skops, que (ao contrário do pickle) recusa
# carregar código arbitrário: tipos fora do sklearn/numpy precisam ser
# declarados como confiáveis, explicitamente, ao salvar o modelo.
SKOPS_TRUSTED_TYPES = [
    "supportai.classifier.models.to_text_list",
    "supportai.classifier.models.SentenceEncoder",
]


def to_text_list(x: Any) -> list[str]:
    """Normaliza a entrada para lista de strings.

    O pyfunc do MLflow (e a API) entregam um DataFrame com a coluna `text`; o
    TfidfVectorizer, se recebesse o DataFrame, iteraria sobre os NOMES das
    colunas, um bug silencioso. Aceitamos DataFrame, Series, array ou lista.
    """
    if isinstance(x, pd.DataFrame):
        x = x[TEXT_COLUMN] if TEXT_COLUMN in x.columns else x.iloc[:, 0]
    return [str(t) for t in x]


def _text_input() -> FunctionTransformer:
    return FunctionTransformer(to_text_list, validate=False)


def build_logreg(cfg: LogRegConfig, c: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c, class_weight=cfg.class_weight, max_iter=cfg.max_iter, random_state=seed
    )


def build_tfidf_pipeline(cfg: TfidfConfig, c: float, seed: int) -> Pipeline:
    """Baseline: TF-IDF de palavras + caracteres concatenados, regressão logística.

    Forte e barato: treina em segundos, roda em CPU em ~ms por ticket e é
    interpretável (pesos por n-grama). É o patamar que o modelo 2 precisa superar.
    """
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=cfg.word_ngram_range,
                    min_df=cfg.min_df,
                    max_features=cfg.max_features_word,
                    sublinear_tf=cfg.sublinear_tf,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=cfg.char_ngram_range,
                    min_df=cfg.min_df,
                    max_features=cfg.max_features_char,
                    sublinear_tf=cfg.sublinear_tf,
                    strip_accents="unicode",
                ),
            ),
        ]
    )
    return Pipeline(
        [("input", _text_input()), ("features", features), ("clf", build_logreg(cfg, c, seed))]
    )


def compact_tfidf_vocabulary(pipeline: Pipeline) -> Pipeline:
    """Converte os valores de `vocabulary_` de np.int64 para int (após o fit).

    O skops serializa cada escalar numpy como um arquivo .npy separado dentro do
    zip: com ~68 mil termos, salvar o modelo levava ~190 s. Com `int` nativo
    leva <1 s e o carregamento cai de ~7,7 s para ~1,7 s, com predições idênticas.
    """
    for _, vectorizer in pipeline.named_steps["features"].transformer_list:
        vectorizer.vocabulary_ = {term: int(idx) for term, idx in vectorizer.vocabulary_.items()}
    return pipeline


class TextEncoder(Protocol):
    """Interface mínima do SentenceTransformer usada aqui (permite um falso nos testes)."""

    def encode(
        self,
        sentences: list[str],
        batch_size: int = ...,
        show_progress_bar: bool | None = ...,
        normalize_embeddings: bool = ...,
    ) -> Any: ...


def load_sentence_transformer(name: str) -> TextEncoder:
    """Import tardio: sentence-transformers/torch são um extra opcional (`--extra embeddings`)."""
    from sentence_transformers import SentenceTransformer

    encoder: TextEncoder = SentenceTransformer(name, device="cpu")
    return encoder


class SentenceEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Transformer sklearn que converte texto em embeddings normalizados.

    Não tem parâmetros treináveis (`fit` é no-op): o encoder é pré-treinado e
    congelado; só a regressão logística em cima dele aprende. O modelo neural
    não entra no pickle (só o nome): ele é recarregado do cache do Hugging Face
    na primeira chamada, o que mantém o artefato do MLflow pequeno.
    """

    def __init__(self, model_name: str, prefix: str = "", batch_size: int = 64) -> None:
        self.model_name = model_name
        self.prefix = prefix
        self.batch_size = batch_size

    def fit(self, x: Sequence[str], y: object = None) -> "SentenceEncoder":
        return self

    def transform(self, x: Sequence[str]) -> npt.NDArray[np.float32]:
        encoder = self.__dict__.get("_encoder")
        if encoder is None:
            encoder = self.__dict__["_encoder"] = load_sentence_transformer(self.model_name)
        texts = [self.prefix + str(t) for t in x]
        vectors = encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_encoder", None)
        return state


def build_embedding_pipeline(
    encoder: EncoderConfig, batch_size: int, clf: LogisticRegression
) -> Pipeline:
    """Monta o pipeline final a partir de uma regressão já treinada em embeddings em cache.

    Treinar a logística sobre embeddings pré-computados evita re-encodar os
    ~10 mil tickets a cada valor de C do grid; como o encoder não aprende nada,
    o pipeline montado depois é equivalente a ter treinado ponta a ponta.
    """
    return Pipeline(
        [
            ("input", _text_input()),
            ("encoder", SentenceEncoder(encoder.name, encoder.prefix, batch_size)),
            ("clf", clf),
        ]
    )
