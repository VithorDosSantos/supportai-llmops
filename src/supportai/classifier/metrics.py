"""Métricas de classificação da triagem.

F1 macro é a métrica principal: dá o mesmo peso a cada classe, então um modelo
que ignora a classe rara (returns_exchanges, ~5%) é punido, ao contrário da
acurácia, que pode ser alta só acertando as classes grandes.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")  # sem display: roda em CI e servidores

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@dataclass(frozen=True)
class EvalResult:
    labels: list[str]
    f1_macro: float
    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion: npt.NDArray[np.int64]

    def as_metrics(self, prefix: str) -> dict[str, float]:
        """Formato plano para `mlflow.log_metrics` (ex.: `test_f1_macro`, `test_f1_it_support`)."""
        metrics = {f"{prefix}_f1_macro": self.f1_macro, f"{prefix}_accuracy": self.accuracy}
        for label in self.labels:
            metrics[f"{prefix}_f1_{label}"] = self.per_class[label]["f1-score"]
        return metrics


def evaluate(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> EvalResult:
    labels = list(labels)
    report: dict[str, Any] = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return EvalResult(
        labels=labels,
        f1_macro=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class={label: {k: float(v) for k, v in report[label].items()} for label in labels},
        confusion=np.asarray(confusion_matrix(y_true, y_pred, labels=labels), dtype=np.int64),
    )


def plot_confusion_matrix(result: EvalResult, title: str) -> Figure:
    """Matriz normalizada por linha (recall por classe), com contagens absolutas no texto."""
    counts = result.confusion
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_sums, out=np.zeros(counts.shape), where=row_sums > 0)

    size = 1.2 * len(result.labels) + 2
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(result.labels)), result.labels, rotation=45, ha="right")
    ax.set_yticks(range(len(result.labels)), result.labels)
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    ax.set_title(title)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            color = "white" if normalized[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{normalized[i, j]:.2f}\n({counts[i, j]})",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    fig.tight_layout()
    return fig
