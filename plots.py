"""Publication-quality figures. Every figure is generated from measured values."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from . import config  # noqa: E402

COLORS = {
    "random_forest": "#1f4e79",
    "logistic_regression": "#c0504d",
    "dummy_majority": "#7f7f7f",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "figure.dpi": config.FIG_DPI,
        "savefig.bbox": "tight",
    }
)


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], path: Path) -> None:
    """One ROC panel containing every model, with AUC in the legend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(
            fpr,
            tpr,
            linewidth=2.0,
            color=COLORS.get(name, None),
            label=f"{_pretty(name)} (AUC = {auc:.4f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="black", label="Chance (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on the Held-out Test Set")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.005)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(path, dpi=config.DPI)
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    path: Path,
    color: str = "Blues",
) -> np.ndarray:
    """Annotated confusion matrix; cell counts plus row-normalised percentages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1])
    row_totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(shares, cmap=color, vmin=0.0, vmax=1.0)
    labels = ["Legitimate", "Phishing"]
    ticks = np.arange(2)
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for i in ticks:
        for j in ticks:
            share = shares[i, j]
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}\n({100 * share:.1f}%)",
                ha="center",
                va="center",
                color="white" if share > 0.55 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised rate")
    fig.savefig(path, dpi=config.DPI)
    plt.close(fig)
    return matrix


def plot_feature_importance(
    features: list[tuple[str, float]], path: Path, title: str | None = None
) -> None:
    """Horizontal bars, most important at the top."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in features][::-1]
    values = [value for _, value in features][::-1]
    height = max(3.2, 0.28 * len(features))
    fig, ax = plt.subplots(figsize=(6.4, height))
    ax.barh(range(len(names)), values, color=COLORS["random_forest"], height=0.7)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Mean decrease in impurity (Random Forest feature importance)")
    ax.set_title(title or f"Top {len(features)} TF-IDF Features - Random Forest")
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    fig.savefig(path, dpi=config.DPI)
    plt.close(fig)


def plot_class_distribution(distribution: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["class"] for row in distribution]
    counts = [row["count"] for row in distribution]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(names, counts, color=[COLORS["dummy_majority"], "#2e75b6"][: len(names)])
    for bar, count, row in zip(bars, counts, distribution):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({row['percent']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Number of emails")
    ax.set_title("CEAS-08 class distribution")
    ax.set_ylim(0, max(counts) * 1.18 if counts else 1)
    fig.savefig(path, dpi=config.DPI)
    plt.close(fig)


def _pretty(name: str) -> str:
    return {
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "dummy_majority": "Majority-class baseline",
    }.get(name, name)
