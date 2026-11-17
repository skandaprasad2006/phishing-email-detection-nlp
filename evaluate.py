"""Metric computation, kept in one place so every table uses the same code."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config

METRIC_COLUMNS = [
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
    "false_positive_rate",
    "false_negative_rate",
    "specificity",
    "negative_predictive_value",
    "support_phishing",
    "support_legitimate",
]

# Metrics reported for cross-validation (all computed on each held-out fold).
CV_SCORING_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def binary_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None = None,
    model_name: str = "",
) -> dict[str, Any]:
    """Complete binary metrics with the phishing class as the positive class."""
    y_true_arr = np.asarray(list(y_true)).astype(int)
    y_pred_arr = np.asarray(list(y_pred)).astype(int)
    pos = config.POSITIVE_LABEL

    tn, fp, fn, tp = confusion_matrix(
        y_true_arr, y_pred_arr, labels=[0, 1]
    ).ravel()

    precision = precision_score(y_true_arr, y_pred_arr, pos_label=pos, zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, pos_label=pos, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, pos_label=pos, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    auc = float("nan")
    if y_score is not None:
        try:
            auc = float(roc_auc_score(y_true_arr, np.asarray(list(y_score))))
        except ValueError:
            auc = float("nan")

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": auc,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "specificity": float(specificity),
        "negative_predictive_value": float(npv),
        "support_phishing": int((y_true_arr == pos).sum()),
        "support_legitimate": int((y_true_arr != pos).sum()),
    }


def roc_points(
    y_true: Iterable[int], y_score: Iterable[float]
) -> tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(np.asarray(list(y_true)).astype(int), np.asarray(list(y_score)))
    auc = float(roc_auc_score(np.asarray(list(y_true)).astype(int), np.asarray(list(y_score))))
    return fpr, tpr, auc


def threshold_table(
    y_true: Iterable[int],
    y_score: Iterable[float],
    thresholds: Iterable[float] = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2)),
) -> pd.DataFrame:
    """Operating points: how FP/FN trade off as the decision threshold moves."""
    y_true_arr = np.asarray(list(y_true)).astype(int)
    scores = np.asarray(list(y_score))
    rows = []
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        metrics = binary_metrics(y_true_arr, pred, y_score=[], model_name=f"t={threshold:.2f}")
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    return frame[
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "false_positive",
            "false_negative",
            "false_positive_rate",
            "false_negative_rate",
        ]
    ]


def metrics_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)[METRIC_COLUMNS]


def format_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Markdown table where percentages are rendered as percentages."""
    percent_cols = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "false_positive_rate",
        "false_negative_rate",
        "specificity",
        "negative_predictive_value",
    }
    display = frame.copy()
    for column in display.columns:
        if column in percent_cols:
            display[column] = display[column].map(
                lambda v: "n/a" if pd.isna(v) else f"{100 * float(v):.2f}%"
            )
        elif pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda v: "n/a" if pd.isna(v) else f"{float(v):.{float_digits}f}"
            )

    header = "| " + " | ".join(str(c) for c in display.columns) + " |"
    divider = "|" + "|".join("---" for _ in display.columns) + "|"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in display.itertuples(index=False)
    ]
    return "\n".join([header, divider, *body])


def label_name(value: Any) -> str:
    return "Phishing" if int(value) == config.POSITIVE_LABEL else "Legitimate"
