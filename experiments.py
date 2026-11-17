"""Experiment implementations: holdout evaluation, CV, ablation, errors, cost."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config, evaluate, pipeline
from .preprocessing import count_urls, mask_text

# Ablation grid. Each entry changes exactly one factor relative to A1 so the
# effect of that factor can be attributed.
ABLATION_CONFIGS: list[dict[str, Any]] = [
    {"name": "A1 default: unigrams, stop words, min_df=1", "vectorizer": {}},
    {"name": "A2 no stop-word removal", "vectorizer": {"stopwords": False}},
    {"name": "A3 bigrams (1,2), min_df=2", "vectorizer": {"ngram_range": (1, 2), "min_df": 2}},
    {"name": "A4 min_df=2", "vectorizer": {"min_df": 2}},
    {"name": "A5 max_features=5,000", "vectorizer": {"max_features": 5000}},
    {"name": "A6 sublinear_tf=True", "vectorizer": {"sublinear_tf": True}},
]

RAW_TEXT_VARIANT = "A0 no text preprocessing (raw body)"


def _scoring() -> dict[str, Any]:
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }


# --------------------------------------------------------------------------- #
# Holdout evaluation
# --------------------------------------------------------------------------- #
def run_holdout(
    train_texts: Iterable[object],
    test_texts: Iterable[object],
    y_train: Iterable[int],
    y_test: Iterable[int],
    model_name: str,
    vectorizer_cfg: dict[str, Any] | None = None,
    seed: int = config.SEED,
    quick: bool = False,
) -> dict[str, Any]:
    """Fit on training texts only, then evaluate once on the untouched test set."""
    pipe = pipeline.build_pipeline(model_name, vectorizer_cfg, seed=seed, quick=quick)
    started = time.perf_counter()
    pipe.fit(train_texts, y_train)
    fit_seconds = time.perf_counter() - started

    y_pred = pipe.predict(test_texts)
    y_score = None
    if model_name != "dummy_majority":
        y_score = pipeline.positive_class_probability(pipe, test_texts)

    metrics = evaluate.binary_metrics(
        y_test, y_pred, y_score, model_name=pipeline.MODEL_LABELS.get(model_name, model_name)
    )
    return {
        "model_key": model_name,
        "pipeline": pipe,
        "metrics": metrics,
        "y_pred": np.asarray(y_pred),
        "y_score": None if y_score is None else np.asarray(y_score),
        "fit_seconds": fit_seconds,
        "n_features": pipeline.feature_matrix_dim(pipe),
        "vectorizer_config": dict(vectorizer_cfg or pipeline.DEFAULT_VECTORIZER),
    }


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def run_cross_validation(
    texts: Iterable[object],
    labels: Iterable[int],
    model_names: list[str],
    vectorizer_cfg: dict[str, Any] | None = None,
    folds: int = config.CV_FOLDS,
    seed: int = config.SEED,
    quick: bool = False,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified k-fold CV with TF-IDF re-fitted inside every training fold.

    This is a separate experiment from the holdout test: its numbers describe
    variance across folds, not the performance of the saved model.
    """
    text_array = np.asarray(list(texts), dtype=object)
    label_array = np.asarray(list(labels)).astype(int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    summary_rows, fold_rows = [], []
    for model_name in model_names:
        pipe = pipeline.build_pipeline(model_name, vectorizer_cfg, seed=seed, quick=quick)
        scores = cross_validate(
            pipe,
            text_array,
            label_array,
            cv=splitter,
            scoring=_scoring(),
            n_jobs=n_jobs,
            return_train_score=False,
        )
        label = pipeline.MODEL_LABELS.get(model_name, model_name)
        row = {"model": label, "folds": folds}
        for metric in evaluate.CV_SCORING_NAMES:
            values = np.asarray(scores[f"test_{metric}"], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            for fold_index, value in enumerate(values, start=1):
                fold_rows.append(
                    {"model": label, "fold": fold_index, "metric": metric, "value": float(value)}
                )
        row["fit_seconds_mean"] = float(np.mean(scores["fit_time"]))
        summary_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #
def run_ablation(
    train_variants: dict[str, Iterable[object]],
    test_variants: dict[str, Iterable[object]],
    y_train: Iterable[int],
    y_test: Iterable[int],
    model_name: str = "random_forest",
    seed: int = config.SEED,
    quick: bool = False,
    include_raw_text: bool = True,
) -> pd.DataFrame:
    """One-factor-at-a-time comparison on the same split and seed."""
    rows: list[dict[str, Any]] = []
    plan = [{"name": RAW_TEXT_VARIANT, "vectorizer": {}}] if include_raw_text else []
    plan += ABLATION_CONFIGS

    for entry in plan:
        key = "raw" if entry["name"] == RAW_TEXT_VARIANT else "clean"
        result = run_holdout(
            train_variants[key],
            test_variants[key],
            y_train,
            y_test,
            model_name,
            vectorizer_cfg=entry["vectorizer"],
            seed=seed,
            quick=quick,
        )
        metrics = result["metrics"]
        rows.append(
            {
                "variant": entry["name"],
                "model": metrics["model"],
                "features": result["n_features"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "false_positive": metrics["false_positive"],
                "false_negative": metrics["false_negative"],
                "fit_seconds": round(result["fit_seconds"], 2),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def error_analysis(
    test_texts: Iterable[object],
    y_test: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None = None,
    top_k: int = 15,
    max_excerpts: int = 10,
) -> dict[str, Any]:
    """Characterise the actual mistakes. No explanation is invented for them."""
    texts = np.asarray(list(test_texts), dtype=object)
    truth = np.asarray(list(y_test)).astype(int)
    predicted = np.asarray(list(y_pred)).astype(int)
    scores = None if y_score is None else np.asarray(list(y_score), dtype=float)

    pos = config.POSITIVE_LABEL
    fp_idx = np.where((truth != pos) & (predicted == pos))[0]
    fn_idx = np.where((truth == pos) & (predicted != pos))[0]
    tp_idx = np.where((truth == pos) & (predicted == pos))[0]

    def _profile(indices: np.ndarray) -> dict[str, Any]:
        selected = [str(texts[i]) for i in indices]
        lengths = [len(text) for text in selected]
        words = [len(text.split()) for text in selected]
        urls = [count_urls(text) for text in selected]
        return {
            "count": int(len(indices)),
            "mean_chars": round(float(np.mean(lengths)), 1) if lengths else 0.0,
            "median_chars": round(float(np.median(lengths)), 1) if lengths else 0.0,
            "mean_words": round(float(np.mean(words)), 1) if words else 0.0,
            "share_with_url": round(float(np.mean([u > 0 for u in urls])), 4) if urls else 0.0,
            "share_shorter_than_200_chars": (
                round(float(np.mean([length < 200 for length in lengths])), 4) if lengths else 0.0
            ),
        }

    def _top_tokens(indices: np.ndarray, k: int = top_k) -> list[tuple[str, int]]:
        selected = [str(texts[i]) for i in indices]
        if len(selected) < 2:
            return []
        try:
            vectorizer = CountVectorizer(stop_words="english", min_df=1, max_features=20000)
            matrix = vectorizer.fit_transform(selected)
        except ValueError:
            return []
        totals = np.asarray(matrix.sum(axis=0)).ravel()
        names = vectorizer.get_feature_names_out()
        order = np.argsort(totals)[::-1][:k]
        return [(str(names[i]), int(totals[i])) for i in order]

    baseline_lengths = [len(str(t)) for t in texts]
    details = {
        "false_positives": _profile(fp_idx),
        "false_negatives": _profile(fn_idx),
        "true_positives": _profile(tp_idx),
        "correct_length_mean": round(float(np.mean(baseline_lengths)), 1) if len(baseline_lengths) else 0.0,
        "false_positive_tokens": _top_tokens(fp_idx),
        "false_negative_tokens": _top_tokens(fn_idx),
        "true_positive_tokens": _top_tokens(tp_idx),
        "false_positive_excerpts": [mask_text(texts[i]) for i in fp_idx[:max_excerpts]],
        "false_negative_excerpts": [mask_text(texts[i]) for i in fn_idx[:max_excerpts]],
        "score_summary": {},
    }
    if scores is not None:
        for name, indices in (("false_positive", fp_idx), ("false_negative", fn_idx)):
            if len(indices):
                values = scores[indices]
                details["score_summary"][name] = {
                    "mean: ": None,
                    "mean": round(float(values.mean()), 4),
                    "min": round(float(values.min()), 4),
                    "max": round(float(values.max()), 4),
                }
    return details


def write_error_rows(
    test_texts: Iterable[object],
    y_test: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None,
    model_label: str,
    path: Path,
) -> int:
    """Masked per-error CSV: no raw addresses or URLs are written to disk."""
    texts = np.asarray(list(test_texts), dtype=object)
    truth = np.asarray(list(y_test)).astype(int)
    predicted = np.asarray(list(y_pred)).astype(int)
    scores = None if y_score is None else np.asarray(list(y_score), dtype=float)
    pos = config.POSITIVE_LABEL

    rows = []
    for index in np.where(truth != predicted)[0]:
        rows.append(
            {
                "model": model_label,
                "error_type": "false_positive" if predicted[index] == pos else "false_negative",
                "true_label": int(truth[index]),
                "predicted_label": int(predicted[index]),
                "phishing_probability": (
                    round(float(scores[index]), 4) if scores is not None else None
                ),
                "chars": len(str(texts[index])),
                "words": len(str(texts[index]).split()),
                "urls": count_urls(texts[index]),
                "masked_excerpt": mask_text(texts[index]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return len(rows)


# --------------------------------------------------------------------------- #
# Cost / deployment measurements
# --------------------------------------------------------------------------- #
def deployment_metrics(
    pipe: Any,
    test_texts: Iterable[object],
    n_single: int = 2000,
    model_label: str = "",
) -> dict[str, Any]:
    """Measure what deployment claims would need to rest on."""
    texts = [str(text) for text in test_texts]
    single = texts[: min(n_single, len(texts))]

    artifact_bytes = None
    with tempfile.TemporaryDirectory() as directory:
        bundle_path = Path(directory) / "pipeline.joblib"
        joblib.dump(pipe, bundle_path, compress=0)
        artifact_bytes = bundle_path.stat().st_size

    pipe.predict(single[:1])  # warm-up, excluded from timing

    latencies = []
    for text in single:
        started = time.perf_counter()
        pipe.predict([text])
        latencies.append((time.perf_counter() - started) * 1000.0)

    batch_started = time.perf_counter()
    pipe.predict(texts)
    batch_seconds = time.perf_counter() - batch_started

    return {
        "model": model_label,
        "artifact_mb": round(artifact_bytes / (1024 * 1024), 3) if artifact_bytes else None,
        "single_email_latency_ms_mean": round(float(np.mean(latencies)), 3),
        "single_email_latency_ms_median": round(float(np.median(latencies)), 3),
        "single_email_latency_ms_p95": round(float(np.percentile(latencies, 95)), 3),
        "batch_throughput_emails_per_second": round(len(texts) / batch_seconds, 1) if batch_seconds else None,
        "timed_emails": len(single),
        "n_features": pipeline.feature_matrix_dim(pipe),
    }
