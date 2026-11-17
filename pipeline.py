"""Leakage-free model pipelines.

Design rule: the TF-IDF vectorizer and the classifier are always composed inside
a single scikit-learn ``Pipeline``. ``Pipeline.fit`` fits the vectorizer on the
training texts only, and ``Pipeline.predict`` / ``predict_proba`` reuse that
fitted vocabulary and those IDF weights. This is the structural fix for the
original defect, where ``TfidfVectorizer.fit_transform`` was called on the whole
dataset before ``train_test_split``.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import get_stopwords

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "dummy_majority": "Majority-class baseline",
}

DEFAULT_VECTORIZER = {
    "stopwords": True,
    "ngram_range": (1, 1),
    "max_features": None,
    "min_df": 1,
    "max_df": 1.0,
    "sublinear_tf": False,
}


def make_vectorizer(**overrides: Any) -> TfidfVectorizer:
    """Build a TF-IDF vectorizer from a fully explicit configuration."""
    cfg = {**DEFAULT_VECTORIZER, **overrides}
    return TfidfVectorizer(
        stop_words=get_stopwords() if cfg["stopwords"] else None,
        ngram_range=tuple(cfg["ngram_range"]),
        max_features=cfg["max_features"],
        min_df=cfg["min_df"],
        max_df=cfg["max_df"],
        sublinear_tf=cfg["sublinear_tf"],
        lowercase=True,
        strip_accents=None,
        dtype=np.float64,
    )


def make_classifier(name: str, seed: int = config.SEED, quick: bool = False) -> Any:
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=config.RF_ESTIMATORS_QUICK if quick else config.RF_ESTIMATORS,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "logistic_regression":
        return LogisticRegression(max_iter=config.LR_MAX_ITER, random_state=seed)
    if name == "dummy_majority":
        return DummyClassifier(strategy="most_frequent", random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(
    model_name: str,
    vectorizer_cfg: dict[str, Any] | None = None,
    seed: int = config.SEED,
    quick: bool = False,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", make_vectorizer(**(vectorizer_cfg or {}))),
            ("clf", make_classifier(model_name, seed=seed, quick=quick)),
        ]
    )


def split_dataset(
    texts: Iterable[object],
    labels: Iterable[int],
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Stratified, seeded split of the *raw text* (before any vectorisation)."""
    text_array = np.asarray(list(texts), dtype=object)
    label_series = pd.Series(list(labels))
    return train_test_split(
        text_array,
        label_series,
        test_size=test_size,
        random_state=seed,
        stratify=label_series,
    )


def positive_class_probability(pipe: Pipeline, texts: Iterable[object]) -> np.ndarray:
    """Probability of the phishing class, looked up by label rather than column.

    ``predict_proba`` columns follow ``clf.classes_``. Hard-coding column 1 would
    be wrong whenever the estimator orders its classes differently, so the index
    is resolved from ``classes_`` and checked.
    """
    classifier = pipe.named_steps["clf"]
    classes = list(classifier.classes_)
    if config.POSITIVE_LABEL not in classes:
        raise ValueError(
            f"Positive label {config.POSITIVE_LABEL} not among estimator classes {classes}."
        )
    index = classes.index(config.POSITIVE_LABEL)
    return pipe.predict_proba(texts)[:, index]


def feature_matrix_dim(pipe: Pipeline) -> int:
    return int(pipe.named_steps["tfidf"].transform([""]).shape[1])


def top_features(pipe: Pipeline, k: int = config.TOP_FEATURES) -> list[tuple[str, float]]:
    """Top-k TF-IDF terms by Random Forest impurity-based importance."""
    classifier = pipe.named_steps["clf"]
    if not hasattr(classifier, "feature_importances_"):
        raise ValueError(
            f"{type(classifier).__name__} exposes no feature_importances_; "
            "use the Random Forest pipeline for this analysis."
        )
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    importances = np.asarray(classifier.feature_importances_, dtype=float)
    if len(names) != len(importances):
        raise AssertionError(
            f"Dimension mismatch: {len(names)} feature names vs "
            f"{len(importances)} importances."
        )
    order = np.argsort(importances)[::-1][:k]
    return [(str(names[i]), float(importances[i])) for i in order]


def legacy_leaky_reproduction(
    texts: Iterable[object],
    labels: Iterable[int],
    model_name: str = "random_forest",
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Reproduce the original protocol exactly, for the audit comparison only.

    Original sequence (code1.py in the upstream repository):
        vectorizer.fit_transform(all bodies) -> train_test_split -> fit -> accuracy

    This deliberately reintroduces the leakage so the paper can quantify it.
    Its results must never be reported as the system's performance.
    """
    from sklearn.metrics import accuracy_score

    text_array = np.asarray(list(texts), dtype=object)
    label_series = pd.Series(list(labels))

    leaky_vectorizer = TfidfVectorizer(stop_words="english")
    matrix = leaky_vectorizer.fit_transform(text_array)  # <- leak: sees test rows
    x_train, x_test, y_train, y_test = train_test_split(  # unstratified, unseeded
        matrix, label_series, test_size=config.TEST_SIZE
    )
    model = make_classifier(model_name, seed=seed)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": model_name,
        "protocol": "legacy (TF-IDF fitted before split, unstratified, unseeded)",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "test_class_distribution": y_test.value_counts().sort_index().to_dict(),
        "vectorizer_vocabulary": int(matrix.shape[1]),
    }
