"""Text cleaning.

Cleaning here is *stateless and per-document*: it depends only on the text it is
given, never on the corpus. It therefore cannot leak information from the test
split, which is why it may be applied before splitting. Every transform that
does learn from data (TF-IDF vocabulary and IDF weights, classifier parameters)
lives inside the scikit-learn Pipeline and is fitted on training folds only.
"""

from __future__ import annotations

import html
import re
from typing import Iterable

import numpy as np
import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+|\b\S+@\S+\.\S+\b", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z]+")
_WS_RE = re.compile(r"\s+")

_STOPWORDS_CACHE: list[str] | None = None


def clean_text(text: object) -> str:
    """Normalise one email body into a TF-IDF friendly string."""
    raw = "" if text is None else str(text)
    if not raw:
        return ""
    raw = html.unescape(raw)
    raw = _TAG_RE.sub(" ", raw)
    raw = _URL_RE.sub(" urltoken ", raw)
    raw = raw.lower()
    raw = _NON_ALPHA.sub(" ", raw)
    return _WS_RE.sub(" ", raw).strip()


def clean_series(values: Iterable[object]) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    return series.astype(str).map(clean_text)


def to_text_array(values: Iterable[object]) -> np.ndarray:
    """Minimal conversion used by the 'no preprocessing' ablation row."""
    series = values if isinstance(values, pd.Series) else pd.Series(list(values))
    return series.fillna("").astype(str).to_numpy(dtype=object)


def get_stopwords() -> list[str]:
    """NLTK's English stop list when available, else scikit-learn's."""
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None:
        return _STOPWORDS_CACHE
    try:  # pragma: no cover - depends on the environment
        from nltk.corpus import stopwords as nltk_stopwords

        try:
            words = nltk_stopwords.words("english")
        except LookupError:
            import nltk

            nltk.download("stopwords", quiet=True)
            words = nltk_stopwords.words("english")
        _STOPWORDS_CACHE = sorted(words)
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        _STOPWORDS_CACHE = sorted(ENGLISH_STOP_WORDS)
    return _STOPWORDS_CACHE


def stopword_source() -> str:
    try:  # pragma: no cover
        import nltk  # noqa: F401
        from nltk.corpus import stopwords  # noqa: F401

        return "NLTK English stopwords"
    except Exception:
        return "scikit-learn ENGLISH_STOP_WORDS (NLTK unavailable)"


def count_urls(text: object) -> int:
    raw = "" if text is None else str(text)
    return len(_URL_RE.findall(raw))


def mask_text(text: object, limit: int = 160) -> str:
    """Redact addresses/URLs and truncate, so excerpts can be logged safely."""
    raw = "" if text is None else str(text)
    raw = _URL_RE.sub("[REDACTED]", raw)
    raw = _TAG_RE.sub(" ", raw)
    raw = _WS_RE.sub(" ", raw).strip()
    return raw[:limit] + ("..." if len(raw) > limit else "")
