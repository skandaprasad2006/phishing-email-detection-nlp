"""Dataset discovery, schema verification and descriptive statistics.

Rule followed throughout: column names, label encoding and every statistic are
read from the file on disk. No value in this module is assumed from the draft
paper.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from . import config


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be located, read or interpreted."""


# --------------------------------------------------------------------------- #
# Locating and reading
# --------------------------------------------------------------------------- #
def find_dataset(path: str | Path | None) -> Path:
    """Return the CSV to load, extracting a zip archive when necessary."""
    if path is None:
        candidates = sorted(
            list(config.DATA_DIR.glob("*.csv")) + list(config.DATA_DIR.glob("*.csv.zip"))
        )
        if not candidates:
            raise DatasetError(
                "No dataset found. Pass --data <file.csv|file.csv.zip> or place the "
                f"Kaggle CEAS_08 export inside {config.DATA_DIR}."
            )
        return candidates[0]

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (config.PROJECT_ROOT / resolved).resolve()
    if resolved.is_dir():
        inner = sorted(
            list(resolved.glob("*.csv")) + list(resolved.glob("*.csv.zip"))
        )
        if not inner:
            raise DatasetError(f"No .csv or .csv.zip found inside {resolved}")
        resolved = inner[0]
    if not resolved.exists():
        raise DatasetError(f"Dataset path does not exist: {resolved}")
    return resolved


def extract_if_zip(path: Path) -> Path:
    """Extract a .csv.zip next to itself and return the contained CSV path."""
    if path.suffix != ".zip":
        return path
    target_dir = config.DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise DatasetError(f"{path.name} contains no CSV member.")
        archive.extract(members[0], target_dir)
    return target_dir / Path(members[0]).name


def load_dataframe(path: str | Path | None) -> tuple[pd.DataFrame, Path]:
    """Load the raw dataset. Only the CSV header is trusted, not its content."""
    csv_path = extract_if_zip(find_dataset(path))
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:  # pragma: no cover - surfaced to the user
        raise DatasetError(f"Failed to read {csv_path.name}: {exc}") from exc
    if df.empty:
        raise DatasetError(f"{csv_path.name} contains no rows.")
    return df, csv_path


# --------------------------------------------------------------------------- #
# Schema verification
# --------------------------------------------------------------------------- #
def resolve_column(
    df: pd.DataFrame, candidates: Sequence[str], kind: str
) -> str:
    """Find a column by exact name first, then case-insensitively."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return str(lookup[candidate.lower()])
    raise DatasetError(
        f"Could not find the {kind} column. Tried {list(candidates)}; "
        f"file has {list(df.columns)}. Fix BODY_COLUMN_CANDIDATES / "
        "LABEL_COLUMN_CANDIDATES in src/config.py."
    )


def optional_column(
    df: pd.DataFrame, candidates: Sequence[str]
) -> str | None:
    try:
        return resolve_column(df, candidates, "optional")
    except DatasetError:
        return None


def normalise_labels(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """Map labels to the integers 0/1 and record exactly what was done.

    Supports the CEAS_08 numeric encoding ({0,1}) and common string encodings.
    Raises instead of silently guessing when the mapping is ambiguous.
    """
    raw_values = series.dropna().unique().tolist()
    if pd.api.types.is_numeric_dtype(series) and set(raw_values) <= {0, 1}:
        mapping = {v: int(v) for v in raw_values}
        return series.astype("int64"), {
            "encoding": "numeric 0/1 as stored in file",
            "raw_values": raw_values,
            "mapping": {str(k): int(v) for k, v in mapping.items()},
        }

    tokens = {str(v).strip().lower() for v in raw_values}
    phishing_like = {"1", "phishing", "phish", "spam", "malicious", "true", "yes"}
    legit_like = {"0", "legitimate", "legit", "ham", "benign", "false", "no", "safe"}

    if tokens <= (phishing_like | legit_like) and tokens & phishing_like:
        mapping: dict[Any, int] = {}
        for value in raw_values:
            key = str(value).strip().lower()
            mapping[value] = 1 if key in phishing_like else 0
        mapped = series.map(mapping)
        return mapped.astype("int64"), {
            "encoding": "string labels mapped to 0/1",
            "raw_values": raw_values,
            "mapping": {str(k): int(v) for k, v in mapping.items()},
        }

    raise DatasetError(
        "Unrecognised label encoding; refusing to guess. Values found: "
        f"{raw_values}. Map them explicitly to 0 (legitimate) / 1 (phishing)."
    )


def prepare_dataset(
    df: pd.DataFrame, csv_path: Path
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Return (texts, labels, meta) after verifying the real schema."""
    body_col = resolve_column(df, config.BODY_COLUMN_CANDIDATES, "email body")
    label_col = resolve_column(df, config.LABEL_COLUMN_CANDIDATES, "label")
    subject_col = optional_column(df, config.SUBJECT_COLUMN_CANDIDATES)
    url_col = optional_column(df, config.URL_COLUMN_CANDIDATES)

    labels, label_meta = normalise_labels(df[label_col])
    texts = df[body_col]

    missing_before = int(texts.isna().sum())
    labels_missing = int(labels.isna().sum())
    if labels_missing:
        keep = labels.notna()
        texts, labels = texts[keep], labels[keep]
    texts = texts.fillna("").astype(str)

    if config.POSITIVE_LABEL not in set(labels.unique().tolist()):
        raise DatasetError(
            f"Positive label {config.POSITIVE_LABEL} is not present after "
            f"normalisation (found {sorted(labels.unique())}). Fix POSITIVE_LABEL "
            "in src/config.py."
        )

    meta = {
        "source_file": csv_path.name,
        "source_path": str(csv_path),
        "n_rows_raw": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "body_column": body_col,
        "label_column": label_col,
        "subject_column": subject_col,
        "url_column": url_col,
        "label_meta": label_meta,
        "missing_bodies": missing_before,
        "missing_labels_dropped": labels_missing,
        "duplicate_bodies": int(texts.duplicated().sum()),
        "n_used": int(len(texts)),
    }
    return texts.reset_index(drop=True), labels.reset_index(drop=True), meta


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #
def class_distribution(labels: pd.Series) -> list[dict[str, Any]]:
    counts = labels.value_counts().sort_index()
    total = int(counts.sum())
    rows = []
    for value, count in counts.items():
        name = "phishing" if int(value) == config.POSITIVE_LABEL else "legitimate"
        rows.append(
            {
                "label": int(value),
                "class": name,
                "count": int(count),
                "percent": round(100.0 * count / total, 2) if total else 0.0,
            }
        )
    return rows


def _length_stats(texts: Iterable[str]) -> dict[str, float]:
    lengths, words = [], []
    for text in texts:
        text = "" if text is None else str(text)
        lengths.append(len(text))
        words.append(len(text.split()))
    if not lengths:
        return {"mean_chars": 0.0, "median_chars": 0.0, "mean_words": 0.0}
    series_len = pd.Series(lengths)
    return {
        "mean_chars": round(float(series_len.mean()), 2),
        "median_chars": round(float(series_len.median()), 2),
        "mean_words": round(float(pd.Series(words).mean()), 2),
    }


def dataset_statistics(
    texts: pd.Series,
    labels: pd.Series,
    meta: dict[str, Any],
    vectorizer: Any | None = None,
    token_vocabulary_size: int | None = None,
) -> dict[str, Any]:
    """All numbers the dataset-statistics table needs, computed from the data."""
    phishing_mask = labels == config.POSITIVE_LABEL
    stats: dict[str, Any] = {
        "total_records": int(len(labels)),
        "class_distribution": class_distribution(labels),
        "missing_bodies": meta.get("missing_bodies", 0),
        "missing_labels_dropped": meta.get("missing_labels_dropped", 0),
        "duplicate_bodies": meta.get("duplicate_bodies", 0),
        "all_lengths": _length_stats(texts),
        "legitimate_lengths": _length_stats(texts[~phishing_mask]),
        "phishing_lengths": _length_stats(texts[phishing_mask]),
        "raw_token_vocabulary": token_vocabulary_size,
        "meta": meta,
    }
    if vectorizer is not None:
        stats["tfidf_vocabulary_size"] = int(len(vectorizer.vocabulary_))
        stats["tfidf_feature_dim"] = int(vectorizer.transform([""]).shape[1])
        stats["tfidf_ngram_range"] = list(vectorizer.ngram_range)
        stats["tfidf_stop_words"] = (
            "none" if vectorizer.stop_words is None else len(set(vectorizer.get_stop_words() or []))
        )
    return stats


def write_dataset_statistics(
    stats: dict[str, Any], md_path: Path, json_path: Path
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Dataset Statistics",
        "",
        "Generated directly from the dataset file by `src/run_experiments.py`.",
        "Every number below is measured, not assumed.",
        "",
        f"- Source file: `{stats['meta']['source_file']}`",
        f"- Rows in file: {stats['meta']['n_rows_raw']:,}",
        f"- Rows used: {stats['total_records']:,}",
        f"- Body column: `{stats['meta']['body_column']}`",
        f"- Label column: `{stats['meta']['label_column']}`",
        f"- Label encoding: {stats['meta']['label_meta']['encoding']}",
        f"- Label mapping: `{stats['meta']['label_meta']['mapping']}`",
        "",
        "## Class distribution",
        "",
        "| Label | Class | Count | Share |",
        "|---|---|---|---|",
    ]
    for row in stats["class_distribution"]:
        lines.append(
            f"| {row['label']} | {row['class']} | {row['count']:,} | {row['percent']:.2f}% |"
        )

    lines += [
        "",
        "## Data quality and text size",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Missing email bodies | {stats['missing_bodies']:,} |",
        f"| Rows dropped for missing label | {stats['missing_labels_dropped']:,} |",
        f"| Exact duplicate bodies | {stats['duplicate_bodies']:,} |",
        f"| Mean body length (characters) | {stats['all_lengths']['mean_chars']:,.1f} |",
        f"| Median body length (characters) | {stats['all_lengths']['median_chars']:,.1f} |",
        f"| Mean body length (whitespace tokens) | {stats['all_lengths']['mean_words']:,.1f} |",
        f"| Mean length, legitimate (characters) | {stats['legitimate_lengths']['mean_chars']:,.1f} |",
        f"| Mean length, phishing (characters) | {stats['phishing_lengths']['mean_chars']:,.1f} |",
    ]
    if "tfidf_vocabulary_size" in stats:
        lines += [
            f"| TF-IDF vocabulary (fit on training split only) | {stats['tfidf_vocabulary_size']:,} |",
            f"| TF-IDF feature dimension | {stats['tfidf_feature_dim']:,} |",
            f"| TF-IDF n-gram range | {tuple(stats['tfidf_ngram_range'])} |",
            f"| Distinct stop words removed | {stats['tfidf_stop_words']} |",
        ]
    if stats.get("raw_token_vocabulary"):
        lines.append(
            f"| Distinct raw whitespace tokens | {stats['raw_token_vocabulary']:,} |"
        )

    lines += [
        "",
        "Note: the TF-IDF vocabulary is fitted on the training split only, so the",
        "reported dimension is the one the classifiers actually saw.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
