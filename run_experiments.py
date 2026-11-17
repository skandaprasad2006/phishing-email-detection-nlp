"""Run the full audit / experiment suite.

Usage
-----
    python -m src.run_experiments --data data/CEAS_08.csv
    python -m src.run_experiments --data CEAS_08.csv.zip --tag main
    python -m src.run_experiments --synthetic 800 --tag smoke --quick --cv-folds 3

Outputs go to `results/` and `reports/` when `--tag main` (the default), and to
`results/_<tag>/` and `reports/_<tag>/` otherwise, so a smoke test can never
overwrite real results.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import (
    config,
    data_io,
    evaluate,
    experiments,
    pipeline,
    plots,
    preprocessing,
    reporting,
)
from .synthetic import synthetic_dataframe

MODEL_KEYS = ["random_forest", "logistic_regression"]


# --------------------------------------------------------------------------- #
@dataclass
class OutputPaths:
    tag: str = "main"
    results: Path = Path()
    reports: Path = Path()

    def __post_init__(self) -> None:
        if self.tag == "main":
            self.results = config.RESULTS_DIR
            self.reports = config.REPORTS_DIR
        else:
            self.results = config.RESULTS_DIR / f"_{self.tag}"
            self.reports = config.REPORTS_DIR / f"_{self.tag}"

    def make(self) -> None:
        self.results.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="CEAS_08 .csv or .csv.zip; defaults to first file in data/")
    parser.add_argument("--synthetic", type=int, default=0, help="use N synthetic rows (code testing only)")
    parser.add_argument("--tag", default="main", help="output tag; anything but 'main' writes to underscore folders")
    parser.add_argument("--quick", action="store_true", help="fewer trees, faster; never report quick-mode numbers")
    parser.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    parser.add_argument("--cv-jobs", type=int, default=1)
    parser.add_argument("--csv-jobs", type=int, default=1, help="unused; reserved")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--max-single-latency-samples", type=int, default=2000)
    return parser.parse_args(argv)


def log(message: str, steps: list[str] | None = None) -> None:
    print(f"[run] {message}", flush=True)
    if steps is not None:
        steps.append(message)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = time.perf_counter()
    timestamp = datetime.now().isoformat(timespec="seconds")
    out = OutputPaths(args.tag)
    out.make()

    steps: list[str] = []
    outputs: list[str] = []

    # ---- 1. Load and verify the dataset -------------------------------- #
    if args.synthetic:
        log(f"loading synthetic data ({args.synthetic} rows) - code test only", steps)
        df = synthetic_dataframe(args.synthetic, seed=config.SEED)
        csv_path = Path("<synthetic>")
    else:
        df, csv_path = data_io.load_dataframe(args.data)
        log(f"loaded {csv_path.name}: {len(df):,} rows x {df.shape[1]} columns", steps)

    texts, labels, meta = data_io.prepare_dataset(df, csv_path)
    log(
        f"schema verified: body='{meta['body_column']}', label='{meta['label_column']}', "
        f"rows used={len(texts):,}",
        steps,
    )

    # ---- 2. Split raw text, then clean (cleaning is stateless) --------- #
    raw_train, raw_test, y_train, y_test = pipeline.split_dataset(texts, labels)
    train_texts = preprocessing.clean_series(pd.Series(raw_train)).to_numpy(dtype=object)
    test_texts = preprocessing.clean_series(pd.Series(raw_test)).to_numpy(dtype=object)
    log(
        f"split: train={len(train_texts):,} test={len(test_texts):,} "
        f"(phishing in test={(np.asarray(y_test) == config.POSITIVE_LABEL).sum():,})",
        steps,
    )

    # ---- 3. Holdout evaluation ---------------------------------------- #
    results: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    for key in MODEL_KEYS:
        result = experiments.run_holdout(
            train_texts, test_texts, y_train, y_test, key, quick=args.quick
        )
        results[key] = result
        metric_rows.append(result["metrics"])
        log(
            f"{result['metrics']['model']}: acc={result['metrics']['accuracy']:.4f} "
            f"auc={result['metrics']['roc_auc']:.4f} features={result['n_features']:,}",
            steps,
        )

    baseline = experiments.run_holdout(
        train_texts, test_texts, y_train, y_test, "dummy_majority", quick=args.quick
    )
    metric_rows.append(baseline["metrics"])
    log(f"baseline accuracy={baseline['metrics']['accuracy']:.4f}", steps)

    metrics_df = evaluate.metrics_frame(metric_rows)
    metrics_path = out.results / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    outputs.append(str(metrics_path.relative_to(config.PROJECT_ROOT)))

    # ---- 4. Dataset statistics (vocabulary from the training fit) ------ #
    rf_pipe = results["random_forest"]["pipeline"]
    training_tokens: set[str] = set()
    for text in raw_train:
        training_tokens.update(str(text).lower().split())
    stats = data_io.dataset_statistics(
        texts,
        labels,
        meta,
        vectorizer=rf_pipe.named_steps["tfidf"],
        token_vocabulary_size=len(training_tokens),
    )
    data_io.write_dataset_statistics(
        stats,
        out.reports / "dataset_statistics.md",
        out.results / "dataset_statistics.json",
    )
    outputs += [
        str((out.reports / "dataset_statistics.md").relative_to(config.PROJECT_ROOT)),
        str((out.results / "dataset_statistics.json").relative_to(config.PROJECT_ROOT)),
    ]
    log(f"dataset statistics written ({stats['total_records']:,} records)", steps)

    # ---- 5. Figures ---------------------------------------------------- #
    y_test_array = np.asarray(y_test).astype(int)

    curves = {}
    for key, result in results.items():
        if result["y_score"] is None:
            continue
        fpr, tpr, auc = evaluate.roc_points(y_test_array, result["y_score"])
        curves[pipeline.MODEL_LABELS[key]] = (fpr, tpr, auc)
    roc_path = out.results / "roc_curve.png"
    plots.plot_roc(curves, roc_path)
    outputs.append(str(roc_path.relative_to(config.PROJECT_ROOT)))

    for key, result in results.items():
        slug = "rf" if key == "random_forest" else "lr"
        path = out.results / f"confusion_matrix_{slug}.png"
        plots.plot_confusion(
            y_test_array,
            result["y_pred"],
            f"{pipeline.MODEL_LABELS[key]} - Confusion Matrix",
            path,
        )
        outputs.append(str(path.relative_to(config.PROJECT_ROOT)))

    class_plot = out.results / "class_distribution.png"
    plots.plot_class_distribution(stats["class_distribution"], class_plot)
    outputs.append(str(class_plot.relative_to(config.PROJECT_ROOT)))
    log("figures: roc_curve.png, confusion_matrix_rf.png, confusion_matrix_lr.png", steps)

    # ---- 6. Feature importance ---------------------------------------- #
    features = pipeline.top_features(rf_pipe, k=config.TOP_FEATURES)
    feature_path = out.results / "feature_importance.png"
    plots.plot_feature_importance(features, feature_path)
    pd.DataFrame(features, columns=["feature", "importance"]).to_csv(
        out.results / "feature_importance.csv", index=False
    )
    outputs += [
        str(feature_path.relative_to(config.PROJECT_ROOT)),
        str((out.results / "feature_importance.csv").relative_to(config.PROJECT_ROOT)),
    ]
    log(f"feature importance: top term '{features[0][0]}'", steps)

    # ---- 7. Operating-point analysis ---------------------------------- #
    threshold_frames: dict[str, pd.DataFrame] = {}
    for key, result in results.items():
        if result["y_score"] is None:
            continue
        frame = evaluate.threshold_table(y_test_array, result["y_score"])
        label = pipeline.MODEL_LABELS[key]
        threshold_frames[label] = frame
        slug = "rf" if key == "random_forest" else "lr"
        frame.to_csv(out.results / f"threshold_analysis_{slug}.csv", index=False)
        outputs.append(str((out.results / f"threshold_analysis_{slug}.csv").relative_to(config.PROJECT_ROOT)))

    # ---- 8. Cross-validation ------------------------------------------ #
    cv_summary = None
    if args.skip_cv:
        log("cross-validation skipped", steps)
    else:
        cv_summary, cv_folds = experiments.run_cross_validation(
            train_texts,
            y_train,
            MODEL_KEYS,
            folds=args.cv_folds,
            quick=args.quick,
            n_jobs=args.cv_jobs,
        )
        cv_summary.to_csv(out.results / "cross_validation_summary.csv", index=False)
        cv_folds.to_csv(out.results / "cross_validation_results.csv", index=False)
        outputs += [
            str((out.results / "cross_validation_summary.csv").relative_to(config.PROJECT_ROOT)),
            str((out.results / "cross_validation_results.csv").relative_to(config.PROJECT_ROOT)),
        ]
        for row in cv_summary.to_dict("records"):
            log(
                f"CV {row['model']}: acc={row['accuracy_mean']:.4f}"
                f"+/-{row['accuracy_std']:.4f} auc={row['roc_auc_mean']:.4f}",
                steps,
            )

    # ---- 9. Ablation -------------------------------------------------- #
    ablation_df = None
    if args.skip_ablation:
        log("ablation skipped", steps)
    else:
        raw_train_array = np.asarray(raw_train, dtype=object)
        ablation_df = experiments.run_ablation(
            {"clean": train_texts, "raw": raw_train_array},
            {"clean": test_texts, "raw": np.asarray(raw_test, dtype=object)},
            y_train,
            y_test,
            model_name="random_forest",
            quick=args.quick,
        )
        ablation_df.to_csv(out.results / "ablation_results.csv", index=False)
        outputs.append(str((out.results / "ablation_results.csv").relative_to(config.PROJECT_ROOT)))
        log(f"ablation: {len(ablation_df)} configurations", steps)

    # ---- 10. Error analysis ------------------------------------------- #
    error_details: dict[str, Any] = {}
    if args.skip_errors:
        log("error analysis skipped", steps)
    else:
        for key, result in results.items():
            label = pipeline.MODEL_LABELS[key]
            error_details[label] = experiments.error_analysis(
                test_texts, y_test, result["y_pred"], result["y_score"]
            )
            slug = "rf" if key == "random_forest" else "lr"
            experiments.write_error_rows(
                test_texts,
                y_test,
                result["y_pred"],
                result["y_score"],
                label,
                out.results / f"error_samples_{slug}.csv",
            )
            outputs.append(str((out.results / f"error_samples_{slug}.csv").relative_to(config.PROJECT_ROOT)))
        log("error analysis written", steps)

    # ---- 11. Inference cost ------------------------------------------- #
    deployment_rows = []
    for key, result in results.items():
        row = experiments.deployment_metrics(
            result["pipeline"],
            test_texts,
            n_single=args.max_single_latency_samples,
            model_label=pipeline.MODEL_LABELS[key],
        )
        deployment_rows.append(row)
    pd.DataFrame(deployment_rows).to_csv(out.results / "deployment_metrics.csv", index=False)
    outputs.append(str((out.results / "deployment_metrics.csv").relative_to(config.PROJECT_ROOT)))
    log("inference cost measured", steps)

    # ---- 12. Legacy protocol comparison -------------------------------- #
    legacy = None
    if args.skip_legacy:
        log("legacy comparison skipped", steps)
    else:
        legacy = pipeline.legacy_leaky_reproduction(texts, labels, model_name="random_forest")
        legacy["model"] = pipeline.MODEL_LABELS["random_forest"]
        pd.DataFrame([legacy]).to_csv(out.results / "legacy_protocol_comparison.csv", index=False)
        outputs.append(str((out.results / "legacy_protocol_comparison.csv").relative_to(config.PROJECT_ROOT)))
        log(f"legacy leaky protocol accuracy={legacy['accuracy']:.4f}", steps)

    # ---- 13. Reports --------------------------------------------------- #
    draft_checks = reporting.verify_draft_arithmetic()
    runtime = time.perf_counter() - started_at
    context: dict[str, Any] = {
        "tag": args.tag,
        "timestamp": timestamp,
        "stats": stats,
        "metrics_df": metrics_df,
        "cv_summary": cv_summary,
        "ablation_df": ablation_df,
        "errors": error_details,
        "deployment": deployment_rows,
        "legacy": legacy,
        "feature_importance": features,
        "thresholds": threshold_frames,
        "draft_checks": draft_checks,
        "quick": args.quick,
        "stopword_source": preprocessing.stopword_source(),
        "rf_estimators": config.RF_ESTIMATORS_QUICK if args.quick else config.RF_ESTIMATORS,
        "cv_folds": 0 if args.skip_cv else args.cv_folds,
        "runtime_seconds": runtime,
        "steps": steps,
        "outputs": outputs,
    }
    reporting.write_initial_audit(out.reports / "initial_research_audit.md", context)
    reporting.write_experimental_results(out.reports / "experimental_results.md", context)
    reporting.write_cv_report(out.reports / "cross_validation_report.md", context)
    reporting.write_error_report(out.reports / "error_analysis.md", context)
    reporting.write_limitations(out.reports / "limitations_and_future_work.md", context)
    reporting.write_final_audit(out.reports / "final_ieee_paper_audit.md", context)
    reporting.write_experiment_log(out.reports / "experiment_log.md", context)

    summary = {
        "tag": args.tag,
        "timestamp": timestamp,
        "quick_mode": args.quick,
        "data_source": meta["source_file"],
        "rows_used": stats["total_records"],
        "runtime_seconds": round(runtime, 1),
        "metrics": metrics_df.to_dict("records"),
        "roc_auc": {name: auc for name, (_, _, auc) in curves.items()},
        "draft_checks_failed": [c["check"] for c in draft_checks if c["status"] == "FAIL"],
        "outputs": sorted(outputs),
    }
    (out.results / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    log(f"done in {runtime:.1f}s - outputs in {out.results} and {out.reports}", steps)
    print("\n=== Holdout metrics ===")
    print(evaluate.format_table(metrics_df))
    failed = summary["draft_checks_failed"]
    print(f"\nDraft self-consistency checks failed: {len(failed)}")
    for item in failed:
        print(f"  - {item}")
    if args.quick or args.synthetic:
        print("\nNOTE: quick/synthetic run. These numbers are NOT results for the paper.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
