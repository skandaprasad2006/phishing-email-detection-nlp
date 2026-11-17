"""Markdown report generation.

Every number written here is injected from a computed object (metrics frames,
statistics dicts, measured timings). Report text never contains a hard-coded
performance value.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import config, evaluate

# Verified during the audit by checking the published record. Metadata in the
# draft does not match the real papers below (wrong title, year, venue or method),
# which is more dangerous than an obviously missing citation: a reviewer who looks
# up the DOI finds a different paper.
REFERENCE_FINDINGS = [
    {
        "ref": "[6]",
        "in_draft": "Alhogail & Alsabih (2023), 'Social engineering tactic detection in phishing "
        "emails using SVM and psychological feature annotation', Computers & Security 130, 103258; "
        "Table I lists 'SVM, psych. feat.' at 96.1%",
        "actual": "Alhogail, A., & Alsabih, A. (2021). Applying machine learning and natural language "
        "processing to detect phishing email. Computers & Security, 110, 102414. "
        "doi:10.1016/j.cose.2021.102414. Method: graph convolutional network + NLP over body text; "
        "reported accuracy 98.2% with FPR 0.015 on the Radev/Nazario-style corpus.",
        "status": "MISATTRIBUTED",
    },
    {
        "ref": "[14]",
        "in_draft": "Basit, Zafar, Liu & Javed (2024), 'A comprehensive taxonomy of phishing email "
        "detection: NLP, URL, and header-based approaches', ACM Computing Surveys 57(3), 1-35",
        "actual": "Basit, A., Zafar, M., Liu, X., et al. (2021). A comprehensive survey of AI-enabled "
        "phishing attacks detection techniques. Telecommunication Systems, 76, 139-154. "
        "doi:10.1007/s11235-020-00733-2",
        "status": "MISATTRIBUTED",
    },
    {
        "ref": "[8]",
        "in_draft": "Maneriker, Reddy & Bou (2024), 'Character-level neural networks for adversarially "
        "obfuscated phishing email detection', IEEE DSC, 1-8, 93.8%",
        "actual": "No matching publication found in this audit. Treat as unverified and replace.",
        "status": "UNVERIFIED",
    },
]

UNVERIFIED_REFS_NOTE = (
    "References [1]-[5], [7], [9]-[13] and [15] were not checked one by one in this audit. "
    "Before submission, every entry needs its DOI resolved and the title, venue, year and "
    "reported metric confirmed against the publisher record. Entries that cannot be confirmed "
    "must be removed, not repaired from memory."
)


# --------------------------------------------------------------------------- #
# Draft self-consistency audit
# --------------------------------------------------------------------------- #
def verify_draft_arithmetic() -> list[dict[str, Any]]:
    """Check the numbers in the current draft against each other.

    These checks use only values printed in the draft itself. They do not depend
    on the dataset, which is the point: some of them already fail.
    """
    claims = config.DRAFT_CLAIMS
    n = claims["reported_test_partition"]
    legit = claims["reported_test_legitimate"]
    phish = claims["reported_test_phishing"]
    fp = claims["reported_rf_false_positives"]
    fn = claims["reported_rf_false_negatives"]

    tp = phish - fn
    tn = legit - fp

    checks: list[dict[str, Any]] = []

    def add(label: str, claimed: float, recomputed: float, rate: bool = True) -> None:
        difference = abs(float(claimed) - float(recomputed))
        tolerance = 0.0005 if rate else 0.5
        checks.append(
            {
                "check": label,
                "claimed": f"{claimed:.4f}" if rate else f"{claimed:,.0f}",
                "recomputed": f"{recomputed:.4f}" if rate else f"{recomputed:,.0f}",
                "difference": f"{difference:.4f}" if rate else f"{difference:,.0f}",
                "status": "PASS" if difference <= tolerance else "FAIL",
            }
        )

    add("Test partition equals legitimate + phishing rows", n, legit + phish, rate=False)
    add(
        "Accuracy printed in Table II vs accuracy implied by the confusion-matrix counts",
        claims["reported_rf_accuracy"],
        (tp + tn) / n,
    )
    add(
        "Precision printed in Table III vs precision implied by the 20 false positives",
        claims["reported_rf_precision"],
        tp / (tp + fp),
    )
    add(
        "Recall printed in Table III vs recall implied by the 15 false negatives",
        claims["reported_rf_recall"],
        tp / (tp + fn),
    )
    add(
        "Majority-class baseline printed in the discussion vs actual phishing share of the test set",
        claims["reported_majority_baseline"],
        phish / n,
    )
    add(
        "Total dataset size printed in Table I/IV vs total implied by a 20% test partition",
        claims["reported_total_emails"],
        n / config.TEST_SIZE,
        rate=False,
    )
    return checks


def _checks_markdown(checks: list[dict[str, Any]]) -> str:
    header = "| Check | Value in draft | Recomputed from draft | Difference | Status |"
    divider = "|---|---|---|---|---|"
    rows = [
        f"| {c['check']} | {c['claimed']} | {c['recomputed']} | {c['difference']} | **{c['status']}** |"
        for c in checks
    ]
    return "\n".join([header, divider, *rows])


# --------------------------------------------------------------------------- #
# 1. Initial research audit
# --------------------------------------------------------------------------- #
def write_initial_audit(path, ctx: dict[str, Any]) -> None:
    stats = ctx["stats"]
    meta = stats["meta"]
    net = stats["class_distribution"]
    checks = ctx["draft_checks"]
    failed = [c for c in checks if c["status"] == "FAIL"]

    lines = [
        "# Initial Research Audit",
        "",
        f"Generated: {ctx['timestamp']}  ",
        f"Run tag: `{ctx['tag']}`  ",
        f"Data source: `{meta['source_file']}`",
        "",
        "This audit was produced by inspecting the upstream repository",
        "(`Ripebrusser/AI-Based-Phishing-Email-Detection-Using-Natural-Language-Processing`,",
        "five scripts `code1.py`-`code5.py`) and the draft paper, then measuring the",
        "dataset directly. Statements about the *old code* are observations of that code.",
        "Statements about the *dataset* are computed below. Nothing here is estimated.",
        "",
        "## 1. What the original implementation actually did",
        "",
        "| Step | Original code | Defect |",
        "|---|---|---|",
        "| Load | `pd.read_csv('CEAS_08.csv')` after zip extraction | Fine, but the file was never schema-checked |",
        "| Input text | `df['body'].astype(str)` | Fine |",
        "| Features | `TfidfVectorizer(stop_words='english').fit_transform(X)` on **all** rows | **Critical: data leakage.** Vocabulary and IDF weights were learned from the test rows |",
        "| Split | `train_test_split(X, y, test_size=0.2)` | No `random_state` (irreproducible), no `stratify` (class balance not preserved) |",
        "| Random Forest | `RandomForestClassifier()` defaults | No `random_state`; results cannot be reproduced |",
        "| Logistic Regression | `LogisticRegression(max_iter=1000)` | No `random_state` |",
        "| Metrics | accuracy only; `precision_score/recall_score/f1_score` on the RF predictions | No LR precision/recall/F1, no ROC/AUC, no per-class breakdown for LR |",
        "| Confusion matrix | `plt.imshow(cm)` with default colormap | No axis labels or class names; not publication quality |",
        "| Baseline | none | No majority-class reference |",
        "| Validation | single holdout split | No cross-validation, no variance estimate |",
        "",
        "The leakage is the most serious defect: because the vectorizer saw the test",
        "documents, the reported scores are optimistic and the size of that optimism",
        "was never quantified. This run quantifies it (section 8).",
        "",
        "## 2. Dataset actually measured",
        "",
        f"- Rows in file: **{meta['n_rows_raw']:,}**",
        f"- Rows used: **{stats['total_records']:,}**",
        f"- Body column resolved to: `{meta['body_column']}`",
        f"- Label column resolved to: `{meta['label_column']}`",
        f"- Label encoding: {meta['label_meta']['encoding']}",
        f"- Label mapping applied: `{meta['label_meta']['mapping']}`",
        f"- Missing bodies: {stats['missing_bodies']:,}",
        f"- Exact duplicate bodies: {stats['duplicate_bodies']:,}",
        "",
        "| Label | Class | Count | Share |",
        "|---|---|---|---|",
    ]
    for row in net:
        lines.append(f"| {row['label']} | {row['class']} | {row['count']:,} | {row['percent']:.2f}% |")

    lines += [
        "",
        "Text size: mean body length "
        f"{stats['all_lengths']['mean_chars']:,.0f} characters "
        f"({stats['all_lengths']['mean_words']:,.0f} whitespace tokens); "
        f"legitimate {stats['legitimate_lengths']['mean_chars']:,.0f} vs phishing "
        f"{stats['phishing_lengths']['mean_chars']:,.0f} characters.",
        "",
        "## 3. Self-consistency of the numbers already printed in the draft",
        "",
        _checks_markdown(checks),
        "",
    ]
    if failed:
        lines += [
            f"**{len(failed)} of {len(checks)} internal checks fail.** The draft's tables",
            "cannot all be true at once, so no combination of them may be reused. The most",
            "consequential failures:",
            "",
        ]
        for check in failed:
            lines.append(
                f"- *{check['check']}* - draft says {check['claimed']}, its own counts "
                f"imply {check['recomputed']}."
            )
        lines += [
            "",
            "The dataset-size discrepancy also means the abstract's \"approximately 18,000",
            "labelled messages\" is not consistent with the 7,831-row test partition printed",
            "in Section V. Only one of these can be right, and the file itself decides which.",
            "",
        ]

    lines += [
        "## 4. Missing experiments (state before this run)",
        "",
        "| Missing item | Status after this run |",
        "|---|---|",
        f"| ROC curve and AUC | Generated: `results/roc_curve.png`, `results/model_metrics.csv` |",
        f"| Complete LR metrics | Generated for both models in `results/model_metrics.csv` |",
        f"| Formal majority-class baseline | Generated (`dummy_majority` row) |",
        f"| Dataset statistics | `reports/dataset_statistics.md` |",
        f"| Stratified 5-fold cross-validation | `reports/cross_validation_report.md` |",
        f"| Feature analysis | `results/feature_importance.png` |",
        f"| Error analysis | `reports/error_analysis.md` |",
        f"| Ablation study | `results/ablation_results.csv` |",
        f"| Inference cost measurements | `results/deployment_metrics.csv` |",
        f"| Single-dataset generalisation limit | `reports/limitations_and_future_work.md` |",
        "",
        "## 5. Corrections applied to the implementation",
        "",
        "1. All learned transforms (TF-IDF vocabulary, IDF weights, classifier",
        "   parameters) now live inside one `sklearn.pipeline.Pipeline`, so they are",
        "   fitted on training data only.",
        "2. The split is stratified and seeded (`random_state=42`), and every estimator",
        "   that accepts a seed receives it.",
        "3. The positive class probability is read from `clf.classes_` instead of",
        "   assuming `predict_proba[:, 1]` is phishing.",
        "4. Every model now reports accuracy, precision, recall, F1, ROC-AUC, the full",
        "   confusion matrix, FPR and FNR from the same code path.",
        "5. Cross-validation re-fits the vectorizer inside each fold.",
        "6. Text cleaning is stateless and per-document, so it cannot leak; this is",
        "   documented rather than assumed.",
        "",
        "## 6. Paper-level inconsistencies to fix in the LaTeX source",
        "",
        "- Section IV states the dataset contains \"approximately 18,000\" messages while",
        "  Section V reports a 7,831-row test set; a 20% test set implies about 39,000 rows.",
        "  Replace both with the measured row count above.",
        "- Section IV calls the split \"stratified\" although the code used no",
        "  `stratify=` argument. It is stratified now; the sentence was false before.",
        "- Table III's per-class supports (3,456 / 4,375) must be recomputed from the new",
        "  split; they cannot be inherited.",
        "- The discussion compares against \"approximately 56%\" without stating that it is",
        "  the predicted-majority rate of this specific test split; report it as a measured",
        "  DummyClassifier result instead.",
        "- The reference list must be verified entry by entry before submission (see",
        "  `reports/final_ieee_paper_audit.md`). Several entries are not traceable in this",
        "  audit and must be replaced with sources whose DOI resolves.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _ablation_table(frame: pd.DataFrame) -> str:
    header = (
        "| Variant | Features | Accuracy | Precision | Recall | F1 | ROC-AUC "
        "| FP | FN | Fit (s) |"
    )
    divider = "|---|---|---|---|---|---|---|---|---|---|"
    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            f"| {row['variant']} | {int(row['features']):,} "
            f"| {100 * row['accuracy']:.2f}% | {100 * row['precision']:.2f}% "
            f"| {100 * row['recall']:.2f}% | {100 * row['f1']:.2f}% "
            f"| {100 * row['roc_auc']:.2f}% | {int(row['false_positive']):,} "
            f"| {int(row['false_negative']):,} | {row['fit_seconds']:.1f} |"
        )
    return "\n".join([header, divider, *rows])


# --------------------------------------------------------------------------- #
# 2. Experimental results
# --------------------------------------------------------------------------- #
def write_experimental_results(path, ctx: dict[str, Any]) -> None:
    metrics_df: pd.DataFrame = ctx["metrics_df"]
    lines = [
        "# Experimental Results",
        "",
        f"Generated: {ctx['timestamp']}  Run tag: `{ctx['tag']}`  ",
        f"Protocol: stratified {int(100 * (1 - config.TEST_SIZE))}:{int(100 * config.TEST_SIZE)} "
        f"train/test holdout split, "
        f"`random_state={config.SEED}`; TF-IDF fitted on training texts only.",
        "",
        "## 1. Holdout performance",
        "",
        evaluate.format_table(metrics_df),
        "",
        "Confusion-matrix counting convention: rows are true labels, columns are",
        "predicted labels, positive class is phishing (label 1).",
        "",
        "## 2. Confusion matrices and derived error rates",
        "",
        "| Model | TP | TN | FP | FN | FPR = FP/(FP+TN) | FNR = FN/(FN+TP) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in metrics_df.to_dict("records"):
        lines.append(
            f"| {row['model']} | {row['true_positive']:,} | {row['true_negative']:,} | "
            f"{row['false_positive']:,} | {row['false_negative']:,} | "
            f"{100 * row['false_positive_rate']:.2f}% | {100 * row['false_negative_rate']:.2f}% |"
        )
    lines += [
        "",
        "Figures: `results/confusion_matrix_rf.png`, `results/confusion_matrix_lr.png`.",
        "",
        "## 3. ROC and AUC",
        "",
        "| Model | ROC-AUC |",
        "|---|---|",
    ]
    for row in metrics_df.to_dict("records"):
        auc = row["roc_auc"]
        lines.append(f"| {row['model']} | {'n/a' if pd.isna(auc) else f'{auc:.4f}'} |")
    lines += [
        "",
        "Figure: `results/roc_curve.png` (all models on one panel with the chance line).",
        "",
    ]

    thresholds = ctx.get("thresholds") or {}
    if thresholds:
        lines += [
            "## 4. Operating-point analysis",
            "",
            "The security trade-off is a threshold choice, not only a model choice:",
            "lowering the threshold catches more phishing at the cost of more false alarms.",
            "",
        ]
        for label, frame in thresholds.items():
            lines += [f"### {label}", "", evaluate.format_table(frame, float_digits=3), ""]

    features = ctx.get("feature_importance") or []
    if features:
        lines += [
            "## 5. Feature importance (Random Forest, top terms)",
            "",
            "These are impurity-based importance scores from the fitted forest. They",
            "indicate which TF-IDF dimensions the model used most; they are not causal",
            "evidence that a term makes an email phishing.",
            "",
            "| Rank | Feature | Importance |",
            "|---|---|---|",
        ]
        for rank, (name, value) in enumerate(features, start=1):
            lines.append(f"| {rank} | `{name}` | {value:.5f} |")
        lines += ["", "Figure: `results/feature_importance.png`.", ""]

    ablation = ctx.get("ablation_df")
    if ablation is not None and not ablation.empty:
        lines += [
            "## 6. Ablation study",
            "",
            "One factor changed at a time, same split and seed, Random Forest. "
            "`features` is the TF-IDF dimensionality that configuration produced.",
            "",
            _ablation_table(ablation),
            "",
            "Read the deltas as descriptive of this dataset and this split; no",
            "significance test was performed on these differences.",
            "",
        ]

    legacy = ctx.get("legacy")
    if legacy:
        lines += [
            "## 7. Effect of the original leakage",
            "",
            "The original protocol (TF-IDF fitted on the whole dataset before an",
            "unstratified, unseeded split) was reproduced verbatim so the leak can be",
            "quantified rather than asserted.",
            "",
            "| Protocol | Model | Accuracy | TF-IDF fitted on |",
            "|---|---|---|---|",
            f"| Legacy (leaky) | {legacy['model']} | {100 * legacy['accuracy']:.2f}% | all rows, including test |",
        ]
        corrected = metrics_df[metrics_df["model"] == legacy["model"]]
        if not corrected.empty:
            value = float(corrected.iloc[0]["accuracy"])
            lines.append(
                f"| Corrected (this run) | {legacy['model']} | {100 * value:.2f}% | training rows only |"
            )
            lines += [
                "",
                f"Difference: {100 * (legacy['accuracy'] - value):+.2f} percentage points"
                " (legacy minus corrected). The two protocols use different splits, so this",
                "is an indicative comparison of protocols, not a paired test on identical rows.",
            ]
        lines.append("")

    deployment = ctx.get("deployment") or []
    if deployment:
        lines += [
            "## 8. Inference cost",
            "",
            "Measured on the test partition inside this run; single-email latency was",
            "averaged over individually scored emails after a warm-up call.",
            "",
            "| Model | Artifact size | Latency mean | Latency p95 | Batch throughput |",
            "|---|---|---|---|---|",
        ]
        for row in deployment:
            lines.append(
                f"| {row['model']} | {row['artifact_mb']} MB | "
                f"{row['single_email_latency_ms_mean']} ms | "
                f"{row['single_email_latency_ms_p95']} ms | "
                f"{row['batch_throughput_emails_per_second']:,} emails/s |"
            )
        lines += [
            "",
            "These are single-machine CPU measurements from one run, not a",
            "production benchmark. They support a statement such as \"classification adds",
            "a few milliseconds of CPU time per message\", and nothing stronger.",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 3. Cross-validation
# --------------------------------------------------------------------------- #
def write_cv_report(path, ctx: dict[str, Any]) -> None:
    summary: pd.DataFrame | None = ctx.get("cv_summary")
    if summary is None or summary.empty:
        path.write_text(
            "# Cross-Validation Report\n\nCross-validation was skipped for this run "
            "(`--skip-cv`). Do not report cross-validation numbers.\n",
            encoding="utf-8",
        )
        return

    lines = [
        "# Stratified Cross-Validation Report",
        "",
        f"Generated: {ctx['timestamp']}  Run tag: `{ctx['tag']}`",
        "",
        f"{int(summary['folds'].iloc[0])}-fold stratified cross-validation, "
        f"`shuffle=True`, `random_state={config.SEED}`, stratified k-fold.",
        "The TF-IDF vectorizer is re-fitted inside every training fold, so no fold's",
        "validation documents influence the vocabulary or the IDF weights.",
        "",
        "**These numbers are a separate experiment from the holdout results.** They",
        "estimate variance across folds; they are not the performance of the model saved",
        "from the holdout run, and the two must not be mixed in one table.",
        "",
        "| Model | Metric | Mean | Std. dev. |",
        "|---|---|---|---|",
    ]
    for row in summary.to_dict("records"):
        for metric in evaluate.CV_SCORING_NAMES:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            lines.append(
                f"| {row['model']} | {metric} | {100 * mean:.2f}% | {100 * std:.2f} pp |"
            )
    lines += [
        "",
        "Mean fit time per fold (seconds): "
        + ", ".join(
            f"{row['model']} {row['fit_seconds_mean']:.1f}s" for row in summary.to_dict("records")
        ),
        "",
        "Per-fold values are in `results/cross_validation_results.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 4. Error analysis
# --------------------------------------------------------------------------- #
def write_error_report(path, ctx: dict[str, Any]) -> None:
    errors: dict[str, dict[str, Any]] = ctx.get("errors") or {}
    lines = [
        "# Error Analysis",
        "",
        f"Generated: {ctx['timestamp']}  Run tag: `{ctx['tag']}`",
        "",
        "Misclassifications on the holdout test set, described by measured properties",
        "only. Each statement below is a count or a rate over the actual errors; no",
        "cause is asserted that the data does not show.",
        "",
    ]
    if not errors:
        lines.append("No error analysis was produced for this run.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines += [
        "| Model | Error type | Count | Mean chars | Median chars | Mean words | Share with a URL | Share under 200 chars |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model_label, detail in errors.items():
        for kind, key in (("False positive (legitimate marked phishing)", "false_positives"),
                          ("False negative (phishing missed)", "false_negatives")):
            profile = detail[key]
            lines.append(
                f"| {model_label} | {kind} | {profile['count']:,} | {profile['mean_chars']:,.0f} | "
                f"{profile['median_chars']:,.0f} | {profile['mean_words']:,.0f} | "
                f"{100 * profile['share_with_url']:.1f}% | "
                f"{100 * profile['share_shorter_than_200_chars']:.1f}% |"
            )

    lines += ["", "Reference: correct-classification length means, per model.", ""]
    for model_label, detail in errors.items():
        lines.append(f"- {model_label}: mean {detail['correct_length_mean']:,.0f} characters across the test set.")

    lines += ["", "## Tokens over-represented in each error class", ""]
    for model_label, detail in errors.items():
        lines += [f"### {model_label}", ""]
        for title, key in (
            ("Most frequent tokens among false positives", "false_positive_tokens"),
            ("Most frequent tokens among false negatives", "false_negative_tokens"),
            ("Most frequent tokens among correctly detected phishing", "true_positive_tokens"),
        ):
            tokens = detail.get(key) or []
            rendered = ", ".join(f"`{token}` ({count})" for token, count in tokens)
            lines += [f"**{title}:** {rendered if rendered else 'not enough samples'}", ""]
        scores = detail.get("score_summary") or {}
        for kind, values in scores.items():
            lines.append(
                f"- {kind.replace('_', ' ').capitalize()} phishing probability: "
                f"mean {values['mean']}, range {values['min']}-{values['max']}."
            )
        lines.append("")

    lines += [
        "## Masked examples",
        "",
        "Addresses and URLs are redacted and excerpts truncated, so this report can be",
        "shared without exposing message content. Full masked rows are in",
        "`results/error_samples_*.csv`.",
        "",
    ]
    for model_label, detail in errors.items():
        lines += [f"### {model_label}", ""]
        for title, key in (
            ("False positives", "false_positive_excerpts"),
            ("False negatives", "false_negative_excerpts"),
        ):
            lines.append(f"**{title}**")
            lines.append("")
            excerpts = detail.get(key) or []
            if not excerpts:
                lines.append("_none_")
            for excerpt in excerpts:
                lines.append(f"- `{excerpt}`")
            lines.append("")

    lines += [
        "## What this analysis does and does not show",
        "",
        "It shows the measured profile of the mistakes: how many, how long, how often",
        "they contain URLs, and which tokens dominate. It does not establish why any",
        "individual email was misclassified; that would require labelling the error",
        "causes manually, which has not been done.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 5. Limitations
# --------------------------------------------------------------------------- #
def write_limitations(path, ctx: dict[str, Any]) -> None:
    stats = ctx["stats"]
    lines = [
        "# Limitations and Future Work",
        "",
        f"Generated: {ctx['timestamp']}  Run tag: `{ctx['tag']}`",
        "",
        "## Limitations supported by this run",
        "",
        f"1. **One dataset.** All results come from `{stats['meta']['source_file']}` "
        f"({stats['total_records']:,} rows). Nothing here demonstrates performance on "
        "another corpus, a different time period, or live mail. Cross-dataset",
        "   generalisation is untested, so no claim about it may be made.",
        "2. **Benchmark conditions.** The split is random within one corpus, so",
        "   near-duplicate campaigns can appear on both sides of the split. This inflates",
        "   scores relative to a temporal split, which was not run.",
        "3. **No adversarial evaluation.** No obfuscated, paraphrased or deliberately",
        "   evasive samples were tested. Robustness to an adaptive attacker is unknown.",
        "4. **Body text only.** Sender, header, subject and URL structure are excluded,",
        "   although the dataset provides some of them.",
        "5. **Bag-of-words representation.** TF-IDF with a linear or tree model cannot",
        "   represent word order, negation or long-range context.",
        "6. **No significance testing between models.** The Random Forest vs Logistic",
        "   Regression comparison is descriptive. A paired test over folds would be",
        "   needed before calling the difference real, and it was not performed.",
        "7. **Static threshold.** Models are evaluated at the default 0.5 threshold; the",
        "   operating-point table is reported for information, not tuned on test data.",
        "8. **Cost figures are single-machine.** Latency and size were measured on the",
        "   CPU of the machine that ran the experiment.",
        "",
        "## Honest framing of the contribution",
        "",
        "This is a comparative evaluation of two classical classifiers over one TF-IDF",
        "representation of one public phishing corpus, with a leakage-free protocol and",
        "a majority-class baseline. TF-IDF plus Random Forest for phishing detection is",
        "well established; the contribution is the corrected, reproducible measurement",
        "and the quantified effect of the original leakage, not a new algorithm.",
        "",
        "## Future work that would be defensible",
        "",
        "- Repeat on a second, temporally disjoint corpus (for example a later CEAS or",
        "  Nazario/Nazario-style archive) and report the drop.",
        "- Replace the random split with a time-ordered split and compare.",
        "- Test an obfuscation suite (homoglyphs, homophone substitution, HTML hiding)",
        "  and report degradation per transformation.",
        "- Add header and URL features as a separate ablation arm.",
        "- Compare against one contextual-embedding baseline under the same protocol.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 6. Final audit
# --------------------------------------------------------------------------- #
def write_final_audit(path, ctx: dict[str, Any]) -> None:
    metrics_df: pd.DataFrame = ctx["metrics_df"]
    checks = ctx["draft_checks"]
    failed = [c for c in checks if c["status"] == "FAIL"]
    cv = ctx.get("cv_summary")

    def _value(model: str, column: str) -> str:
        row = metrics_df[metrics_df["model"] == model]
        if row.empty:
            return "n/a"
        value = float(row.iloc[0][column])
        return "n/a" if pd.isna(value) else f"{100 * value:.2f}%"

    lines = [
        "# Final Review-Style Assessment",
        "",
        f"Generated: {ctx['timestamp']}  Run tag: `{ctx['tag']}`",
        "",
        "## A. What was already correct",
        "",
        "- The problem framing (TF-IDF + classical classifiers on a labelled public",
        "  corpus) is sound and reproducible in principle.",
        "- The choice of body text as input and a binary legitimate/phishing target is",
        "  consistent with the stated goal.",
        "- The draft correctly identifies that a false negative is more costly than a",
        "  false positive, which is the right security framing.",
        "- The pipeline diagram matches the implemented architecture.",
        f"- The dataset's own schema does use `{ctx['stats']['meta']['body_column']}` and "
        f"`{ctx['stats']['meta']['label_column']}`, so the original column choice was right.",
        "",
        "## B. What was defective",
        "",
        "1. **Data leakage.** TF-IDF was fitted on the full corpus before splitting.",
        "2. **Irreproducible split.** No seed and no stratification.",
        "3. **Incomplete metrics.** Logistic Regression had accuracy only; no ROC/AUC,",
        "   no FPR/FNR, no per-class breakdown.",
        "4. **No baseline.** The 56% figure in the discussion was neither computed nor",
        "   attributed to a classifier.",
        "5. **Internally inconsistent draft numbers.** "
        f"{len(failed)} of {len(checks)} arithmetic checks failed (see "
        "`reports/initial_research_audit.md`).",
        "6. **Unverified references.** The bibliography could not be substantiated in",
        "   this audit; every entry needs a resolvable DOI before submission.",
        "7. **Unsupported deployment language.** Efficiency claims were made without",
        "   any timing measurement.",
        "",
        "## C. What was fixed",
        "",
        "- All fitted transforms moved inside one `Pipeline`; the split happens on raw",
        "  text, stratified and seeded.",
        "- Complete metric set computed from one shared code path for every model,",
        "  including the majority-class baseline.",
        "- ROC/AUC, confusion matrices, FPR/FNR and an operating-point sweep added.",
        "- Stratified 5-fold CV with per-fold vectorizer fitting added.",
        "- Feature-importance extraction with an explicit dimension assertion.",
        "- Ablation over preprocessing and TF-IDF settings on a fixed split.",
        "- Error analysis restricted to measured properties with masked excerpts.",
        "- Inference cost measured rather than asserted.",
        "",
        "## D. Actual results from this run",
        "",
        f"- Random Forest: accuracy {_value('Random Forest', 'accuracy')}, "
        f"precision {_value('Random Forest', 'precision')}, "
        f"recall {_value('Random Forest', 'recall')}, "
        f"F1 {_value('Random Forest', 'f1')}, "
        f"ROC-AUC {_value('Random Forest', 'roc_auc')}.",
        f"- Logistic Regression: accuracy {_value('Logistic Regression', 'accuracy')}, "
        f"precision {_value('Logistic Regression', 'precision')}, "
        f"recall {_value('Logistic Regression', 'recall')}, "
        f"F1 {_value('Logistic Regression', 'f1')}, "
        f"ROC-AUC {_value('Logistic Regression', 'roc_auc')}.",
        f"- Majority-class baseline: accuracy {_value('Majority-class baseline', 'accuracy')}.",
        "",
    ]
    if cv is not None and not cv.empty:
        lines += ["Cross-validation means (separate experiment):", ""]
        for row in cv.to_dict("records"):
            lines.append(
                f"- {row['model']}: accuracy {100 * row['accuracy_mean']:.2f}% "
                f"(+/- {100 * row['accuracy_std']:.2f} pp), "
                f"ROC-AUC {100 * row['roc_auc_mean']:.2f}% "
                f"(+/- {100 * row['roc_auc_std']:.2f} pp) over {int(row['folds'])} folds."
            )
        lines.append("")

    lines += [
        "## E. Remaining limitations",
        "",
        "See `reports/limitations_and_future_work.md`. The binding ones for review are",
        "single-dataset evaluation, a random (not temporal) split, no adversarial",
        "testing, and no significance test between the two classifiers.",
        "",
        "## F. Reviewer-style verdict",
        "",
        "The work is now technically correct and reproducible, which it was not",
        "before. It is still a straightforward classical-ML comparison on one benchmark.",
        "As a conference submission it is defensible only if the paper states exactly",
        "that, reports the corrected numbers, and drops any novelty or deployment claim",
        "the measurements do not support. The main remaining risk is not the modelling:",
        "it is the bibliography, which must be verified entry by entry before the paper",
        "is submitted anywhere.",
        "",
        "### Checklist",
        "",
        "| Item | Status |",
        "|---|---|",
        "| Scripts execute end to end | see `results/run_summary.json` and `reports/experiment_log.md` |",
        "| Every figure generated from measured values | yes - no value is hard-coded in the plotting code |",
        "| Tables verified against raw outputs | `results/*.csv` are the raw outputs |",
        "| Draft numbers reconciled | no - the draft's own numbers disagree; they must be replaced |",
        "| References verified | **no - see section G; 2 confirmed misattributions, 1 unverified** |",
        "| Author block corrected in LaTeX | **no - requires the .tex source** |",
        "| PDF compiles | not attempted in this run |",
        "",
        "## G. Bibliography audit",
        "",
        "| Ref | What the draft claims | What the published record shows | Status |",
        "|---|---|---|---|",
    ]
    for finding in REFERENCE_FINDINGS:
        lines.append(
            f"| {finding['ref']} | {finding['in_draft']} | {finding['actual']} | "
            f"**{finding['status']}** |"
        )
    lines += ["", UNVERIFIED_REFS_NOTE, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 7. Experiment log
# --------------------------------------------------------------------------- #
def write_experiment_log(path, ctx: dict[str, Any]) -> None:
    lines = [
        "# Experiment Log",
        "",
        f"- Timestamp: {ctx['timestamp']}",
        f"- Run tag: `{ctx['tag']}`",
        f"- Data source: `{ctx['stats']['meta']['source_file']}`",
        f"- Rows used: {ctx['stats']['total_records']:,}",
        f"- Seed: {config.SEED}",
        f"- Test size: {config.TEST_SIZE}",
        f"- Quick mode: {ctx.get('quick', False)}",
        f"- Stop words from: {ctx.get('stopword_source', 'n/a')}",
        f"- Random Forest trees: {ctx.get('rf_estimators', 'n/a')}",
        f"- CV folds: {ctx.get('cv_folds', 'n/a')}",
        f"- Total runtime: {ctx.get('runtime_seconds', 0):.1f} s",
        "",
        "## Steps executed",
        "",
    ]
    for step in ctx.get("steps", []):
        lines.append(f"- {step}")
    lines += [
        "",
        "## Outputs",
        "",
    ]
    for name in sorted(ctx.get("outputs", [])):
        lines.append(f"- `{name}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
