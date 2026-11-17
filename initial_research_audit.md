# Initial Research Audit

Generated: 2026-11-17T20:20:26  
Run tag: `smoke`  
Data source: `<synthetic>`

This audit was produced by inspecting the upstream repository
(`Ripebrusser/AI-Based-Phishing-Email-Detection-Using-Natural-Language-Processing`,
five scripts `code1.py`-`code5.py`) and the draft paper, then measuring the
dataset directly. Statements about the *old code* are observations of that code.
Statements about the *dataset* are computed below. Nothing here is estimated.

## 1. What the original implementation actually did

| Step | Original code | Defect |
|---|---|---|
| Load | `pd.read_csv('CEAS_08.csv')` after zip extraction | Fine, but the file was never schema-checked |
| Input text | `df['body'].astype(str)` | Fine |
| Features | `TfidfVectorizer(stop_words='english').fit_transform(X)` on **all** rows | **Critical: data leakage.** Vocabulary and IDF weights were learned from the test rows |
| Split | `train_test_split(X, y, test_size=0.2)` | No `random_state` (irreproducible), no `stratify` (class balance not preserved) |
| Random Forest | `RandomForestClassifier()` defaults | No `random_state`; results cannot be reproduced |
| Logistic Regression | `LogisticRegression(max_iter=1000)` | No `random_state` |
| Metrics | accuracy only; `precision_score/recall_score/f1_score` on the RF predictions | No LR precision/recall/F1, no ROC/AUC, no per-class breakdown for LR |
| Confusion matrix | `plt.imshow(cm)` with default colormap | No axis labels or class names; not publication quality |
| Baseline | none | No majority-class reference |
| Validation | single holdout split | No cross-validation, no variance estimate |

The leakage is the most serious defect: because the vectorizer saw the test
documents, the reported scores are optimistic and the size of that optimism
was never quantified. This run quantifies it (section 8).

## 2. Dataset actually measured

- Rows in file: **900**
- Rows used: **900**
- Body column resolved to: `body`
- Label column resolved to: `label`
- Label encoding: numeric 0/1 as stored in file
- Label mapping applied: `{'1': 1, '0': 0}`
- Missing bodies: 0
- Exact duplicate bodies: 40

| Label | Class | Count | Share |
|---|---|---|---|
| 0 | legitimate | 450 | 50.00% |
| 1 | phishing | 450 | 50.00% |

Text size: mean body length 146 characters (22 whitespace tokens); legitimate 144 vs phishing 149 characters.

## 3. Self-consistency of the numbers already printed in the draft

| Check | Value in draft | Recomputed from draft | Difference | Status |
|---|---|---|---|---|
| Test partition equals legitimate + phishing rows | 7,831 | 7,831 | 0 | **PASS** |
| Accuracy printed in Table II vs accuracy implied by the confusion-matrix counts | 0.9927 | 0.9955 | 0.0028 | **FAIL** |
| Precision printed in Table III vs precision implied by the 20 false positives | 0.9954 | 0.9954 | 0.0000 | **PASS** |
| Recall printed in Table III vs recall implied by the 15 false negatives | 0.9915 | 0.9966 | 0.0051 | **FAIL** |
| Majority-class baseline printed in the discussion vs actual phishing share of the test set | 0.5600 | 0.5587 | 0.0013 | **FAIL** |
| Total dataset size printed in Table I/IV vs total implied by a 20% test partition | 18,000 | 39,155 | 21,155 | **FAIL** |

**4 of 6 internal checks fail.** The draft's tables
cannot all be true at once, so no combination of them may be reused. The most
consequential failures:

- *Accuracy printed in Table II vs accuracy implied by the confusion-matrix counts* - draft says 0.9927, its own counts imply 0.9955.
- *Recall printed in Table III vs recall implied by the 15 false negatives* - draft says 0.9915, its own counts imply 0.9966.
- *Majority-class baseline printed in the discussion vs actual phishing share of the test set* - draft says 0.5600, its own counts imply 0.5587.
- *Total dataset size printed in Table I/IV vs total implied by a 20% test partition* - draft says 18,000, its own counts imply 39,155.

The dataset-size discrepancy also means the abstract's "approximately 18,000
labelled messages" is not consistent with the 7,831-row test partition printed
in Section V. Only one of these can be right, and the file itself decides which.

## 4. Missing experiments (state before this run)

| Missing item | Status after this run |
|---|---|
| ROC curve and AUC | Generated: `results/roc_curve.png`, `results/model_metrics.csv` |
| Complete LR metrics | Generated for both models in `results/model_metrics.csv` |
| Formal majority-class baseline | Generated (`dummy_majority` row) |
| Dataset statistics | `reports/dataset_statistics.md` |
| Stratified 5-fold cross-validation | `reports/cross_validation_report.md` |
| Feature analysis | `results/feature_importance.png` |
| Error analysis | `reports/error_analysis.md` |
| Ablation study | `results/ablation_results.csv` |
| Inference cost measurements | `results/deployment_metrics.csv` |
| Single-dataset generalisation limit | `reports/limitations_and_future_work.md` |

## 5. Corrections applied to the implementation

1. All learned transforms (TF-IDF vocabulary, IDF weights, classifier
   parameters) now live inside one `sklearn.pipeline.Pipeline`, so they are
   fitted on training data only.
2. The split is stratified and seeded (`random_state=42`), and every estimator
   that accepts a seed receives it.
3. The positive class probability is read from `clf.classes_` instead of
   assuming `predict_proba[:, 1]` is phishing.
4. Every model now reports accuracy, precision, recall, F1, ROC-AUC, the full
   confusion matrix, FPR and FNR from the same code path.
5. Cross-validation re-fits the vectorizer inside each fold.
6. Text cleaning is stateless and per-document, so it cannot leak; this is
   documented rather than assumed.

## 6. Paper-level inconsistencies to fix in the LaTeX source

- Section IV states the dataset contains "approximately 18,000" messages while
  Section V reports a 7,831-row test set; a 20% test set implies about 39,000 rows.
  Replace both with the measured row count above.
- Section IV calls the split "stratified" although the code used no
  `stratify=` argument. It is stratified now; the sentence was false before.
- Table III's per-class supports (3,456 / 4,375) must be recomputed from the new
  split; they cannot be inherited.
- The discussion compares against "approximately 56%" without stating that it is
  the predicted-majority rate of this specific test split; report it as a measured
  DummyClassifier result instead.
- The reference list must be verified entry by entry before submission (see
  `reports/final_ieee_paper_audit.md`). Several entries are not traceable in this
  audit and must be replaced with sources whose DOI resolves.
