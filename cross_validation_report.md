# Stratified Cross-Validation Report

Generated: 2026-11-17T20:20:26  Run tag: `smoke`

3-fold stratified cross-validation, `shuffle=True`, `random_state=42`, stratified k-fold.
The TF-IDF vectorizer is re-fitted inside every training fold, so no fold's
validation documents influence the vocabulary or the IDF weights.

**These numbers are a separate experiment from the holdout results.** They
estimate variance across folds; they are not the performance of the model saved
from the holdout run, and the two must not be mixed in one table.

| Model | Metric | Mean | Std. dev. |
|---|---|---|---|
| Random Forest | accuracy | 61.53% | 3.07 pp |
| Random Forest | precision | 61.10% | 3.83 pp |
| Random Forest | recall | 64.44% | 5.91 pp |
| Random Forest | f1 | 62.57% | 2.95 pp |
| Random Forest | roc_auc | 64.69% | 1.74 pp |
| Logistic Regression | accuracy | 68.75% | 2.73 pp |
| Logistic Regression | precision | 68.43% | 4.09 pp |
| Logistic Regression | recall | 70.28% | 2.93 pp |
| Logistic Regression | f1 | 69.25% | 1.69 pp |
| Logistic Regression | roc_auc | 68.40% | 5.22 pp |

Mean fit time per fold (seconds): Random Forest 0.1s, Logistic Regression 0.0s

Per-fold values are in `results/cross_validation_results.csv`.
