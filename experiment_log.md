# Experiment Log

- Timestamp: 2026-11-17T20:20:26
- Run tag: `smoke`
- Data source: `<synthetic>`
- Rows used: 900
- Seed: 42
- Test size: 0.2
- Quick mode: True
- Stop words from: scikit-learn ENGLISH_STOP_WORDS (NLTK unavailable)
- Random Forest trees: 25
- CV folds: 3
- Total runtime: 5.4 s

## Steps executed

- loading synthetic data (900 rows) - code test only
- schema verified: body='body', label='label', rows used=900
- split: train=720 test=180 (phishing in test=90)
- Random Forest: acc=0.6222 auc=0.6715 features=112
- Logistic Regression: acc=0.7056 auc=0.6995 features=112
- baseline accuracy=0.5000
- dataset statistics written (900 records)
- figures: roc_curve.png, confusion_matrix_rf.png, confusion_matrix_lr.png
- feature importance: top term 'urltoken'
- CV Random Forest: acc=0.6153+/-0.0307 auc=0.6469
- CV Logistic Regression: acc=0.6875+/-0.0273 auc=0.6840
- ablation: 7 configurations
- error analysis written
- inference cost measured
- legacy leaky protocol accuracy=0.6444

## Outputs

- `reports\_smoke\dataset_statistics.md`
- `results\_smoke\ablation_results.csv`
- `results\_smoke\class_distribution.png`
- `results\_smoke\confusion_matrix_lr.png`
- `results\_smoke\confusion_matrix_rf.png`
- `results\_smoke\cross_validation_results.csv`
- `results\_smoke\cross_validation_summary.csv`
- `results\_smoke\dataset_statistics.json`
- `results\_smoke\deployment_metrics.csv`
- `results\_smoke\error_samples_lr.csv`
- `results\_smoke\error_samples_rf.csv`
- `results\_smoke\feature_importance.csv`
- `results\_smoke\feature_importance.png`
- `results\_smoke\legacy_protocol_comparison.csv`
- `results\_smoke\model_metrics.csv`
- `results\_smoke\roc_curve.png`
- `results\_smoke\threshold_analysis_lr.csv`
- `results\_smoke\threshold_analysis_rf.csv`
