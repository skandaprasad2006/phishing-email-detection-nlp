# Experimental Results

Generated: 2026-11-17T20:20:26  Run tag: `smoke`  
Protocol: stratified 80:20 train/test holdout split, `random_state=42`; TF-IDF fitted on training texts only.

## 1. Holdout performance

| model | accuracy | precision | recall | f1 | roc_auc | true_positive | true_negative | false_positive | false_negative | false_positive_rate | false_negative_rate | specificity | negative_predictive_value | support_phishing | support_legitimate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 62.22% | 63.10% | 58.89% | 60.92% | 67.15% | 53 | 59 | 31 | 37 | 34.44% | 41.11% | 65.56% | 61.46% | 90 | 90 |
| Logistic Regression | 70.56% | 72.29% | 66.67% | 69.36% | 69.95% | 60 | 67 | 23 | 30 | 25.56% | 33.33% | 74.44% | 69.07% | 90 | 90 |
| Majority-class baseline | 50.00% | 0.00% | 0.00% | 0.00% | n/a | 0 | 90 | 0 | 90 | 0.00% | 100.00% | 100.00% | 50.00% | 90 | 90 |

Confusion-matrix counting convention: rows are true labels, columns are
predicted labels, positive class is phishing (label 1).

## 2. Confusion matrices and derived error rates

| Model | TP | TN | FP | FN | FPR = FP/(FP+TN) | FNR = FN/(FN+TP) |
|---|---|---|---|---|---|---|
| Random Forest | 53 | 59 | 31 | 37 | 34.44% | 41.11% |
| Logistic Regression | 60 | 67 | 23 | 30 | 25.56% | 33.33% |
| Majority-class baseline | 0 | 90 | 0 | 90 | 0.00% | 100.00% |

Figures: `results/confusion_matrix_rf.png`, `results/confusion_matrix_lr.png`.

## 3. ROC and AUC

| Model | ROC-AUC |
|---|---|
| Random Forest | 0.6715 |
| Logistic Regression | 0.6995 |
| Majority-class baseline | n/a |

Figure: `results/roc_curve.png` (all models on one panel with the chance line).

## 4. Operating-point analysis

The security trade-off is a threshold choice, not only a model choice:
lowering the threshold catches more phishing at the cost of more false alarms.

### Random Forest

| threshold | precision | recall | f1 | false_positive | false_negative | false_positive_rate | false_negative_rate |
|---|---|---|---|---|---|---|---|
| 0.050 | 52.05% | 98.89% | 68.20% | 82 | 1 | 91.11% | 1.11% |
| 0.100 | 52.15% | 94.44% | 67.19% | 78 | 5 | 86.67% | 5.56% |
| 0.150 | 52.53% | 92.22% | 66.94% | 75 | 7 | 83.33% | 7.78% |
| 0.200 | 53.64% | 90.00% | 67.22% | 70 | 9 | 77.78% | 10.00% |
| 0.250 | 56.62% | 85.56% | 68.14% | 59 | 13 | 65.56% | 14.44% |
| 0.300 | 56.91% | 77.78% | 65.73% | 53 | 20 | 58.89% | 22.22% |
| 0.350 | 57.52% | 72.22% | 64.04% | 48 | 25 | 53.33% | 27.78% |
| 0.400 | 60.00% | 70.00% | 64.62% | 42 | 27 | 46.67% | 30.00% |
| 0.450 | 63.64% | 62.22% | 62.92% | 32 | 34 | 35.56% | 37.78% |
| 0.500 | 63.10% | 58.89% | 60.92% | 31 | 37 | 34.44% | 41.11% |
| 0.550 | 65.33% | 54.44% | 59.39% | 26 | 41 | 28.89% | 45.56% |
| 0.600 | 66.15% | 47.78% | 55.48% | 22 | 47 | 24.44% | 52.22% |
| 0.650 | 68.63% | 38.89% | 49.65% | 16 | 55 | 17.78% | 61.11% |
| 0.700 | 73.81% | 34.44% | 46.97% | 11 | 59 | 12.22% | 65.56% |
| 0.750 | 80.00% | 31.11% | 44.80% | 7 | 62 | 7.78% | 68.89% |
| 0.800 | 76.00% | 21.11% | 33.04% | 6 | 71 | 6.67% | 78.89% |
| 0.850 | 75.00% | 13.33% | 22.64% | 4 | 78 | 4.44% | 86.67% |
| 0.900 | 90.91% | 11.11% | 19.80% | 1 | 80 | 1.11% | 88.89% |
| 0.950 | 85.71% | 6.67% | 12.37% | 1 | 84 | 1.11% | 93.33% |

### Logistic Regression

| threshold | precision | recall | f1 | false_positive | false_negative | false_positive_rate | false_negative_rate |
|---|---|---|---|---|---|---|---|
| 0.050 | 50.00% | 100.00% | 66.67% | 90 | 0 | 100.00% | 0.00% |
| 0.100 | 50.00% | 100.00% | 66.67% | 90 | 0 | 100.00% | 0.00% |
| 0.150 | 50.00% | 100.00% | 66.67% | 90 | 0 | 100.00% | 0.00% |
| 0.200 | 49.72% | 98.89% | 66.17% | 90 | 1 | 100.00% | 1.11% |
| 0.250 | 50.90% | 94.44% | 66.15% | 82 | 5 | 91.11% | 5.56% |
| 0.300 | 52.32% | 87.78% | 65.56% | 72 | 11 | 80.00% | 12.22% |
| 0.350 | 58.56% | 72.22% | 64.68% | 46 | 25 | 51.11% | 27.78% |
| 0.400 | 67.02% | 70.00% | 68.48% | 31 | 27 | 34.44% | 30.00% |
| 0.450 | 68.89% | 68.89% | 68.89% | 28 | 28 | 31.11% | 31.11% |
| 0.500 | 72.29% | 66.67% | 69.36% | 23 | 30 | 25.56% | 33.33% |
| 0.550 | 74.07% | 66.67% | 70.18% | 21 | 30 | 23.33% | 33.33% |
| 0.600 | 75.00% | 63.33% | 68.67% | 19 | 33 | 21.11% | 36.67% |
| 0.650 | 78.43% | 44.44% | 56.74% | 11 | 50 | 12.22% | 55.56% |
| 0.700 | 75.00% | 23.33% | 35.59% | 7 | 69 | 7.78% | 76.67% |
| 0.750 | 75.00% | 10.00% | 17.65% | 3 | 81 | 3.33% | 90.00% |
| 0.800 | 0.00% | 0.00% | 0.00% | 0 | 90 | 0.00% | 100.00% |
| 0.850 | 0.00% | 0.00% | 0.00% | 0 | 90 | 0.00% | 100.00% |
| 0.900 | 0.00% | 0.00% | 0.00% | 0 | 90 | 0.00% | 100.00% |
| 0.950 | 0.00% | 0.00% | 0.00% | 0 | 90 | 0.00% | 100.00% |

## 5. Feature importance (Random Forest, top terms)

These are impurity-based importance scores from the fitted forest. They
indicate which TF-IDF dimensions the model used most; they are not causal
evidence that a term makes an email phishing.

| Rank | Feature | Importance |
|---|---|---|
| 1 | `urltoken` | 0.09634 |
| 2 | `today` | 0.03698 |
| 3 | `meeting` | 0.03670 |
| 4 | `schedule` | 0.03492 |
| 5 | `team` | 0.03321 |
| 6 | `review` | 0.03267 |
| 7 | `details` | 0.03197 |
| 8 | `week` | 0.03025 |
| 9 | `report` | 0.02915 |
| 10 | `note` | 0.02900 |
| 11 | `thanks` | 0.02811 |
| 12 | `information` | 0.02785 |
| 13 | `updated` | 0.02735 |
| 14 | `regards` | 0.02726 |
| 15 | `let` | 0.01976 |
| 16 | `know` | 0.01266 |
| 17 | `claim` | 0.00969 |
| 18 | `attached` | 0.00940 |
| 19 | `confirm` | 0.00935 |
| 20 | `account` | 0.00903 |

Figure: `results/feature_importance.png`.

## 6. Ablation study

One factor changed at a time, same split and seed, Random Forest. `features` is the TF-IDF dimensionality that configuration produced.

| Variant | Features | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| A0 no text preprocessing (raw body) | 115 | 63.33% | 65.38% | 56.67% | 60.71% | 66.72% | 27 | 39 | 0.1 |
| A1 default: unigrams, stop words, min_df=1 | 112 | 62.22% | 63.10% | 58.89% | 60.92% | 67.15% | 31 | 37 | 0.1 |
| A2 no stop-word removal | 152 | 63.33% | 66.22% | 54.44% | 59.76% | 66.35% | 25 | 41 | 0.1 |
| A3 bigrams (1,2), min_df=2 | 515 | 64.44% | 66.67% | 57.78% | 61.90% | 68.70% | 26 | 38 | 0.1 |
| A4 min_df=2 | 112 | 62.22% | 63.10% | 58.89% | 60.92% | 67.15% | 31 | 37 | 0.1 |
| A5 max_features=5,000 | 112 | 62.22% | 63.10% | 58.89% | 60.92% | 67.15% | 31 | 37 | 0.1 |
| A6 sublinear_tf=True | 112 | 62.78% | 63.86% | 58.89% | 61.27% | 64.52% | 30 | 37 | 0.1 |

Read the deltas as descriptive of this dataset and this split; no
significance test was performed on these differences.

## 7. Effect of the original leakage

The original protocol (TF-IDF fitted on the whole dataset before an
unstratified, unseeded split) was reproduced verbatim so the leak can be
quantified rather than asserted.

| Protocol | Model | Accuracy | TF-IDF fitted on |
|---|---|---|---|
| Legacy (leaky) | Random Forest | 64.44% | all rows, including test |
| Corrected (this run) | Random Forest | 62.22% | training rows only |

Difference: +2.22 percentage points (legacy minus corrected). The two protocols use different splits, so this
is an indicative comparison of protocols, not a paired test on identical rows.

## 8. Inference cost

Measured on the test partition inside this run; single-email latency was
averaged over individually scored emails after a warm-up call.

| Model | Artifact size | Latency mean | Latency p95 | Batch throughput |
|---|---|---|---|---|
| Random Forest | 0.666 MB | 18.028 ms | 22.391 ms | 7,358.2 emails/s |
| Logistic Regression | 0.007 MB | 0.432 ms | 0.697 ms | 87,044.8 emails/s |

These are single-machine CPU measurements from one run, not a
production benchmark. They support a statement such as "classification adds
a few milliseconds of CPU time per message", and nothing stronger.
