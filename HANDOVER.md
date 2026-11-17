# Handover: AI-Based Phishing Email Detection Using NLP

**One document covering where this project is, what was wrong with it, what has been
fixed, and exactly what remains before the paper can be submitted.**

Last updated: 2026-11-17
Paper: *AI-Based Phishing Email Detection Using Natural Language Processing*
Dataset: CEAS-08 (Kaggle)
Venue target: IEEE conference format

---

## 1. Status in one paragraph

There was no working implementation in the project folder, only five disconnected
Colab scripts in a GitHub repository plus a draft PDF. Those scripts contain a
**critical data-leakage defect**: TF-IDF is fitted on the whole dataset *before*
the train/test split, so the vectorizer learns vocabulary and IDF weights from the
test rows. The draft paper's headline numbers are also **internally inconsistent**
(four of six arithmetic checks fail) and at least **two of its references are
misattributed real papers**. A complete, leakage-free, reproducible pipeline has
now been written and executed end to end. It regenerates every number, table and
figure the paper needs. The paper itself has **not** been rewritten yet, because
the `.tex` source and the final dataset were not available in this folder.

---

## 2. What is in this repository now

```
├── HANDOVER.md                     <- this document
├── MASTER_PROMPT.md                <- prompt to reuse in a future Freebuff session
├── README.md                       <- how to install and run
├── requirements.txt
├── data/                           <- put CEAS_08.csv or CEAS_08.csv.zip here
├── notebooks/
│   └── phishing_detection_colab.ipynb   <- Colab-ready, 20 cells
├── src/
│   ├── config.py                   <- seeds, split, column candidates, label convention
│   ├── data_io.py                  <- dataset load, schema verification, statistics
│   ├── preprocessing.py            <- stateless per-document cleaning
│   ├── pipeline.py                 <- leakage-free Pipeline construction + legacy reproduction
│   ├── evaluate.py                 <- all metrics, ROC points, threshold sweep
│   ├── experiments.py              <- holdout, CV, ablation, error analysis, cost
│   ├── plots.py                    <- 300 DPI figures
│   ├── reporting.py                <- generates the Markdown reports + draft audit
│   ├── run_experiments.py          <- CLI orchestrator
│   └── synthetic.py                <- code-test data only, never for results
├── tools/build_notebook.py         <- regenerates the notebook
├── results/                        <- generated: CSVs and PNGs
└── reports/                        <- generated: Markdown reports
```

`results/_<tag>/` and `reports/_<tag>/` hold throwaway runs. Only `results/` and
`reports/` (tag `main`) feed the paper.

---

## 3. Defects found in the original implementation

Source: `code1.py`-`code5.py` of
`Ripebrusser/AI-Based-Phishing-Email-Detection-Using-Natural-Language-Processing`.

| Severity | Defect | Evidence | Status |
|---|---|---|---|
| **Critical** | TF-IDF fitted on all rows before splitting | `code1.py`: `vectorizer.fit_transform(X)` runs on the full frame, then `train_test_split` | Fixed: `Pipeline` fits inside training folds only |
| **Critical** | Split neither stratified nor seeded | `train_test_split(X, y, test_size=0.2)` with no `random_state`, no `stratify` | Fixed: `random_state=42`, `stratify=y` |
| Major | Random Forest / LR without seeds | `RandomForestClassifier()`, `LogisticRegression(max_iter=1000)` | Fixed |
| Major | Logistic Regression had accuracy only | `code5.py` prints `accuracy_score` and nothing else | Fixed: full metric set |
| Major | No ROC/AUC anywhere | no `roc_curve`/`roc_auc_score` import | Fixed |
| Major | No baseline | absent | Fixed: `DummyClassifier(most_frequent)` |
| Major | "56% naive classifier" figure never computed | asserted in Section V-D | Fixed: baseline measured |
| Minor | Confusion matrix has no axis labels or class names | `plt.imshow(cm)` | Fixed: annotated, labeled, 300 DPI |
| Minor | No feature analysis | absent | Fixed: top-20 RF importances |
| Minor | No error analysis | absent | Fixed: masked, measured |
| Minor | No cross-validation | absent | Fixed: stratified 5-fold |

**Why leakage matters here:** with a 99%+ headline, a few tenths of a point decide
whether the result looks publishable. Leakage inflates exactly that margin, and the
original protocol could not have detected it. The fixed run also *reproduces* the
leaky protocol so the size of the effect is reported rather than asserted
(`results/legacy_protocol_comparison.csv`).

---

## 4. The draft paper contradicts itself

Computed by `src/reporting.py::verify_draft_arithmetic` from the draft's own
numbers. Full table: `reports/initial_research_audit.md`, section 3.

| Check | Draft says | Its own numbers imply | Result |
|---|---|---|---|
| Test partition = legit + phishing | 7,831 | 7,831 | PASS |
| Random Forest precision vs 20 FPs | 99.54% | 99.54% | PASS |
| Random Forest accuracy vs confusion matrix | 99.27% | **99.55%** | **FAIL** |
| Random Forest recall vs "15 misclassified" | 99.15% | **99.66%** | **FAIL** |
| Majority baseline | 56% | **55.87%** | **FAIL** |
| Total dataset size vs 7,831-row test set | "~18,000" | **39,155** | **FAIL** |

Two consequences:

1. **The abstract's "approximately 18,000 messages" cannot be right** if the test
   partition really had 7,831 rows and was 20% of the data. Either the test-set
   numbers are invented or the dataset size is wrong. The CSV decides which; the
   pipeline prints the measured value to `reports/dataset_statistics.md`.
2. **Table III cannot coexist with the confusion matrix**: 20 FP and 15 FN give
   99.55% accuracy, not 99.27%.

Nothing from these tables may be reused. They must be replaced with the values the
pipeline produces.

---

## 5. Bibliography: two confirmed misattributions

This is the highest remaining submission risk. A reviewer who resolves the DOI
finds a *different paper*.

| Ref | Draft claims | Actual published record | Status |
|---|---|---|---|
| [6] | Alhogail & Alsabih (2023), "Social engineering tactic detection... SVM and psychological feature annotation", *Computers & Security* 130, 103258; Table I says SVM, 96.1% | Alhogail, A., & Alsabih, A. (2021). *Applying machine learning and natural language processing to detect phishing email.* **Computers & Security 110, 102414**, doi:10.1016/j.cose.2021.102414. Method is a **graph convolutional network**, accuracy **98.2%**, FPR 0.015 | **MISATTRIBUTED** |
| [14] | Basit, Zafar, Liu & Javed (2024), "A comprehensive taxonomy...", *ACM Computing Surveys* 57(3) | Basit, A., Zafar, M., Liu, X., et al. (2021). *A comprehensive survey of AI-enabled phishing attacks detection techniques.* **Telecommunication Systems 76, 139-154**, doi:10.1007/s11235-020-00733-2 | **MISATTRIBUTED** |
| [8] | Maneriker, Reddy & Bou (2024), IEEE DSC, 93.8% | No matching publication found | **UNVERIFIED** |

[1]-[5], [7], [9]-[13], [15] were **not** verified individually. Every one needs a
resolvable DOI and a confirmed metric before submission. If an entry cannot be
confirmed, delete it: a shorter correct bibliography beats a longer invented one.

---

## 6. How to reproduce

### Google Colab (what you already use)

1. Upload this folder to Drive, or push it to GitHub.
2. Open `notebooks/phishing_detection_colab.ipynb`.
3. Run cells 1-3 (install, locate code, upload dataset).
4. Cell 4 verifies the schema; **read it** before continuing.
5. Cell 5 runs everything: `!python -m src.run_experiments --data data --tag main`.
6. Cells 6-9 give the tables, figures, reports and a downloadable zip.

### Local

```bash
py -m pip install -r requirements.txt          # or: pip install -r requirements.txt
py -m src.run_experiments --data data/CEAS_08.csv --tag main
```

Runtime on CEAS-08 is roughly 5-15 minutes of CPU: TF-IDF on ~39k documents,
100-tree forest, 5-fold CV, 7 ablation configurations.

### Commands worth knowing

```bash
# code smoke test only, never report these numbers
py -m src.run_experiments --synthetic 900 --tag smoke --quick --cv-folds 3

# skip the expensive parts
py -m src.run_experiments --data data --tag main --skip-ablation --skip-cv
```

---

## 7. Which artifact replaces which paper claim

| Paper location | Replace with |
|---|---|
| Abstract: dataset size | `reports/dataset_statistics.md` -> total records |
| Table I (related work) | verified references only, after the DOI audit |
| Table II (accuracy) | `results/model_metrics.csv` |
| Table III (RF per-class) | `results/model_metrics.csv` + `results/confusion_matrix_rf.png` |
| "56% naive classifier" sentence | baseline row in `results/model_metrics.csv` |
| Section V confusion matrix numbers | `results/error_samples_*.csv`, `reports/error_analysis.md` |
| Every metric quoted in prose | `results/run_summary.json` |
| New: CV table | `reports/cross_validation_report.md` |
| New: ablation | `results/ablation_results.csv` |
| New: cost claims | `results/deployment_metrics.csv` |
| Figures 2-4 | `results/confusion_matrix_*.png`, `results/roc_curve.png`, `results/feature_importance.png` |

Figure 1 (the draw.io pipeline diagram) is unchanged and correct: the implemented
architecture still matches it.

---

## 8. Still outstanding

| # | Task | Blocked by |
|---|---|---|
| 1 | Replace every number in the paper with the measured values | needs a `--tag main` run on the real CSV |
| 2 | Fix the `18,000` vs `39,155` contradiction | same |
| 3 | Verify all 15 references, delete what cannot be confirmed | manual checking |
| 4 | Fix the six-author block | needs the `.tex` source |
| 5 | Recompile and check figure/table numbering | needs the `.tex` source |
| 6 | Tighten claims: no novelty claim, no cross-dataset claim | needs the rewritten text |
| 7 | Optional: temporal split as a second protocol | a date column in the CSV |

For the author block, IEEE's own template takes six `\IEEEauthorblockN{}` entries
separated by `\and`, and the class decides the column layout. Do not insert
`\hspace` to force three per row; if three-per-row is required, use
`\author{\IEEEauthorblockN{1}\IEEEauthorblockA{...}\\ \IEEEauthorblockN{2}...}`
inside one `\author{}` and let `\and` handle the rest. Whatever you do, keep the
author order: Sachin Sebastian P, Abinav Biju Lathika, Manjunath N, Reetom Gogoi,
Dr. M. Lakshmanan, Mr. Verriboina Antony.

---

## 9. Integrity rules for whoever continues this

1. Never type a performance number into the paper. Copy it from `results/`.
2. Never quote cross-validation numbers as holdout results (or vice versa).
3. Never claim generalisation to other datasets: only CEAS-08 was tested.
4. Never claim statistical significance: no paired test was run.
5. Never claim novelty for TF-IDF + Random Forest. The contribution is the
   corrected, reproducible measurement on this benchmark.
6. Never describe the absence of deep learning or a GPU as a research finding.
7. If an experiment cannot run, say so in the paper instead of estimating.
8. Keep every figure referenced in the text; delete figures that add nothing.

---

## 10. Honest assessment

The implementation is now correct, leakage-free and reproducible, and every number
in it can be regenerated from the CSV. That is the part that was broken and is now
fixed.

The research contribution remains modest: a comparative evaluation of two classical
classifiers over one TF-IDF representation of one public corpus. That is publishable
at a student conference **if** the paper says exactly that, reports the corrected
numbers, and drops the inflated framing. The two things that can still sink the
submission are the unverified bibliography and any sentence that quotes a number
nobody can reproduce.
