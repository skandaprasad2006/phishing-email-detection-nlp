# Master prompt for Freebuff

Paste this into a new Freebuff session in this folder. It assumes the code in
`src/`, the notebook in `notebooks/` and `HANDOVER.md` already exist.

---

## MASTER TASK: IEEE research paper audit, experimental finalisation and LaTeX rewrite

**Project:** AI-Based Phishing Email Detection Using Natural Language Processing
**Role:** senior machine-learning researcher, IEEE conference reviewer, Python engineer, reproducibility auditor.

### Context you must not re-derive

A leakage-free pipeline already exists in `src/` and has been executed end to end.
It produces every result the paper needs. Read `HANDOVER.md` first, then
`README.md`, then `src/run_experiments.py`.

Facts already established (do not "discover" them again, verify them if cheap):

- The original implementation was five scripts (`code1.py`-`code5.py`) that fitted
  TF-IDF on the full corpus **before** the split. That leak is fixed by using one
  `sklearn.pipeline.Pipeline` per model.
- The draft's printed numbers are internally inconsistent: 4 of 6 arithmetic checks
  fail. They are listed in `reports/initial_research_audit.md`.
- The draft claims ~18,000 emails but reports a 7,831-row test partition, which
  implies ~39,155 rows.
- At least two references are misattributed real papers; three findings are in
  `reports/final_ieee_paper_audit.md` section G.

### Step 0 - locate the inputs

Ask for, or find: the dataset (`data/CEAS_08.csv` or `.zip`) and the paper source
(`.tex` plus the compiled PDF, and the draw.io-exported `pipeline.png`). If the
`.tex` is absent, do everything else and stop before the LaTeX step; do not invent
a paper from scratch.

### Step 1 - run the experiments (no paper editing yet)

```bash
py -m pip install -r requirements.txt
py -m src.run_experiments --data data --tag main
```

Confirm these exist and are non-trivial: `results/model_metrics.csv`,
`results/roc_curve.png`, `results/confusion_matrix_rf.png`,
`results/confusion_matrix_lr.png`, `results/feature_importance.png`,
`results/cross_validation_summary.csv`, `results/ablation_results.csv`,
`results/deployment_metrics.csv`, `results/run_summary.json`,
`reports/dataset_statistics.md`, `reports/experimental_results.md`,
`reports/cross_validation_report.md`, `reports/error_analysis.md`,
`reports/limitations_and_future_work.md`, `reports/final_ieee_paper_audit.md`.

If the run fails, fix the code and re-run. Do not hand-edit CSVs.

### Step 2 - build a numbers ledger

Create `paper/numbers_ledger.md`: one row per number that will appear in the paper
(value, exact source file, and the section where it will be used). Every subsequent
paper claim must cite a ledger row. This is the mechanism that prevents fabricated
values from creeping back in.

### Step 3 - rewrite the paper

Preserve: title, the six authors and their order (Sachin Sebastian P, Abinav Biju
Lathika, Manjunath N, Reetom Gogoi, Dr. M. Lakshmanan, Mr. Verriboina Antony),
affiliations, emails, the existing draw.io methodology figure, and any reference
that survives verification.

Rewrite with measured values only, and cut what cannot be supported:

- **Abstract:** state the dataset size measured from the CSV, the protocol
  (stratified holdout + 5-fold CV), and the actual accuracy/precision/recall/F1/AUC.
  No novelty claim.
- **Introduction:** keep the motivation; remove any sentence implying a new method.
- **Literature survey:** only verified references. Where a prior result used a
  different dataset or protocol, say so in the same sentence. Delete the table row
  if the metric cannot be confirmed.
- **Method:** describe the leakage-free protocol explicitly - raw text split first,
  TF-IDF fitted on training text only, both classifiers inside one Pipeline.
  Mention that cleaning is stateless and per-document.
- **Experimental setup:** measured dataset statistics, class distribution, vocabulary
  size, hyperparameters as actually used (`n_estimators=100`, `random_state=42`,
  `max_iter=1000`, `test_size=0.2`, stratified, 5-fold CV).
- **Results:** holdout table, confusion matrices with FPR/FNR, ROC-AUC, baseline,
  then - clearly separated - the cross-validation table, the ablation table, and the
  cost measurements. Each figure must be referenced and discussed in the text.
- **Discussion:** what the numbers mean, including the effect of removing the leak
  (`results/legacy_protocol_comparison.csv`).
- **Limitations:** single dataset, random (not temporal) split, no adversarial
  testing, body text only, no significance test between models. Copy from
  `reports/limitations_and_future_work.md`.
- **Conclusion:** no deployment claim beyond "CPU inference costs milliseconds per
  message on one machine".

Author block: six authors in the order above, using standard IEEE
`\IEEEauthorblockN{}`/`\IEEEauthorblockA{}` with `\and` - no `\hspace` hacks.

### Step 4 - verify before declaring done

1. Every number in the `.tex` appears in `paper/numbers_ledger.md`.
2. Every ledger row points at a file that exists.
3. No sentence claims cross-dataset generalisation, statistical significance, or
   novelty of TF-IDF + Random Forest.
4. Figure and table numbering is sequential and every one is cited in the text.
5. All references have a resolvable DOI and a verified metric.
6. `pdflatex` + `bibtex` compile with no undefined references.
7. Page count is within the conference limit without padding.

### Step 5 - final report

Write `reports/submission_checklist.md` with: what changed, the numbers used and
their sources, what could not be verified, and a reviewer-style verdict covering
technical correctness, experimental rigour, reproducibility, writing quality and
contribution. Do not give a positive verdict merely because the checklist is green.

### Non-negotiable rules

1. Never fabricate a result, a dataset statistic or a citation.
2. Never reintroduce leakage: no `fit_transform` on data that includes test rows.
3. Never present CV numbers as holdout numbers.
4. Never delete working code to make a number look better.
5. Never change the research scope without stating why.
6. If something cannot be measured, say so in the paper.
7. Do not touch files unrelated to the paper or the experiments.
