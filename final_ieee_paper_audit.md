# Final Review-Style Assessment

Generated: 2026-11-17T20:20:26  Run tag: `smoke`

## A. What was already correct

- The problem framing (TF-IDF + classical classifiers on a labelled public
  corpus) is sound and reproducible in principle.
- The choice of body text as input and a binary legitimate/phishing target is
  consistent with the stated goal.
- The draft correctly identifies that a false negative is more costly than a
  false positive, which is the right security framing.
- The pipeline diagram matches the implemented architecture.
- The dataset's own schema does use `body` and `label`, so the original column choice was right.

## B. What was defective

1. **Data leakage.** TF-IDF was fitted on the full corpus before splitting.
2. **Irreproducible split.** No seed and no stratification.
3. **Incomplete metrics.** Logistic Regression had accuracy only; no ROC/AUC,
   no FPR/FNR, no per-class breakdown.
4. **No baseline.** The 56% figure in the discussion was neither computed nor
   attributed to a classifier.
5. **Internally inconsistent draft numbers.** 4 of 6 arithmetic checks failed (see `reports/initial_research_audit.md`).
6. **Unverified references.** The bibliography could not be substantiated in
   this audit; every entry needs a resolvable DOI before submission.
7. **Unsupported deployment language.** Efficiency claims were made without
   any timing measurement.

## C. What was fixed

- All fitted transforms moved inside one `Pipeline`; the split happens on raw
  text, stratified and seeded.
- Complete metric set computed from one shared code path for every model,
  including the majority-class baseline.
- ROC/AUC, confusion matrices, FPR/FNR and an operating-point sweep added.
- Stratified 5-fold CV with per-fold vectorizer fitting added.
- Feature-importance extraction with an explicit dimension assertion.
- Ablation over preprocessing and TF-IDF settings on a fixed split.
- Error analysis restricted to measured properties with masked excerpts.
- Inference cost measured rather than asserted.

## D. Actual results from this run

- Random Forest: accuracy 62.22%, precision 63.10%, recall 58.89%, F1 60.92%, ROC-AUC 67.15%.
- Logistic Regression: accuracy 70.56%, precision 72.29%, recall 66.67%, F1 69.36%, ROC-AUC 69.95%.
- Majority-class baseline: accuracy 50.00%.

Cross-validation means (separate experiment):

- Random Forest: accuracy 61.53% (+/- 3.07 pp), ROC-AUC 64.69% (+/- 1.74 pp) over 3 folds.
- Logistic Regression: accuracy 68.75% (+/- 2.73 pp), ROC-AUC 68.40% (+/- 5.22 pp) over 3 folds.

## E. Remaining limitations

See `reports/limitations_and_future_work.md`. The binding ones for review are
single-dataset evaluation, a random (not temporal) split, no adversarial
testing, and no significance test between the two classifiers.

## F. Reviewer-style verdict

The work is now technically correct and reproducible, which it was not
before. It is still a straightforward classical-ML comparison on one benchmark.
As a conference submission it is defensible only if the paper states exactly
that, reports the corrected numbers, and drops any novelty or deployment claim
the measurements do not support. The main remaining risk is not the modelling:
it is the bibliography, which must be verified entry by entry before the paper
is submitted anywhere.

### Checklist

| Item | Status |
|---|---|
| Scripts execute end to end | see `results/run_summary.json` and `reports/experiment_log.md` |
| Every figure generated from measured values | yes - no value is hard-coded in the plotting code |
| Tables verified against raw outputs | `results/*.csv` are the raw outputs |
| Draft numbers reconciled | no - the draft's own numbers disagree; they must be replaced |
| References verified | **no - see section G; 2 confirmed misattributions, 1 unverified** |
| Author block corrected in LaTeX | **no - requires the .tex source** |
| PDF compiles | not attempted in this run |

## G. Bibliography audit

| Ref | What the draft claims | What the published record shows | Status |
|---|---|---|---|
| [6] | Alhogail & Alsabih (2023), 'Social engineering tactic detection in phishing emails using SVM and psychological feature annotation', Computers & Security 130, 103258; Table I lists 'SVM, psych. feat.' at 96.1% | Alhogail, A., & Alsabih, A. (2021). Applying machine learning and natural language processing to detect phishing email. Computers & Security, 110, 102414. doi:10.1016/j.cose.2021.102414. Method: graph convolutional network + NLP over body text; reported accuracy 98.2% with FPR 0.015 on the Radev/Nazario-style corpus. | **MISATTRIBUTED** |
| [14] | Basit, Zafar, Liu & Javed (2024), 'A comprehensive taxonomy of phishing email detection: NLP, URL, and header-based approaches', ACM Computing Surveys 57(3), 1-35 | Basit, A., Zafar, M., Liu, X., et al. (2021). A comprehensive survey of AI-enabled phishing attacks detection techniques. Telecommunication Systems, 76, 139-154. doi:10.1007/s11235-020-00733-2 | **MISATTRIBUTED** |
| [8] | Maneriker, Reddy & Bou (2024), 'Character-level neural networks for adversarially obfuscated phishing email detection', IEEE DSC, 1-8, 93.8% | No matching publication found in this audit. Treat as unverified and replace. | **UNVERIFIED** |

References [1]-[5], [7], [9]-[13] and [15] were not checked one by one in this audit. Before submission, every entry needs its DOI resolved and the title, venue, year and reported metric confirmed against the publisher record. Entries that cannot be confirmed must be removed, not repaired from memory.
