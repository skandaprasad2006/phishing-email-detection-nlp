# Limitations and Future Work

Generated: 2026-11-17T20:20:26  Run tag: `smoke`

## Limitations supported by this run

1. **One dataset.** All results come from `<synthetic>` (900 rows). Nothing here demonstrates performance on another corpus, a different time period, or live mail. Cross-dataset
   generalisation is untested, so no claim about it may be made.
2. **Benchmark conditions.** The split is random within one corpus, so
   near-duplicate campaigns can appear on both sides of the split. This inflates
   scores relative to a temporal split, which was not run.
3. **No adversarial evaluation.** No obfuscated, paraphrased or deliberately
   evasive samples were tested. Robustness to an adaptive attacker is unknown.
4. **Body text only.** Sender, header, subject and URL structure are excluded,
   although the dataset provides some of them.
5. **Bag-of-words representation.** TF-IDF with a linear or tree model cannot
   represent word order, negation or long-range context.
6. **No significance testing between models.** The Random Forest vs Logistic
   Regression comparison is descriptive. A paired test over folds would be
   needed before calling the difference real, and it was not performed.
7. **Static threshold.** Models are evaluated at the default 0.5 threshold; the
   operating-point table is reported for information, not tuned on test data.
8. **Cost figures are single-machine.** Latency and size were measured on the
   CPU of the machine that ran the experiment.

## Honest framing of the contribution

This is a comparative evaluation of two classical classifiers over one TF-IDF
representation of one public phishing corpus, with a leakage-free protocol and
a majority-class baseline. TF-IDF plus Random Forest for phishing detection is
well established; the contribution is the corrected, reproducible measurement
and the quantified effect of the original leakage, not a new algorithm.

## Future work that would be defensible

- Repeat on a second, temporally disjoint corpus (for example a later CEAS or
  Nazario/Nazario-style archive) and report the drop.
- Replace the random split with a time-ordered split and compare.
- Test an obfuscation suite (homoglyphs, homophone substitution, HTML hiding)
  and report degradation per transformation.
- Add header and URL features as a separate ablation arm.
- Compare against one contextual-embedding baseline under the same protocol.
