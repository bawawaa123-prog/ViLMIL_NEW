# Final External Validation Summary

## Paper-ready comparison

| Setting | AUC | ACC | F1 | Note |
|---|---:|---:|---:|---|
| Single-fold mean | 0.881284 | 0.748256 | 0.640074 | Mean over 5 folds |
| 5-fold ensemble @ 0.5 | 0.924729 | 0.813953 | 0.515152 | Probability average, default cutoff |
| 5-fold ensemble @ best F1 threshold | 0.924729 | 0.886628 | 0.606061 | Threshold=0.759977 |

## Fold threshold stability

| Fold | AUC | ACC @ 0.5 | F1 @ 0.5 | Best thr | Best ACC | Best F1 | Positive count @ best | Threshold std | Threshold IQR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.919095 | 0.793605 | 0.481752 | 0.994495 | 0.880814 | 0.568421 | 58 | 0.420344 | 0.757900 |
| 1 | 0.931244 | 0.610465 | 0.355769 | 0.975363 | 0.906977 | 0.609756 | 45 | 0.402101 | 0.903004 |
| 2 | 0.902720 | 0.889535 | 0.577778 | 0.702777 | 0.898256 | 0.578313 | 46 | 0.325751 | 0.035552 |
| 3 | 0.751871 | 0.636628 | 0.309392 | 0.955245 | 0.787791 | 0.396694 | 84 | 0.418588 | 0.929390 |
| 4 | 0.901488 | 0.811047 | 0.511278 | 0.836012 | 0.848837 | 0.527273 | 73 | 0.401773 | 0.679190 |

## Interpretation

Folds 1 and 3 are the main sources of instability. Fold 1 has unusually low ACC/F1 at the default threshold despite a high AUC, which points to calibration/threshold mismatch rather than poor ranking. Fold 3 has the lowest AUC by far and also the weakest best-threshold F1, so it is the clearest fold-level degradation. Fold 0 is also weaker than folds 2 and 4, but less severe than folds 1 and 3.

## Files

- [paper_ready_comparison_table.csv](paper_ready_comparison_table.csv)
- [fold_threshold_stability_metrics.csv](fold_threshold_stability_metrics.csv)
- [ensemble_5fold_avg_threshold_best.csv](ensemble_5fold_avg_threshold_best.csv)
- [ensemble_5fold_avg_threshold_scan.csv](ensemble_5fold_avg_threshold_scan.csv)
- [summary.csv](summary.csv)
- [result.csv](result.csv)
