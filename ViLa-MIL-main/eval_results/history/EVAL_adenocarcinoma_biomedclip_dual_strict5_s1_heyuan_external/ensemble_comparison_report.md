# Heyuan External Validation Comparison

## Metrics Overview

| Method | AUC | ACC | F1 | Notes |
|---|---:|---:|---:|---|
| Single-fold mean | 0.881284 | 0.748256 | 0.640074 | Mean over 5 folds from `result.csv` |
| 5-fold probability average @ 0.5 | 0.924729 | 0.813953 | 0.515152 | Averaging `p_1` across 5 folds |
| 5-fold probability average @ best F1 threshold | 0.924729 | 0.886628 | 0.606061 | Threshold = 0.759977 |

## Interpretation

The ensemble improves ranking quality substantially, with AUC rising from 0.8813 to 0.9247. Under the default 0.5 threshold, the ensemble favors recall and produces many positive calls, which hurts F1. After threshold tuning, ACC and F1 improve, indicating the main issue is calibration rather than separation.

## Files

- [`summary.csv`](summary.csv)
- [`result.csv`](result.csv)
- [`ensemble_5fold_avg.csv`](ensemble_5fold_avg.csv)
- [`ensemble_5fold_avg_metrics.csv`](ensemble_5fold_avg_metrics.csv)
- [`ensemble_5fold_avg_threshold_best.csv`](ensemble_5fold_avg_threshold_best.csv)
- [`ensemble_5fold_avg_threshold_scan.csv`](ensemble_5fold_avg_threshold_scan.csv)
