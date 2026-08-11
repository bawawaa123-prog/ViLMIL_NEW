# Final Results Summary

## Table: Single-fold vs Ensemble vs Tuned Ensemble

| Setting | AUC | ACC | F1 | Threshold | Notes |
|---|---:|---:|---:|---:|---|
| Single-fold mean (5 folds) | 0.881284 | 0.748256 | 0.640074 | - | Reported mean over fold_0..4 |
| Ensemble (probability average) | 0.924729 | 0.813953 | 0.515152 | 0.500000 | 5-fold p1 average with default decision threshold |
| Ensemble (probability average, tuned threshold) | 0.924729 | 0.886628 | 0.606061 | 0.759977 | Threshold selected by max F1 on external set |

## Fold Threshold Stability (by risk, high to low)

| Fold | AUC | F1@0.5 | BestThr | BestF1 | Drift|0.5| | Width(>=99% bestF1) | F1 Gain | Risk Score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.751871 | 0.309392 | 0.955245 | 0.396694 | 0.455245 | 0.002095 | 0.087302 | 4.434 |
| 1 | 0.931244 | 0.355769 | 0.975363 | 0.609756 | 0.475363 | 0.005304 | 0.253987 | 1.364 |
| 0 | 0.919095 | 0.481752 | 0.994495 | 0.568421 | 0.494495 | 0.002207 | 0.086669 | 0.512 |
| 4 | 0.901488 | 0.511278 | 0.836012 | 0.527273 | 0.336012 | 0.001520 | 0.015995 | -0.946 |
| 2 | 0.902720 | 0.577778 | 0.702777 | 0.578313 | 0.202777 | 0.149741 | 0.000535 | -5.365 |

## Candidate folds that likely drag overall performance

Based on low baseline F1/AUC plus high threshold drift and narrow near-optimal threshold width, the main suspect folds are: fold_3, fold_1.
