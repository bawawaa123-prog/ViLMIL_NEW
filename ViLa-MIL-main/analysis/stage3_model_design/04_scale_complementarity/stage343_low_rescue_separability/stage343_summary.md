# Stage 3.4.3 Low-rescue separability / confidence-aware late-fusion feasibility

This is a validation-only diagnostic over the existing Stage 3.4.2 prediction CSVs. No model inference, retraining, learned gate, calibration, or test evaluation was performed.

## Input checks

Both folds passed the required 194-slide, unique-ID, non-missing-label checks and matched the Stage 3.4.2 disagreement counts (Fold 0: 25; Fold 1: 24). Rescue/harmful counts were Fold 0: 4/21 and Fold 1: 5/19.

## Margin separability

- `high_margin` separability AUC: Fold 0 `0.536`, Fold 1 `0.695`; best orientations: `le`, `le`.
- `low_margin` separability AUC: Fold 0 `0.702`, Fold 1 `0.663`; best orientations: `le`, `ge`.
- `margin_delta` separability AUC: Fold 0 `0.619`, Fold 1 `0.716`; best orientations: `le`, `ge`.
- `log_margin_ratio` separability AUC: Fold 0 `0.607`, Fold 1 `0.705`; best orientations: `le`, `ge`.

A separability AUC near 0.5 indicates little rescue-versus-harmful separation. Fold disagreement counts are small, so these statistics are descriptive and high variance.

## Cross-fold transfer

- Fold 0 -> Fold 1: `always_high` / `no_switch` (threshold `n/a`); target switched/rescue/harm/net = 0/0/0/0; accuracy delta `+0.0000`, AUC delta `+0.0000`, macro-F1 delta `+0.0000`.
- Fold 1 -> Fold 0: `threshold` `margin_delta` `ge` threshold `1.5803`; target switched/rescue/harm/net = 4/0/4/-4; accuracy delta `-0.0206`, AUC delta `-0.0272`, macro-F1 delta `-0.0229`.

The within-fold rules are exploratory upper bounds only; they are not cross-fold generalization results. Rule selection maximized source-fold net_gain over the fixed one-dimensional candidate space, then froze the rule for the target fold.

## Artifacts

See `disagreement_cases.csv`, `feature_summary.csv`, `feature_separability.csv`, `cross_fold_transfer.csv`, `selective_fusion_metrics.csv`, `within_fold_optimal_rules.csv`, and `class_distribution.csv`.
