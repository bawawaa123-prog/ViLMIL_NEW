# Experiment Plan

All runs use the same strict5 case-level folds, seed, early stopping, current Stage 1 feature/preprocessing convention, and report AUC, Accuracy, Macro-F1. High-only and original Dual checkpoints are controls, not replaced.

| ID | Configuration | Purpose |
|---|---|---|
| E0 | High-only | Strong diagnostic floor (`AUC 0.9657` mean in Step 2.6) |
| E1 | Original Dual: independent branches, `logits_low + logits_high` | Reproduce Stage 1 control (`AUC 0.9712`, lower mean Acc/F1) |
| E2 | Asymmetric fusion only; no coordinate mapping, low global context gate | Test calibration/asymmetry independently |
| E3 | E2 + coordinate variable mapping and unmapped/padding masks | Test spatial alignment and coarse context |
| E4 | E3 + BiomedCLIP semantic high-patch routing | Test pathology-text evidence selection |
| E5 | Complete MVP: E4 + gated residual with small-init `alpha` and all logging | Primary proposed model |

The requested ablation names map to E0, E1, E2, E3, E4, and E5 respectively. Add a low-only run only as a sanity reference; it is already available from Step 2.6.

## Acceptance criteria

Primary: E5 must beat E0 on mean AUC without reducing mean Accuracy and Macro-F1 beyond predeclared tolerance (for example 0.01), and must improve at least 3/5 folds. Secondary: gate values should be non-saturated, mapped/unmapped coverage should match audit statistics, and removing mapping/semantic guidance should produce interpretable deltas.

## Diagnostics

Log per slide/fold: high parent coverage, unmapped count and fraction, number of high children per low quantiles, padding fraction, selected/weighted high count, mean route gate, residual norm, `logits_high` versus residual contribution, and failures for empty regions. Save top routed coordinates for qualitative overlays.

## No-go conditions

Stop or revert if feature-coordinate pairing changes, high evidence is silently dropped, route gates saturate at 0/1, E5 is consistently below E0, or residual magnitude routinely exceeds high-logit magnitude. Official BiomedCLIP preprocessing remains a separate sensitivity experiment, never mixed into the main causal ablation.

