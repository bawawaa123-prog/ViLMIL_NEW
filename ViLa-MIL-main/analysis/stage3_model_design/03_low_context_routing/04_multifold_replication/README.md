# Stage 3.3.2 Multi-fold Replication

This directory trains only E1 folds 1-4. Fold 0 remains in the completed
pilot directory and E0 is never retrained. Existing Stage 2 High-only and
Stage 1 Dual strict5 summaries are read as controls.

Run remaining E1 folds. The script creates its log/results directories and
skips any fold that already contains `fold_summary.csv`, so it is safe to
resume after interruption. Note that the script's fold indices are zero-based:
`FOLD=1` is the second strict5 split; the completed pilot fold 0 is elsewhere.

```bash
bash analysis/stage3_model_design/03_low_context_routing/04_multifold_replication/run_remaining_e1.sh
```

Aggregate after training:

```bash
/opt/conda/envs/vila_mil_overlay_rt/bin/python \
  analysis/stage3_model_design/03_low_context_routing/04_multifold_replication/aggregate_replication.py
```

Outputs:

```text
results/E1_fold1..E1_fold4/       # independent training outputs
logs/E1_fold1..E1_fold4.log
aggregate/paired_fold_results.csv
aggregate/paired_summary.json
```

The aggregator checks that completed pilot fold 0 E0 equals the Stage 2
High-only fold 0 row, then reports E1 mean +/- std, Stage 2 E0 mean +/- std,
per-fold E1-E0 deltas, mean delta, improved-fold count, and worst degradation.
It also includes existing Dual values as reference columns without retraining,
plus E1 stop epoch, summed runtime, finite-diagnostics status, and whether both
routing gradient branches were observed.

Implementation check: `LowParentContext` currently uses vectorized
`repeat_interleave` plus `index_add_`; no per-high-row Python loop remains.
This stage does not change that implementation because current pilot speed is
acceptable and no mapping semantics are changed.

## Completed strict5 result

All five E1 folds completed. Fold 0 is the original pilot; folds 1-4 are the
replication runs. Existing Stage 2 High-only is the paired E0 control.

| Fold | E0 AUC | E1 AUC | Delta | E0 Acc | E1 Acc | Delta | E0 F1 | E1 F1 | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9820 | 0.9869 | +0.0049 | 0.9381 | 0.9485 | +0.0103 | 0.9329 | 0.9438 | +0.0108 |
| 2 | 0.9587 | 0.9355 | -0.0232 | 0.9124 | 0.9021 | -0.0103 | 0.9053 | 0.8913 | -0.0140 |
| 3 | 0.9738 | 0.9746 | +0.0008 | 0.9639 | 0.9536 | -0.0103 | 0.9597 | 0.9477 | -0.0119 |
| 4 | 0.9559 | 0.9616 | +0.0057 | 0.9067 | 0.9067 | 0.0000 | 0.8971 | 0.8956 | -0.0015 |
| 5 | 0.9581 | 0.9530 | -0.0051 | 0.9326 | 0.9378 | +0.0052 | 0.9269 | 0.9314 | +0.0045 |

Population mean +/- std:

| Model | Test AUC | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|
| E0 High-only | 0.9657 +/- 0.0103 | 0.9308 +/- 0.0204 | 0.9244 +/- 0.0221 |
| E1 routing | 0.9623 +/- 0.0177 | 0.9297 +/- 0.0214 | 0.9220 +/- 0.0239 |
| Current Dual reference | 0.9712 +/- 0.0114 | 0.9287 +/- 0.0194 | 0.9202 +/- 0.0218 |

Mean E1-E0 deltas were AUC `-0.0034`, accuracy `-0.0010`, and Macro-F1
`-0.0024`. E1 improved 3/5 folds for AUC and 2/5 for accuracy and F1. The
largest degradation was fold 2: AUC `-0.0232`, accuracy `-0.0103`, F1
`-0.0140`.

Stop epochs for E1 folds 1-5 were `13/27/14/16/13`; summed epoch runtimes were
approximately `13.7/26.9/13.2/14.9/12.1` minutes. Every fold's diagnostic file
was finite and showed non-zero context-projection and route-score gradients.
Sampled final route means were approximately `0.192/0.183/0.155/0.195/0.214`,
with non-zero routing residuals in every fold.

Conclusion: the routing mechanism trains correctly, but this form does not
provide a stable strict5 improvement over High-only. The positive fold-0 pilot
did not replicate as a mean gain. Do not claim improved performance and do not
advance to Stage 3.4 without a separate design review of route magnitude,
normalization, and fold-2 failure behavior.
