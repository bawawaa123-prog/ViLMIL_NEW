# Stage 3.3.3 Routing Failure Analysis & Stabilization

## Scope

This stage prepares inference-only residual-scale intervention. It does not
implement Stage 3.4, change the routing formula, regenerate mappings, or train
strict5.

The current `LowParentContext` was rechecked: it uses vectorized reverse CSR
(`repeat_interleave` + `index_add_`) and contains no per-high-row Python loop.

## Intervention

The model now accepts runtime-only `config.routing_scale` (default `1.0`):

```text
h_routed = h_high + lambda * routing_residual
```

The value is not a parameter and does not alter checkpoint `state_dict`
schema. Production/default behavior remains lambda=1.0. The analysis script
tests lambda `0, 0.1, 0.25, 0.5, 1.0` on the same trained checkpoint and test
slides.

## Analysis command

Run from the repository root. This is inference only and loads the existing
Fold 1 (pilot) and Fold 2 (replication) E1 checkpoints:

```bash
/opt/conda/envs/vila_mil_overlay_rt/bin/python \
  analysis/stage3_model_design/03_low_context_routing/05_failure_analysis/run_lambda_sweep.py \
  --folds 1 2
```

Outputs:

```text
analysis/stage3_model_design/03_low_context_routing/05_failure_analysis/lambda_results/
```

Each fold emits a lambda summary CSV and slide-level JSONL diagnostics. The
diagnostics include route mean/std/min/max, context/high norms, scaled
residual/high norm ratio, mapped/unmapped counts, parent count, and padding
statistics. No feature or patch tensor is persisted.

## Stabilization decision

No stabilization module is implemented yet. First run the lambda sweep. If
lambda=0 or a small lambda materially restores Fold 2 performance, the next
minimal candidate is normalized low context plus one explicit bounded residual
scale. If lambda=0 remains poor, the likely cause is training co-adaptation or
the context formulation itself, not residual magnitude alone.

Stage 3.4 remains blocked pending this analysis.

## Completed lambda intervention

The inference sweep was run on the existing E1 checkpoints for Fold 1 (the
positive pilot fold) and Fold 2 (the largest strict5 failure). Each test split
contains 194 slides.

### Fold 1

| lambda | AUC | Accuracy | Macro-F1 | residual/high |
|---:|---:|---:|---:|---:|
| 0.00 | 0.9821 | 0.9278 | 0.9222 | 0.000 |
| 0.10 | 0.9826 | 0.9278 | 0.9222 | 0.0085 |
| 0.25 | 0.9831 | 0.9330 | 0.9276 | 0.0212 |
| 0.50 | 0.9850 | 0.9381 | 0.9329 | 0.0425 |
| 1.00 | 0.9869 | 0.9485 | 0.9438 | 0.0849 |

Fold 1 benefits monotonically from the full learned residual. Its lambda=0
inference is close to, but not exactly the same as, the independently trained
E0 checkpoint.

### Fold 2

| lambda | AUC | Accuracy | Macro-F1 | residual/high |
|---:|---:|---:|---:|---:|
| 0.00 | 0.9412 | 0.9124 | 0.9020 | 0.000 |
| 0.10 | 0.9420 | 0.9072 | 0.8959 | 0.0233 |
| 0.25 | 0.9420 | 0.9124 | 0.9020 | 0.0583 |
| 0.50 | 0.9427 | 0.9072 | 0.8967 | 0.1167 |
| 1.00 | 0.9355 | 0.9021 | 0.8913 | 0.2334 |

Fold 2 improves slightly when lambda is reduced from 1.0, but even lambda=0
remains below the independently trained E0 AUC `0.9587`. Therefore residual
scale is a real contributor, but it is not the only failure mechanism.

## Cross-fold diagnosis

The mean diagnostic statistics were very similar between folds:

| Fold | route mean | route std | context norm | high norm | mapped ratio | parent count | padding ratio | residual/high at lambda=1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2200 | 0.0328 | 4217.7 | 4295.9 | 0.9746 | 1.830 | 0.000136 | 0.0849 |
| 2 | 0.1800 | 0.0251 | 4223.9 | 4298.6 | 0.9781 | 1.864 | 0.000126 | 0.2334 |

Mapped/unmapped proportions, parent counts, padding ratios, context norms, and
high norms do not show a large data-coverage difference explaining Fold 2.
Route standard deviation is small relative to its mean in both folds, which
suggests the learned route behaves substantially like a slide/bag-level scale
factor rather than strongly patch-specific selection. This is a diagnostic
signal, not proof of exact global constancy.

## Failure mode and stabilization decision

The most likely failure is a combination of:

1. excessive residual magnitude in Fold 2 (`0.233` of the high-feature norm at
   lambda=1 versus `0.085` in Fold 1); and
2. training co-adaptation: the lambda=0 E1 checkpoint does not recover the
   separately trained E0 result, even though the inference residual is removed.

The data mapping itself is not the leading suspect in this comparison.

A minimal stabilization is justified for a small controlled follow-up, but not
for full strict5 yet: normalize low context before `Wc`, and add one explicit
bounded/small residual scale (for example a fixed initial scale or a bounded
learned scalar). Do not combine this with semantic routing, new fusion, or new
losses. Validate only Fold 1 + Fold 2 first and compare lambda/scale behavior.

Stage 3.3.3 diagnosis is complete. Stage 3.4 remains blocked; the next action
should be a separately named stabilization ablation, not a broad architecture
expansion.
