# Stage 3.3.4 Stabilized Low-conditioned Routing

## Scope

This stage implements one minimal stabilization only. It does not implement
Stage 3.4, semantic routing, fusion, hard selection, new loss, or any mapping
regeneration.

## Formula

For the router branch only, normalize the high row and low parent context:

```text
h_r = normalize(h_high)
c_r = normalize(c_low)
route = sigmoid(RouteMLP([h_r, c_r, valid, padding]))
r_raw = route * Wc(c_r)
r = r_raw * min(1, 0.10 * ||h_high|| / ||r_raw||)
h_routed = h_high + r
```

The fixed residual cap is `rho=0.10`, applied independently per mapped high
patch. Invalid/unmapped high rows have zero route and therefore remain exactly
the original high feature. Original high features used by the diagnostic
branch are not normalized or replaced; normalization is local to router/context
inputs and the context projection branch.

`Wc` and the final route layer remain zero initialized, preserving the exact
zero-init High baseline. Stabilization is explicit and disabled by default via
`--use_routing_stabilization`; `--use_low_context_routing` remains independent.
The module adds no parameters, so old routing checkpoints load strictly.

## Files

- `models/cross_scale_modules.py`: normalized router inputs and per-row rho cap.
- `models/model_ViLa_MIL_BiomedCLIP.py`: explicit config wiring.
- `main.py`, `utils/core_utils.py`: `--use_routing_stabilization` plumbing.
- `run_stabilized_folds.sh`: Fold 1/2 (zero-based 0/1) S1 training commands.
- `test_stabilized_router.py`: isolated tests.

## Validation completed

```bash
python -m py_compile models/cross_scale_modules.py \
  models/model_ViLa_MIL_BiomedCLIP.py utils/core_utils.py main.py \
  analysis/stage3_model_design/03_low_context_routing/06_stabilization/test_stabilized_router.py
python -m unittest \
  analysis.stage3_model_design.03_low_context_routing.06_stabilization.test_stabilized_router -v
bash -n analysis/stage3_model_design/03_low_context_routing/06_stabilization/run_stabilized_folds.sh
```

Tests cover zero-init exact baseline, unmapped preservation, finite
forward/backward, residual/high ratio `<=0.10` on mapped rows, non-zero context
and route projection gradients, and strict old-router state_dict loading.

## Fold 1/2 validation commands

Codex does not run these training jobs. Run both stabilized folds with:

```bash
bash analysis/stage3_model_design/03_low_context_routing/06_stabilization/run_stabilized_folds.sh
```

Results are isolated under:

```text
analysis/stage3_model_design/03_low_context_routing/06_stabilization/results/S1_fold0/
analysis/stage3_model_design/03_low_context_routing/06_stabilization/results/S1_fold1/
```

S0 is the existing Stage 3.3.2 Fold 1/2 result and is not retrained. Compare
training/validation loss, AUC, Accuracy, Macro-F1, early stopping epoch, and
diagnostics. Do not select rho or modify the structure from test metrics.

The stabilization is ready for the requested Fold 1/2 controlled validation;
Stage 3.4 remains out of scope.

## Completed S1 Fold 1/2 validation

The requested stabilized training completed for strict5 folds 1 and 2 (runner
indices 0 and 1). S0 below is the existing Stage 3.3.2 routing run.

| Strict fold | S0 val AUC | S1 val AUC | S0 val Acc | S1 val Acc | S0 test AUC* | S1 test AUC* |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9734 | 0.9698 | 0.9330 | 0.9433 | 0.9869 | 0.9812 |
| 2 | 0.9870 | 0.9891 | 0.9588 | 0.9639 | 0.9355 | 0.9493 |

`*` Test columns are recorded for completeness only. The stabilization
decision is based on train/validation behavior, as required.

### Training/validation interpretation

- **Fold 1:** S1 early-stopped at epoch 13, same as S0. Validation AUC was
  `0.9698` versus S0 `0.9734`; validation accuracy was `0.9433` versus S0
  `0.9330`. The original Fold 1 positive signal was not clearly preserved in
  validation AUC, although validation accuracy was higher.
- **Fold 2:** S1 early-stopped at epoch 21 versus S0 epoch 27. Validation AUC
  improved from `0.9870` to `0.9891`, and validation accuracy from `0.9588` to
  `0.9639`. This supports improved validation stability for the problematic
  fold, without relying on test metrics.

### Residual and routing diagnostics

All sampled S1 diagnostics were finite. The observed aggregate
`residual_high_ratio` stayed below the required `rho=0.10` cap:

| Strict fold | ratio max | last sampled ratio | route mean range | route std max |
|---:|---:|---:|---:|---:|
| 1 | 0.0259 | 0.0258 | 0.417-0.961 | 0.3395 |
| 2 | 0.0360 | 0.0346 | 0.377-0.991 | 0.3922 |

Mapped/unmapped rows remained present and diagnostics covered mapped ratios from
approximately `0.840-1.000` (Fold 1) and `0.756-1.000` (Fold 2). No NaN/Inf was
observed. Context/high feature norms remained finite.

One new issue is visible: normalized router input caused route means to become
high late in training (`~0.95-0.98` in the last samples), with non-trivial
route std. The residual cap prevents feature explosion, but it does not prevent
the route gate from becoming close to an all-mapped global gate. This suggests
the stabilization solved magnitude control more directly than
patch-specificity.

## Stage decision

The stabilization is partially successful:

1. The residual explosion is controlled: observed residual/high ratios are far
   below `0.10` in both folds.
2. Fold 2 validation stability improved and stopped earlier.
3. Fold 1 validation AUC did not improve, so the positive S0 signal is not
   consistently retained.
4. The route gate remains relatively global/saturated after normalization.

Do not run full strict5 or enter Stage 3.4 yet. The next decision should be a
small design review of normalized router input and gate saturation; no rho
search is justified from these two folds. The S1 implementation is ready for a
pre-declared follow-up only if the route saturation behavior is addressed.
