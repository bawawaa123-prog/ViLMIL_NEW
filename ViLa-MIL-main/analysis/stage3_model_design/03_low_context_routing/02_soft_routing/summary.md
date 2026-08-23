# Stage 3.3.2 Soft Low-conditioned High Routing

## Scope

Implemented the minimum differentiable low-conditioned high route. No semantic
routing, hard selection, asymmetric slide residual, auxiliary loss, or Stage 3.4
fusion was added.

## Routing module

`models/cross_scale_modules.py` now contains `HighRouter`:

```text
r = sigmoid(RouteMLP([h_high, c_low, valid_parent, padding_ratio]))
h_routed = h_high + r * Wc(c_low)
```

`Wc` is a zero-initialized linear projection. The final route projection is
also zero initialized, so the enabled module starts exactly at the original
high feature path. Context-validity and high-validity masks force rows without
usable low parents to have zero residual. Thus unmapped rows remain present and
unchanged. The module is soft and fully differentiable for mapped rows.

The route is applied to `x_l` immediately before the existing high
`cross_attention_1`, shared gated attention pooling, BiomedCLIP text alignment,
and `logits_high` computation. The high classifier and loss are unchanged.

## Training/data plumbing

- `main.py`: added `--use_low_context_routing` and `--mapping_path`; enabling
  routing requires a mapping path and sets dataset `return_mapping=True`.
- `utils/core_utils.py`: passes the optional per-slide mapping from collate
  through train, validate, and summary into `model(..., mapping=mapping)`.
- Default routing-off remains the legacy five-item loader and calls the model
  with `mapping=None`.
- `ViLa_MIL_BiomedCLIP` creates `HighRouter` only when enabled; mapping is
  otherwise only the Stage 3.3.1 context plumbing.

## Validation

Syntax:

```text
python -m py_compile models/cross_scale_modules.py \
  models/model_ViLa_MIL_BiomedCLIP.py datasets/dataset_generic.py \
  utils/utils.py utils/core_utils.py main.py
```

Unit/synthetic tests:

```text
python -m unittest discover \
  -s analysis/stage3_model_design/03_low_context_routing/02_soft_routing \
  -p 'test_*.py' -v
```

Result: 4 tests passed. Covered zero-init exact parity, mapped gradient,
0/1/many parent contexts, unmapped/empty/invalid masks, finite output, and
loader mapping payload round-trip.

Real one-slide smoke:

```text
/opt/conda/envs/vila_mil_overlay_rt/bin/python \
  analysis/stage3_model_design/03_low_context_routing/02_soft_routing/run_real_routing_smoke.py
```

Slide `2460239-B2` used real 5x/20x H5 features and the Stage 3.1 mapping:
`low=187`, `high=2592`, output shape `(1, 2)`, finite forward. The observed
single-slide forward elapsed time was `2.033 s`; the materialized per-high
context tensor was `5.06 MiB` (`2592 x 512` float32). This is not currently a
clear bottleneck, so no broader refactor was made. No full strict5 or training
was run.

Checkpoint compatibility smoke used:

```text
analysis/stage2_yiyuan_data_audit/06_scale_ablation/high_only/adenocarcinoma_biomedclip_high_only_strict5_s1/s_0_checkpoint.pt
```

With routing disabled, real BiomedCLIP construction and `strict=True` load
passed: `missing_keys=[]`, `unexpected_keys=[]`, `model_keys=372`, and the
legacy model has no `high_router` attribute. Existing `scale_mode=high` (and
the unchanged low/dual branches) remain available. Enabling routing adds only
the explicitly versioned router parameters, so an old checkpoint should be
loaded with the default routing-off configuration.

## Acceptance

Stage 3.3.2 implementation and focused validation passed. The next step is
Stage 3.4 only after reviewing the small pilot; no Stage 3.4 functionality is
included here.
