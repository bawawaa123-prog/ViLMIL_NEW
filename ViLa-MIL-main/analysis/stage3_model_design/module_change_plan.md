# Module Change Plan

## Keep unchanged

- `models/model_ViLa_MIL_BiomedCLIP.py`: BiomedCLIP loading, current 512-D feature contract, prompt learner, text preprocessing and frozen-by-default text configuration.
- `datasets/dataset_generic.py`: slide split semantics and H5 feature/coordinate co-loading, with the important rule that feature-H5 `coords` stay paired with feature rows.
- `utils/core_utils.py`: training/evaluation loop and strict5 early stopping for the first comparison.
- `main.py`: task definitions, optimizer protocol, and existing `scale_mode` baselines.

## Add or modify in small units

| Area | MVP change | Explicit non-goal |
|---|---|---|
| Mapping | New coordinate mapper utility/cache that returns variable parent lists, overlap weights, padding ratio, and unmapped mask | No row-index pairing; no fixed 16-child reshape |
| Dataset | Optional mapped sample return: `mapping`, `high_parent_index`, masks; preserve legacy five-item return behind a flag | No feature regeneration |
| Model | `LowRegionContext`, `HighRouter`, `MaskedGatedAttention`, `AsymmetricResidualFusion` | No new backbone or full transformer |
| Text | Reuse current high prompts for patch-text routing score | No second text tower |
| Logging | Parent coverage, unmapped count, selected counts, padding statistics, gate mean | No full experiment dashboard |

## Existing-module decisions

- `learnable_image_center`: remove from the MVP forward path. If checkpoint compatibility is needed, leave the parameter registered but unused and mark it deprecated; delete only after baseline parity is recorded.
- `cross_attention_1`: replace with region-aware masked pooling. A later ablation may restore it inside the high branch, but it is not required for the first causal test.
- `cross_attention_2`: remove from MVP routing/classification. Text/image normalized similarity is easier to inspect and avoids a text query attending to an unstructured concatenation of all patches.
- Attention pooling: retain and make mask-aware. Use one low-region pool and one routed high pool; do not use a single attention over concatenated low/high bags.
- `logits_low + logits_high`: remove as proposed-model behavior. Keep it only as the `Original Dual` control.

## Suggested file ownership

`models/model_ViLa_MIL_BiomedCLIP.py` should eventually become a thin composition layer. Put pure geometry in `utils/cross_scale_mapping.py`, reusable attention/gating in `models/cross_scale_modules.py`, and sample-contract adaptation in `datasets/dataset_generic.py` or a dedicated `datasets/mapped_wsi.py`. This separation permits unit tests without loading BiomedCLIP.

