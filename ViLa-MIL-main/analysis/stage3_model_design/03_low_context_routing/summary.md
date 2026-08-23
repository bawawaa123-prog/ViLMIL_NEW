# Stage 3.3.1 Mapping-to-model Context Plumbing

## Scope and preflight

This stage only constructs low-to-high context. It does not change logits, loss, routing, semantic guidance, fusion, or classifier behavior. A real BiomedCLIP preflight passed before and after the change using the local cache and the Stage 2 high-only s_0 checkpoint.

Both strict loads passed with missing_keys=[], unexpected_keys=[], and state_dict_keys=372. The new context module has 0 parameters.

## Implementation

- Added LowParentContext to models/cross_scale_modules.py.
- Added ViLa_MIL_BiomedCLIP.build_mapping_context(x_s, x_l, mapping).
- Extended forward(..., mapping=None). mapping=None runs the existing Stage 3.2 path. Supplied mapping builds context and stores it in _last_mapping_context, but does not affect logits.
- Existing image center, cross-attention modules, BiomedCLIP, prompts, pooling, loss, and scale selection remain unchanged.

## Tensor flow

    x_s [1,N_low,512] -> low_features [N_low,512]
    x_l [1,N_high,512] -> high row count N_high
    high_parent_ptr [N_high+1], parent_indices [E], parent_weight [E]
    high_parent_context [N_high,512]
    high_parent_context_valid_mask [N_high]
    high_has_parent_mask [N_high]
    unmapped_high_indices [N_unmapped]
    high_valid_mask [N_high], high_padding_ratio [N_high]

For every high row, reverse CSR selects low feature rows from parent_indices; valid parent_weight values are renormalized. Zero-parent high rows receive finite zero context and remain present. Invalid low parents are excluded, with zero fallback if no usable parent remains. No fixed parent/child count is used.

## Validation

Commands:

    python -m unittest discover -s analysis/stage3_model_design/03_low_context_routing -p test_*.py -v
    /opt/conda/envs/vila_mil_overlay_rt/bin/python analysis/stage3_model_design/03_low_context_routing/run_real_mapping_smoke.py

Four synthetic tests passed: 0/1/many parents, weighted multiple parents, invalid/padding masks, empty segments, finite backward, row-index bounds, and unmapped preservation.

Real smoke passed:

    2460239-B2: low=187 high=2592 unmapped=180 context_shape=(2592, 512)

All context values were finite and the unmapped count matched the Stage 3.1 cache.

## Compatibility

- Legacy forward(x_s, coord_s, x_l, coords_l, label) remains valid.
- mapping=None does not construct context and does not alter prediction.
- Mapping is optional and not consumed by the current training loop.
- No source feature, coordinate, split, mapping cache, or checkpoint was changed.
- Real strict checkpoint load passed after the new module was added with no missing/unexpected keys.

Stage 3.3.1 is accepted. No Stage 3.3.2 functionality was implemented.
