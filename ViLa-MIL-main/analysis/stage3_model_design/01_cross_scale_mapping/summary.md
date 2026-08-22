# Stage 3.1 Summary

## Implementation

- `utils/cross_scale_mapping.py` builds deterministic coordinate-value bbox overlap mappings from the `coords` dataset inside each feature H5. It never reads or relies on standalone coordinate-H5 row order.
- Mapping is variable-cardinality CSR: `parent_ptr`, `child_indices`, `child_overlap_area`, `child_weight`, plus reverse CSR `high_parent_ptr`/`parent_indices`.
- `high_has_parent_mask` and `unmapped_high_indices` explicitly retain every high patch without a low parent. `low_valid_mask`, `high_valid_mask`, and per-patch padding ratios are stored.
- `.npz` caches are self-validating on load. `datasets.dataset_generic.Generic_MIL_Dataset` and `Generic_Split` accept optional `mapping_path` and `return_mapping=True`; default behavior is unchanged.
- With mapping enabled, a sample is `(features_s, coords_s, features_l, coords_l, label, slide_id, mapping_dict)`. The existing transformer collate returns the five legacy tensors plus slide IDs and a list of ragged mapping dictionaries. Mapping is not passed to the current model call.

## Small-sample validation

Command:

```bash
/opt/conda/envs/vila_mil_overlay_rt/bin/python \
  analysis/stage3_model_design/run_mapping_sanity.py --limit 3
```

Passed for 3 real slides:

| slide | low | high | edges | low coverage | high coverage | unmapped high |
|---|---:|---:|---:|---:|---:|---:|
| 2460239-B2 | 187 | 2,592 | 4,595 | 1.000000 | 0.930556 | 180 |
| 2460242-B2 | 426 | 6,128 | 12,923 | 1.000000 | 0.990372 | 59 |
| 2460399-B2 | 285 | 4,103 | 8,791 | 1.000000 | 0.995369 | 19 |

The sanity runner also checks every CSR edge by recomputing positive bbox overlap and checks `mapped_high union unmapped_high == all high indices`.

Additional checks passed:

- 6 synthetic unit tests: variable 0/1/many children, multiple parents, non-integer scale, CSR bounds/weights, padding masks, and NPZ round trip.
- Mapping-enabled dataset smoke test: 7-item sample and ragged mapping payload; all child indices index the feature-H5 high rows.
- Legacy dataset/collate smoke test: original 6-item sample and 5-item transformer batch.
- Python syntax compilation passed for mapper, dataset, collate, generator, and sanity runner.

## Full generation

For reproducibility, the full-run command was:

```bash
set -o pipefail
/opt/conda/envs/vila_mil_overlay_rt/bin/python \
  analysis/stage3_model_design/01_cross_scale_mapping/build_mappings.py \
  --output-dir analysis/stage3_model_design/01_cross_scale_mapping/full_output \
  2>&1 | tee analysis/stage3_model_design/01_cross_scale_mapping/full_build.log
status=${PIPESTATUS[0]}
echo "mapping_status=${status}"
test "${status}" -eq 0
```

Expected full-run checks are 968 processed slides, 210,489 low patches, 2,997,694 high patches, and approximately 57,570 unmapped high patches (1.92%), subject to the generated metadata/features being unchanged. Please provide `full_build.log`, `full_output/mapping_statistics.csv`, `full_output/summary.md`, and the `mapping_status` for final review.

## Full-run acceptance

The requested full run completed with `mapping_status=0`. Generated outputs contain:

- 968/968 slide mapping caches
- 210,489 low patches and 2,997,694 high patches
- 5,640,393 positive-area CSR edges
- 57,570 unmapped high patches (`0.019205`, or `1.9205%`)
- high-parent coverage range `0.756356` to `1.000000`

`CrossScaleMapping.load_npz(...).validate()` and complete high-index coverage checks passed on representative first, middle, last, and random caches. The minimum coverage and aggregate counts match the Step 2.4 audit (`57,570` unmapped high patches and `5,640,393` overlap relations). No source feature/coordinate H5, split, or Stage 1 output was changed by cache generation.

## Compatibility and gate

No feature H5, coordinate H5, split, model forward, logits, loss, or Stage 1 output was modified. Without `mapping_path`/`return_mapping`, the existing dual/high/low data path remains unchanged. Stage 3.1 is fully accepted and can proceed to Stage 3.2.
