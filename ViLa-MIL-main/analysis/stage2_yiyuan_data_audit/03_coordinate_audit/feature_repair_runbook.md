# Step 2.3 nominal-5x feature repair runbook

## Scope

The Step 2.3 audit found 10 nominal-5x (actual approximately 2.5x) feature H5
files whose patch counts do not match the current coordinate H5 files. This
repair targets only those 10 slides and 2,411 patches. It does not regenerate
coordinates, process nominal 20x features, rescan all 968 slides, or run Step
2.4.

The extraction command reads each patch directly from its WSI using the
coordinate H5 attributes (`patch_level=2`, `patch_size=256`) and preserves the
coordinate H5 row order in the staged feature H5. It uses the same custom
ImageNet normalization and 224 x 224 resize as the existing BiomedCLIP helper.

## Safety model

- The default extraction-script invocation is a read-only preflight.
- `--execute` writes candidates only under `feature_repair_staging_5x/`.
- Existing files under `data/yiyuan/features_biomedclip_5x/` are not changed by
  extraction or verification.
- Do not run `--commit-staged --yes` yet. After review, that mode will validate
  all 10 candidates, copy all old feature files to a timestamped backup
  directory, and atomically replace the originals.

## Commands to run

Run from the `ViLa-MIL-main` project root:

```bash
cd /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main
set -o pipefail

/opt/conda/envs/vila_mil_overlay_rt/bin/python \
  analysis/stage2_yiyuan_data_audit/scripts/reextract_yiyuan_5x_features.py \
  --execute \
  --batch-size 32 \
  2>&1 | tee analysis/stage2_yiyuan_data_audit/03_coordinate_audit/feature_repair_extraction.log
extract_status=${PIPESTATUS[0]}
echo "extract_status=${extract_status}"
test "${extract_status}" -eq 0 || exit "${extract_status}"

/opt/conda/envs/vila_mil_overlay_rt/bin/python \
  analysis/stage2_yiyuan_data_audit/scripts/verify_yiyuan_5x_feature_repair.py \
  2>&1 | tee analysis/stage2_yiyuan_data_audit/03_coordinate_audit/feature_repair_verification.log
verify_status=${PIPESTATUS[0]}
echo "verify_status=${verify_status}"
test "${verify_status}" -eq 0 || exit "${verify_status}"
```

Both commands must exit with status 0. The verifier should report:

```text
Validated 10 repair targets: 10 passed, 0 failed
```

If GPU memory is insufficient, rerun the extraction command with
`--batch-size 16`. Already valid staged slides are skipped, so a completed
candidate is not recomputed. Use `--overwrite-staging` only when the script
reports that an existing staged file is invalid.

## Outputs to return for review

Provide these files after both commands finish:

- `feature_repair_extraction.log`
- `feature_repair_verification.log`
- `feature_repair_verification.csv`

Also report both shell exit statuses. Keep the staged H5 files in
`feature_repair_staging_5x/`; there is no need to upload or paste their binary
contents. Do not commit the staged files or start Step 2.4 before the review.
