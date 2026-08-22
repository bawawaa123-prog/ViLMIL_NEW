# Step 2.3 feature repair post-check

## Execution result

The targeted BiomedCLIP re-extraction completed successfully for the 10
nominal-5x (actual approximately 2.5x) slides identified by the Step 2.3
audit. The run used the local BiomedCLIP snapshot and wrote only to
`feature_repair_staging_5x/`.

- Extraction exit status: `0`
- Verification exit status: `0`
- Repair targets: `10/10`
- Staged patch count: `2,411`
- Verification result: `10 passed, 0 failed`
- Coordinate order: exact for all 10 staged files
- Feature shape: `[N,512]` for every staged file
- Feature dtype: `float32` for every staged file
- Feature values: finite for every staged file

The NumPy/SciPy compatibility warning printed by the environment did not
affect extraction or verification; both commands completed successfully.

## Independent post-check

An independent read-only check compared every staged file against its source
coordinate H5. All 10 coordinate arrays matched in shape and row-by-row values;
the total staged count was 2,411. The same check confirmed that the old
production feature files still have their pre-repair shorter lengths, so no
production file was overwritten by this run.

## Current decision

- Staged repair: **verified**.
- Production feature repair: **committed and independently verified**.
- Post-commit 5x alignment: 968/968 count-equal and coordinate-set-equal; 12
  exact-order pairs and 956 reordered-only pairs.
- Post-commit 20x alignment: 968/968 count-equal and coordinate-set-equal; 0
  exact-order pairs and 968 reordered-only pairs.
- No count or coordinate-set mismatch remains in either scale.
- The complete Step 2.3 audit has now been rerun. Its final report records
  `critical=0`, `warning=2227`, `info=82`, `Step 2.3 pass=True`, and
  `ready_step_24=True`.
- Step 2.4 may now begin, subject to the order-mismatch caveat below.

## Final audit caveat

The 1,924 reordered-only pairs remain a property of the existing extraction
layout: feature rows are paired with the `coords` stored in the same feature
H5, not with the row order of the separate coordinate H5. Any downstream
consumer must retain that feature-H5-local pairing and must not index feature
rows by the separate coordinate-H5 row number.

The commit has already been performed. The backup is:

```bash
`feature_repair_backup_5x_20260820_180126/`
```

The backup contains all 10 old feature H5 files and `commit_manifest.csv`.
The production verification CSV is
`feature_repair_production_verification.csv`.

After regenerating the complete Step 2.3 audit, review its updated summary
before deciding whether to enter Step 2.4.
