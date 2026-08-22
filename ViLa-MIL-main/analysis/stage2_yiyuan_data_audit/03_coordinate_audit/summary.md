# Step 2.3: Yiyuan Two-scale Coordinate and Feature Alignment Audit

## Scope and method

- Population: 968 slides; 1936 slide-scale coordinate H5 files and 1936 paired feature H5 files.
- Existing directory labels `5x/20x` are retained. Per Step 2.2, their actual magnifications are approximately `2.5x/10x`.
- Runtime: Python 3.12.6, OpenSlide Python 1.3.1 (library 4.0.0), h5py 3.12.1, NumPy 2.5.0.
- Level-0 WSI dimensions and dimension-derived actual level downsamples come from the completed Step 2.2 outputs.
- Boundary checks use the full continuous level-0 footprint `patch_size * actual downsample`, not only coordinate top-lefts.
- Spacing is inferred from positive adjacent x differences within equal-y rows and adjacent y differences within equal-x columns. No fixed stride is assumed.
- Feature matrices are not loaded; their first dimension/shape/dtype are read from H5 metadata. Both coordinate arrays are loaded and compared row by row and after sorting.
- Code-path review: the current BiomedCLIP extractor enumerates PNG files with `os.listdir`, uses `shuffle=False`, and appends each feature batch and its filename-derived coordinates together. This explains arbitrary cross-file order while preserving feature-H5-internal row pairing.
- Existing coordinate generation defaults to `use_padding=True`; boundary-crossing footprints are therefore reported as padding warnings, not treated alone as coordinate corruption.
- Coverage comparison is limited to footprint bbox overlap, normalized bbox-center distance, and representative thumbnails. No low-to-high parent-child mapping is constructed.

## Coordinate counts

- nominal 5x (actual ~2.5x): total=210,489; min=20, p25=147.7500, median=212, p75=277, max=519, mean=217.45, SD=91.65.
- nominal 20x (actual ~10x): total=2,997,694; min=241, p25=2,075, median=3,018.5000, p75=3,983, max=7,752, mean=3,096.79, SD=1,385.08.

## Grid, validity, and alignment results

- Coordinate read success: 5x=968/968; 20x=968/968.
- Duplicate coordinates: 0 across 0 slide-scale pairs / 0 slides.
- Coordinate shape/dtype: all 1,936 datasets are `[N,2]`; modal dtype is `int32`, with `int64` in 42 slide-scale pairs across 21 slides. Feature coords are uniformly `int64`; features are uniformly `[N,512] float32`.
- Invalid level-0 top-left coordinates: 0 across 0 slides.
- Out-of-bounds full patch footprints: 1,482 across 303 slide-scale pairs / 196 slides (5x=392 patches/144 slides; 20x=1,090 patches/159 slides).
- Data-derived dominant spacing: 5x x/y=4096 level-0 px on 968/968 slides; 20x x/y=1024 level-0 px on 968/968 slides.
- Obvious dominant-spacing/isolation anomalies: 0 records across 0 slides. Separately, 40 informational nonmultiple-gap review flags affect 28 slides.
- Coordinate/feature counts all equal: 1936/1936 slide-scale pairs.
- Coordinate values and order exactly equal: 12/1936 slide-scale pairs.
- Same coordinate multiset but different order: 1924 slide-scale pairs across 968 slides.
- Count/value mismatches: 0 5x pairs across 0 slides; current coordinate H5 contains 0 coordinates absent from feature H5, while feature H5 contains 0 coordinates absent from current coordinate H5.
- Per scale: 5x count-equal=968/968, exact-order=12/968, reordered-only=956/968; 20x count-equal=968/968, exact-order=0/968, reordered-only=968/968.
- Fully aligned coordinate-feature pairs: 12/1936.
- Preliminary scale coverage: bbox IoU min=0.4021, median=0.9682, max=1.0000; warning/critical records=0 across 0 slides.

## Anomalies

- Records: critical=0, warning=2227, statistical-info=82; affected slides=968.
- Issue counts: {'coord_feature_order_mismatch': 1924, 'coordinate_dtype_nonmodal': 42, 'patch_footprint_out_of_bounds': 303, 'spacing_nonmultiple_fraction_high': 40}.
- `patch_count_outer_outlier` is descriptive only. It is not treated as evidence of invalid coordinates or a regeneration requirement.
- Feature-refresh slides (0, all nominal 5x): .

## Representative visual review

- `25032146B2` (few_patches): [figures/coverage_25032146B2.png](figures/coverage_25032146B2.png).
- `2476358-B2` (median_patches): [figures/coverage_2476358-B2.png](figures/coverage_2476358-B2.png).
- `2485803-B2` (many_patches): [figures/coverage_2485803-B2.png](figures/coverage_2485803-B2.png).
- `2460239-B2` (anomaly_patch_footprint_out_of_bounds): [figures/coverage_2460239-B2.png](figures/coverage_2460239-B2.png).
- `2469230-B` (anomaly_spacing_nonmultiple_fraction_high): [figures/coverage_2469230-B.png](figures/coverage_2469230-B.png).
- `2486859-B2` (lowest_bbox_iou): [figures/coverage_2486859-B2.png](figures/coverage_2486859-B2.png).
- Red shows nominal 5x (actual ~2.5x) patch footprints; cyan shows nominal 20x (actual ~10x) patch footprints. The third panel overlays both.
- Manual review of the generated figures: low/high coverage follows the same tissue regions in the few/median/many-patch examples, with the expected coarser low-scale footprint. No figure shows a global low/high displacement.
- The minimum-bbox-IoU example (`2486859-B2`) still targets the same right-hand tissue fragment, but high-scale coverage is broader than low-scale coverage; both scales largely omit the separate left fragment. Edge-crossing examples are consistent with the configured padding behavior.

## Required answers

1. All 968 slides' two-scale coordinates read successfully: **True** (5x=968/968, 20x=968/968).
2. Patch-count distributions: **5x median 212, range 20-519; 20x median 3,018.5000, range 241-7,752**. Full quartiles and means are above and in `coordinate_statistics.csv`.
3. Duplicate or illegal coordinates: **duplicates=0; illegal top-lefts=0**.
4. Patch footprints crossing WSI boundaries: **1,482 patches across 196 slides**.
5. Obvious spacing-anomaly slides: **0**. All slides share the learned dominant spacing; 28 slides retain informational nonmultiple-gap flags, consistent with separately anchored contour grids.
6. Coordinate H5 vs feature H5: counts equal **1936/1936**; coordinate values and order equal **12/1936**; reordered-only pairs **1924**. The remaining 0 pairs are 5x count/set mismatches.
7. Issues that may misalign features and actual patches: **True**. Ten 5x feature files omit 0 current coordinates. In addition, using original coordinate-H5 row order with feature arrays would mispair 1,924 pairs; consumers must use the `coords` stored beside `features`.
8. Clearly abnormal overall two-scale coordinate coverage: **False** (0 slides flagged).
9. Data can proceed to Step 2.4 low/high spatial-mapping audit: **True**.
10. Clear evidence requiring coordinate/feature regeneration: **coordinate regeneration=False; feature regeneration=False** (0 nominal-5x feature files).

## Conclusion

- Step 2.3 pass: **True**.
- Ready for Step 2.4: **True**. Step 2.4 was not performed here.
- Coordinate regeneration currently required: **False**. Boundary padding should be reviewed against study policy, but is intentional in the current generator.
- Feature regeneration currently required: **False**, limited to the 0 listed nominal-5x feature files.
