# Step 2.2: Yiyuan WSI Metadata and Physical Scale Audit

## Scope and formulas

- Population: 968 slides, dynamically matched to Step 2.0 inventory.
- OpenSlide: python 1.3.1, library 4.0.0; h5py 3.12.1.
- No WSI pixels or coordinate arrays were read. Only OpenSlide metadata and H5 dataset shape/dtype/attributes were accessed.
- Coordinates are level-0 positions: `create_patches_fp.py` generates candidates in level-0 space and downstream `read_region((x,y), level, size)` consumes level-0 locations.
- Effective magnification = objective power / actual OpenSlide level downsample.
- Effective MPP-X/Y = level-0 MPP-X/Y x dimension-derived level downsample-X/Y.
- Physical FOV-X/Y = patch size at selected level x dimension-derived downsample-X/Y x level-0 MPP-X/Y.
- Explicit fallback chains only: objective uses `openslide.objective-power`, then `aperio.AppMag`, then `hamamatsu.SourceLens`; MPP uses `openslide.mpp-x/y`, then explicit `aperio.MPP`. Missing values are never inferred from a nominal magnification.
- Cross-WSI anomaly threshold: >5% from the dataset median. H5/WSI downsample mismatch threshold: >0.1%. Nominal branch-name warning: >10%.

## WSI metadata completeness

- Successfully read: 968/968
- Metadata/coordinate critical-anomaly slides: 0
- Vendor counts: {'aperio': 968}
- Objective power counts: {'40.0': 968}
- Level-count distribution: {'4': 870, '3': 98}
- Pyramid topology signatures (level count and downsamples rounded to 0.1): unique=2, top 10={'4|1.0,4.0,16.0,64.0': 870, '3|1.0,4.0,16.0': 98}
- Exact downsample sequences: 862 (small dimension-rounding differences are retained in CSV and are not by themselves topology anomalies).
- MPP-X: n=968, min=0.264186, median=0.264186, max=0.264186, mean=0.264186, SD=0, CV=0% um/pixel
- MPP-Y: n=968, min=0.264186, median=0.264186, max=0.264186, mean=0.264186, SD=0, CV=0% um/pixel

## Coordinate H5 configuration

- 5x `(patch_level, patch_size)` counts: {('2', '256'): 968}
- 20x `(patch_level, patch_size)` counts: {('1', '256'): 968}
- Readable and complete H5 metadata rows: 1936/1936

## Actual magnification

- Nominal 5x branch actual magnification: n=968, min=2.49943, median=2.49979, max=2.5, mean=2.49979, SD=8.28088e-05, CV=0.003313% x
- Nominal 20x branch actual magnification: n=968, min=9.9995, median=9.99981, max=10, mean=9.99982, SD=8.85856e-05, CV=0.0008859% x
- High/low actual magnification ratio: n=968, min=4, median=4.00026, max=4.00072, mean=4.00026, SD=0.000128199, CV=0.003205%
- Slides with systematic nominal-magnification warnings: 968

## Physical FOV

- Nominal 5x branch geometric-mean FOV: n=968, min=1082.11, median=1082.2, max=1082.35, mean=1082.2, SD=0.035849, CV=0.003313% um
- Nominal 20x branch geometric-mean FOV: n=968, min=270.526, median=270.532, max=270.54, mean=270.531, SD=0.00239655, CV=0.0008859% um
- Low/high physical FOV ratio: n=968, min=4, median=4.00026, max=4.00072, mean=4.00026, SD=0.000128198, CV=0.003205%

## Anomalies

- Total anomaly records: 2034
- Slides with cross-WSI scale anomalies: 0
- Slides with pyramid structural anomalies: 98
- Slides with critical missing/unreadable metadata or H5 configuration: 0
- Anomaly type counts: {'pyramid_level_count_outlier': 98, 'systematic_nominal_magnification_deviation': 1936}

## Required answers

1. Metadata read success: **968/968**.
2. Current 5x configuration: **('2', '256')** as `(patch_level, patch_size)`.
3. Current 20x configuration: **('1', '256')** as `(patch_level, patch_size)`.
4. Nominal 5x uniformly close to 5x: **False**.
5. Nominal 20x uniformly close to 20x: **False**.
6. High/low ratio stably close to 4: **True**.
7. Physical FOV cross-WSI consistency within 5% of branch medians: **True**.
8. Pyramid structure consistent within the 5% per-level criterion: **False**.
9. Metadata/magnification/FOV anomaly slides: critical=0, cross-WSI scale=0, pyramid structure=98, systematic nominal-name deviation=968.
10. Classification: **5x/20x naming is imprecise, but the underlying data are scale-consistent**.
11. Ready for Step 2.3 coordinate audit: **True** (Step 2.3 was not executed).
12. Clear current evidence that patches must be regenerated: **False**. A systematic naming mismatch alone is not regeneration evidence when physical scales are internally consistent.

## Current conclusion

- 5x/20x naming is imprecise, but the underlying data are scale-consistent.
- Step 2.3 readiness: True.
- Patch-regeneration evidence from this metadata audit: False.
