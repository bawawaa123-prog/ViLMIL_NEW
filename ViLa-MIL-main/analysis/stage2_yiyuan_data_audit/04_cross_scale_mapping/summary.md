# Step 2.4: Yiyuan Cross-scale Spatial Mapping Audit

## Scope and method
- Population: 968 slides; mapping uses level-0 continuous bboxes from actual downsample and patch size.
- Nominal 5x is treated as actual ~2.5x; nominal 20x as actual ~10x.
- Mapping is coordinate-value based; no H5 row-order assumption is used.
- A relation is an overlap with positive area. Child/parent counts additionally report high-center-inside and high-fully-inside conditions.
- No WSI pixels or feature matrices are read.

## Results
- low patches with >=1 high overlap: median=1.0000, min=0.9941, max=1.0000.
- high patches with >=1 low overlap: median=0.9834, min=0.7564, max=1.0000.
- low patches with exactly 16 high overlaps: median=0.0195, min=0.0000, max=0.5084.
- Mean per-slide median low child count: 28.838; exact-16 overlap ratio across all low patches: 0.0212.
- Low patches with >=1 overlapping high: 210,487/210,489 (1.0000); high patches with >=1 low parent: 2,940,124/2,997,694 (0.9808).
- High-center-inside relations: 2,833,741; high-fully-inside relations: 2,828,032; positive-area overlap relations: 5,640,393.
- Unmapped: low without child=2; high without parent=57,570.
- Mapping anomaly records: critical=0, warning=12, info=967.
- The full relationship CSV contains one row per low patch and its exact overlapping high indices; the reverse CSV contains one row per high patch and parent indices.

## Padding severity
- 5x: total 210,489; <=5%=210,126 (99.8275%), 5-10%=49 (0.0233%), 10-25%=91 (0.0432%), 25-50%=111 (0.0527%), >50%=112 (0.0532%).
- 20x: total 2,997,694; <=5%=2,996,654 (99.9653%), 5-10%=69 (0.0023%), 10-25%=299 (0.0100%), 25-50%=341 (0.0114%), >50%=331 (0.0110%).
- Severe >50% padding affects 112 low patches and 331 high patches; 94 slides have at least one >50% padding patch in either scale.

## Required answers

- Stable low/high spatial correspondence: **yes** under the operational thresholds low-with-child >=95% and high-with-parent >=90%.
- Rule-like 16-child structure: **no** for strict positive-overlap count=16; exact-16 is not required for valid coordinate mapping because bbox boundary offsets can create neighboring overlaps.
- Spatial shift: no global shift is inferred unless per-slide anomalies identify one; coordinate bboxes are compared directly.
- Coordinate/patch regeneration: **not indicated by mapping alone; use offline mapping unless critical anomalies are present**.
- Step 2.4 pass: **True** (critical anomalies=0; stable correspondence=True).
- The 12 warning slides with high-parent coverage below 90% are listed in `mapping_anomalies.csv`; the lowest is `2486859-B2` at 75.64%.

## Runtime and outputs
- Runtime: Python 3.12.6, NumPy 2.5.0.
- Outputs: `mapping_statistics.csv`, `low_to_high_mapping.csv`, `high_to_low_statistics.csv`, `unmapped_regions.csv`, `padding_severity.csv`, `mapping_anomalies.csv`, and `figures/`.
