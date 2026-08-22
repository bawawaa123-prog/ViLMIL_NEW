# Step 2.7: Stage 2 Data Audit Final Summary and Model Design Decision

## Executive conclusion

Stage 2 data auditing is **conditionally passed for model development**. The 968-slide inventory is complete, both scales are readable, coordinates are geometrically valid at level 0, feature values are numerically healthy, and low/high bboxes have reliable spatial correspondence. No new evidence requires regenerating WSI patches, coordinates, or features. The main unresolved data-governance issue is that the available strict split is proven case_id-level, but patient-level identity cannot be confirmed from repository metadata.

## Audit status by step

| Step | Status | Final finding |
|---|---|---|
| 2.0 inventory | Pass | 968/968 WSI, coordinate, and feature pairs; no missing/orphan files |
| 2.1 identity/split | Conditional | No literal case_id leakage; 59 naming-based candidate groups cross folds require hospital confirmation |
| 2.2 WSI metadata | Pass with naming caveat | Actual scales ~2.5x/~10x; ratio and physical FOV stable; 98 pyramid-depth outliers but no critical metadata failures |
| 2.3 coordinates/alignment | Pass with consumer constraint | No duplicate/illegal top-lefts; count/set equality 1936/1936; row order usually differs |
| 2.4 cross-scale mapping | Pass | Low-child coverage 99.999%; high-parent coverage 98.08%; variable mapping is valid, exact 16-child rule is not |
| 2.5 features | Pass with preprocessing caveat | All `[N,512] float32`, no NaN/Inf/zero vectors; project preprocessing differs from official preprocessing |
| 2.6 scale ablation | Pass | High-only is strongest single-scale; dual has small AUC gain but worse mean Accuracy/F1 |

## Data regeneration decision

**WSI patches: no. Coordinates: no. Features: no.**

The 1,482 boundary-crossing footprints across 196 slides are consistent with the existing padding-enabled generator. Severe padding (`>50%`) affects 443 patches across 94 slides, so it should be tracked in later modeling, but it is not evidence of systematic corruption. The previous 5x feature repair was staged, committed, and verified; current feature H5 contents are complete and numerically valid.

## Scale and spatial structure

The directory labels remain `5x/20x` for compatibility, but reports and models must state actual approximately `2.5x/10x`. Dominant level-0 spacings are 4096 and 1024 pixels. Low/high footprints overlap reliably, but exact 16-child regularity is not present: only about 2.12% of low patches have exactly 16 positive-overlap high patches. This reflects contours, offsets, boundaries, and variable sampling rather than a failed scale relationship.

Therefore, a coordinate-based variable parent-child mapping can be built directly offline. It must use coordinate values and continuous bboxes, not row indices, and must support multiple parents/children and edge cases. About 1.92% of high patches have no low parent; these must be handled explicitly.

## Feature validity and preprocessing

All 1,936 feature H5 files are `[N,512] float32`; total vectors are 3,208,183, with zero NaN, Inf, and zero vectors. Current features use direct bilinear resize and ImageNet normalization, not the official BiomedCLIP transform. In the controlled A/B sample, cosine similarity was median 0.9830 (range 0.9734–0.9903), so the difference is meaningful but not catastrophic. The current Stage 1 baseline remains a valid, coherent baseline only if this preprocessing convention is kept fixed and documented.

## Scale ablation decision

| Setting | AUC | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Low-only | 0.9220 ± 0.0260 | 0.8564 ± 0.0346 | 0.8385 ± 0.0377 |
| High-only | 0.9657 ± 0.0103 | 0.9308 ± 0.0204 | 0.9244 ± 0.0221 |
| Dual-scale | 0.9712 ± 0.0114 | 0.9287 ± 0.0194 | 0.9202 ± 0.0218 |

High-only is clearly stronger than Low-only. Dual improves AUC over High-only by about `+0.0055`, but mean Accuracy changes by `-0.0020` and Macro-F1 by `-0.0042`. The AUC improvement occurs in 4/5 folds; Accuracy and F1 improve only in fold 4. Thus the result does not support a claim that the current dual-scale design is uniformly or robustly superior.

## Main structural bottleneck

The primary bottleneck is not missing data or invalid geometry. It is the model's fusion assumption: independent low/high attention and text-conditioned logits are added with equal, uncalibrated weight. Low-scale context is weaker as a direct classifier, while high-scale evidence carries most diagnostic signal. Equal logit addition can therefore perturb decision calibration and explain why dual AUC rises slightly while thresholded Accuracy/F1 decline.

## Recommended Stage 2 direction

Use High as the main diagnostic branch and Low as a coarse context/routing signal. For each low patch, retrieve a variable set of overlapping high patches using the audited coordinate mapping; condition high-patch selection or aggregation on low-scale tissue context and BiomedCLIP pathology semantics; then apply a learned asymmetric fusion or gated residual context mechanism. Keep unparented high patches in an explicit global/edge pathway. Evaluate against the unchanged High-only and Stage 1 Dual baselines on the same strict folds.

Stage 2 should **retain** the current features, coordinates, actual-downsample metadata, strict fold protocol, early stopping, and High-only baseline. It should **retire as the main design** equal logits addition and any row-order-based cross-scale pairing. It should **add** variable coordinate mapping, padding/unmapped masks, learned asymmetric fusion, and logging of scale-specific evidence.

## Final status

**Step 2.7: complete.** No data regeneration is currently justified. Model development may proceed under the requirements in `stage2_model_requirements.md`, with patient-level identity semantics kept as an explicit qualification on generalization claims.
