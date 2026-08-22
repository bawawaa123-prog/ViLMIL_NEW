# Stage 2 Decisions

## Data disposition

- Do not regenerate WSI patches, coordinates, or BiomedCLIP features based on the completed audits.
- The systematic 5x/20x naming mismatch is a documentation issue: actual magnifications are approximately 2.5x and 10x, with a stable ratio of about 4.
- Existing feature repair and post-verification are accepted. Current H5 files are numerically valid and coordinate-set complete.
- Preserve the existing Stage 1 feature/preprocessing convention for comparability. Do not silently replace it with official BiomedCLIP preprocessing.

## Split qualification

The strict5 split has no confirmed literal `case_id` leakage and covers all 968 slides per fold. It is nevertheless a provisional case-level split, not a proven patient-level split: naming-based candidate groups cross folds and require hospital confirmation. This is a governance limitation, not a reason to discard the current baseline before identity semantics are resolved.

## Spatial decision

The data support an offline coordinate-value based low-to-high mapping. Mapping must use continuous level-0 patch bboxes and variable overlap lists. The observed data do not support hard-coding one low patch to exactly 16 high patches. Edge/unmapped high patches and padding must remain explicit states.

## Model decision

High-only is the strongest single-scale diagnostic baseline. Low-scale information should be retained as tissue-level context, routing, and coarse-to-fine evidence selection. The current equal `logits_low + logits_high` fusion is not accepted as the principal cross-scale design because it improves AUC only slightly while reducing mean Accuracy and Macro-F1 relative to High-only.

Step 2.7 decision: **Stage 2 audit complete; data are usable for controlled model development with the above constraints.**
