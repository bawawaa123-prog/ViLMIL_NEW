# Stage 2 Model Requirements

These are design constraints, not an implementation.

## Required roles

1. The high-scale branch (directory `20x`, actual ~10x) must be the primary diagnostic evidence branch.
2. The low-scale branch (directory `5x`, actual ~2.5x) must provide tissue-level context, coarse localization, and/or routing of suspicious regions.
3. The model must use the Step 2.4 coordinate-based mapping, not H5 row order.

## Mapping constraints

- Parent-child cardinality must be variable; never hard-code `1 low = 16 high`.
- Support zero, one, and many high children per low patch.
- Preserve high patches without a low parent (about 1.92% overall) through an explicit policy such as an unparented/global bucket, rather than silent deletion.
- Use level-0 continuous bboxes, overlap/center criteria, and actual per-slide downsamples.
- Keep padding ratio available as a feature or mask; severe padding is nonblocking but should not be treated as ordinary tissue evidence without qualification.

## Fusion constraints

- Do not use uncalibrated equal `logits_low + logits_high` as the core fusion.
- High evidence must be able to dominate the final decision when low context is uninformative or contradictory.
- Prefer learned asymmetric fusion, gated residual context, or high-level logits conditioned on low-level context.
- Report High-only, Low-only, and Dual/ proposed model results on the same strict folds.

## Semantic and training constraints

- Keep the current Stage 1 BiomedCLIP feature convention fixed for the main comparison; official preprocessing is a separate sensitivity branch.
- Use BiomedCLIP pathology semantics to guide coarse-to-fine evidence selection, for example low-scale context-conditioned high-patch scoring or prompt-compatible region gates.
- Preserve early stopping and the exact strict5 evaluation protocol for controlled comparisons.
- Log parent coverage, unmapped counts, padding statistics, selected high-patch counts, and per-scale contributions.
- Treat the current split as provisional case-level unless hospital identity semantics confirm patient grouping.
