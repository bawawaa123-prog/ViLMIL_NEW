# Implementation Plan

Each stage is independently testable and revertible. Do not combine stages until the preceding contract and baseline test pass.

## Stage 3.1: Mapping and mask MVP

Implement only the offline coordinate mapper and a dataset-side optional mapping payload. Read feature-H5 coordinates, compute continuous level-0 bboxes with per-slide downsample/footprint metadata, and emit CSR `parent_ptr/child_index/child_weight`, `unmapped_high_index`, overlap statistics, and padding masks. Preserve the legacy five-item loader by default.

Real files: add `utils/cross_scale_mapping.py`; add focused tests under `tests/` (or the repository's existing test location); minimally extend `datasets/dataset_generic.py` behind `return_mapping=False` and add a CLI/cache script under `analysis/stage3_model_design/` or `utils/`. Do not change model behavior or train.

Independent tests: synthetic variable cardinalities 0/1/many, coordinate permutation, overlap weights, empty parent, and audit-level coverage/unmapped count. Revert by disabling the flag or deleting only the new utility.

## Stage 3.2: High-only parity module

Extract mask-aware gated attention and high pooling into reusable modules while keeping output numerically equivalent to the existing `scale_mode=high` path within a tolerance. No low context, mapping, or text routing yet.

Real files: add `models/cross_scale_modules.py`; refactor `models/model_ViLa_MIL_BiomedCLIP.py` with a compatibility path; add module unit tests. This creates the stable high branch for later changes.

## Stage 3.3: Low context conditioned high branch

Consume Stage 3.1 mapping. Pool low regions, gather variable high-child context, and add a soft low-conditioned high route. Keep semantic routing disabled and keep the final classifier high-only plus a zero-initialized/small residual gate.

Real files: `models/model_ViLa_MIL_BiomedCLIP.py`, `models/cross_scale_modules.py`, `utils/core_utils.py` only if logging fields require it. Test forward/backward, unmapped fallback, and no NaN on empty segments.

## Stage 3.4: Asymmetric residual fusion

Enable the learned `alpha * gate * residual` fusion and compare it to E0 and E1. Add explicit logging of high and residual contributions. Keep low auxiliary loss off.

Real files: model module and minimal result/log serialization in `utils/core_utils.py`.

## Stage 3.5: Semantic guidance

Use existing BiomedCLIP high pathology text embeddings to modulate route scores. Freeze text by default; expose one flag for semantic routing. Validate that routing changes selected coordinates without changing feature extraction/preprocessing.

Real files: model module, possibly `main.py` for a flag/config, and diagnostics in `utils/core_utils.py`.

## Stage 3.6: Full MVP integration and ablation runner

Wire the complete E0-E5 matrix, preserve original Dual as a control, add configuration/version manifests, and run only the small strict5 pilot needed to decide whether the MVP merits broader experimentation.

Real files: `main.py`, `utils/core_utils.py`, model/dataset modules, and a new analysis runner. No data regeneration.

## Deferred enhancements

Wait until E5 shows a reproducible gain before adding hard top-k selection, multi-head cross-attention, learned parent competition for multiply-parent high patches, low auxiliary classification, text fine-tuning, padding-aware learned uncertainty, or official preprocessing. Each is a separate later ablation, not part of Stage 3.1.

