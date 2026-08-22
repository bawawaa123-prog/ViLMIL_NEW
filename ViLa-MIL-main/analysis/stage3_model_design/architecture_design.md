# Stage 3.0 Architecture Design

## Recommended MVP

**High-dominant Spatially Aligned Coarse-to-Fine ViLa-MIL** uses the actual approximately 10x branch (`data_folder_l`, nominal `20x`) as the diagnostic evidence branch. The actual approximately 2.5x branch (`data_folder_s`, nominal `5x`) is used to summarize tissue context and route high patches. The final classifier is high-only by default, with a learned low-conditioned residual gate.

The model has four logical stages:

1. Encode low and high 512-D frozen Stage 1 features with small trainable projection/normalization layers.
2. Build an offline coordinate-value mapping from each low bbox to a variable list of overlapping high bboxes. Keep `parent_mask`, `padding_mask`, and `unmapped_high_mask` explicit.
3. Pool each low region into a context vector. Use that vector, plus pathology text semantics, to score high children for evidence selection. High patches with no parent use a separate unparented bucket and are never silently dropped.
4. Classify the selected/pooled high evidence, then add only a gated residual from low context: `z = z_high + alpha * gate(z_low, z_high) * r(z_low)`. `alpha` is initialized small (for example 0.1), so high evidence dominates at initialization.

This is asymmetric conditioning, not two independent classifiers whose logits are summed.

## Coordinate mapping

For every slide, read coordinates stored next to features. Construct continuous level-0 bboxes from the audited per-slide downsample and patch footprint. For low region `s`, define children `C(s)` as all high bboxes with positive overlap (retain overlap ratio and center-distance metadata). A high patch may belong to zero, one, or several low regions. Aggregation uses normalized overlap weights, not a fixed child count. The implementation must not assume `1 low = 16 high`; only about 2.12% of low patches have exactly 16 positive-overlap children, and about 1.92% of high patches have no parent.

## Low context and routing

Low features are first pooled per low region using masked gated attention. A low slide context is then obtained with a second masked attention over low regions. The region context is broadcast to its variable high children. Routing score is:

`route_i = w_r^T tanh(W_h h_i + W_c c_parent(i) + W_t t_cls + W_g g_i)`

where `g_i` contains overlap and padding metadata and `t_cls` is a pathology text semantic vector. Select top-k high patches per low region, with a minimum of one child when a valid child exists; use a global top-k fallback for unparented high patches. The first MVP can use soft weighting instead of hard top-k to keep gradients and implementation simple.

## Semantic guidance

BiomedCLIP pathology text semantics participate at the high-patch evidence-selection step, before high pooling/classification. Keep the existing prompt learner and text encoder. Encode high-resolution prompts once per slide, normalize them, and compute patch-text compatibility after projection. Text does not create a second equal classifier; it supplies a class-aware routing prior. For stability, semantic guidance is detached/frozen in the first MVP unless the existing prompt-learning path is explicitly enabled by the experiment.

## Module disposition

Keep the Stage 1 feature convention, text prompt learner, text encoder, 512-D feature interface, cross-entropy loss, and strict5 protocol. Replace the shared prototype `cross_attention_1` as the main pooling mechanism with explicit masked gated attention and coordinate-aware grouping. Retire `learnable_image_center` from the MVP; it is a global latent query with no spatial meaning and is duplicated across scales. Retire `cross_attention_2` as the primary text-image fusion; use normalized dot-product compatibility for routing. Keep a small residual text projection only if required by checkpoint compatibility. Keep attention pooling, but split it into low-region pooling and routed high pooling with masks.

## Why this MVP

Stage 2 shows High-only AUC `0.9657` versus Low-only `0.9220`; the old Dual AUC gain is only `+0.0055` while Accuracy/F1 decrease. Therefore the first new model must make it possible for low context to help selection without overriding high diagnostic evidence. Coordinate mapping, masks, and a single gated residual are the minimum changes needed to test that hypothesis.

