# Step 2.5: BiomedCLIP Feature and Preprocessing Audit

## Code-path conclusion
- Checkpoint: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`).
- Image path: `model.encode_image(batch)`.
- Current project preprocessing: `Resize((224,224))`, `ToTensor()`, ImageNet mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)`.
- The official preprocess returned by `create_model_from_pretrained()` is intentionally discarded by the current extractor.
- Current extractor does not L2-normalize image features. The feature H5 stores raw encoder outputs.

## Scan status
- Feature H5 pairs inspected: 1936; shape-valid records: 1936.
- Full value scan requested: **True**; A/B inference requested: **True**.
- Anomaly records: **0**.
- All 1,936 files are `[N,512] float32`; total vectors are 3,208,183 (5x=210,489; 20x=2,997,694).
- NaN values: 0; Inf values: 0; zero vectors: 0.
- 5x per-slide median norm: median 79.8274, range 74.3781-84.4194.
- 20x per-slide median norm: median 80.7096, range 73.3658-84.7222.
- Norm variation is continuous and slide-dependent; no isolated numerical failure was detected.

## Preprocessing A/B result
- Controlled sample: 16 identical WSI patches from four representative slides.
- Current-vs-official cosine similarity: median 0.9830, range 0.9734-0.9903.
- Embedding L2 distance: median 14.4809, range 11.3354-17.3498.
- Current norm: median 75.4531; official norm: median 76.5613.
- The difference is reproducible and is caused by interpolation/crop and normalization differences; it is not an I/O or numerical-corruption issue.

## Interpretation and decision
- The current pipeline is **not official BiomedCLIP preprocessing**. It is a project-specific ImageNet-normalized direct resize pipeline.
- The difference is **meaningful but not catastrophic** in the controlled sample: embeddings remain highly similar (cosine >=0.9734) but are not interchangeable with official-preprocess embeddings.
- No evidence supports re-extracting features for numerical corruption, shape failure, or invalid values.
- A full official-preprocess baseline is recommended as a separate experiment if strict comparability with official BiomedCLIP is required; it should not silently replace the current baseline.
- The current Stage 1 baseline remains usable as a coherent project baseline, provided its preprocessing convention is documented and kept fixed.
- Step 2.6 scale ablation may proceed using the current baseline; any official-preprocess comparison should be treated as a separate sensitivity/baseline branch.

## Outputs
- `feature_statistics.csv`, `feature_anomalies.csv`, `preprocessing_comparison.csv`, `summary.md`, and `figures/`.
