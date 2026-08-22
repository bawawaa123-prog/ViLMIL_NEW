# Step 2.5 execution runbook

The audit is read-only. It never changes feature H5, coordinate H5, WSI, or
model files. The feature scan reads vectors in chunks and the A/B mode loads
BiomedCLIP only for a small controlled patch sample.

## What was already checked

- Python syntax and `git diff --check` passed.
- Metadata-only scan on one slide found valid `[N,512] float32` H5 files.
- Two-patch A/B test completed on `2460239-B2`.
- Current project transform: resize directly to `224 x 224`, bilinear,
  ImageNet mean/std.
- Official transform: resize to 224 with bicubic, center crop 224, CLIP
  mean/std, plus official mode/tensor conversion.
- Sample cosine similarities were `0.98697` and `0.98076`; this is a real
  preprocessing difference, not an exact-equivalence result.

## Full audit command

Run from the project root:

```bash
cd /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main
set -o pipefail

/opt/conda/envs/vila_mil_overlay_rt/bin/python \
  analysis/stage2_yiyuan_data_audit/scripts/audit_biomedclip_features.py \
  --feature-scan \
  --ab-compare \
  --ab-slides 25032146B2 2476358-B2 2485803-B2 2486859-B2 \
  --ab-patches-per-slide 4 \
  --batch-size 8 \
  2>&1 | tee analysis/stage2_yiyuan_data_audit/05_biomedclip_feature_audit/audit_feature.log
feature_status=${PIPESTATUS[0]}
echo "feature_status=${feature_status}"
test "${feature_status}" -eq 0 || exit "${feature_status}"
```

Expected full scan scope is 1,936 feature H5 files and approximately 3.2
million vectors. No feature re-extraction is part of this command.

## Return after completion

Please provide:

- `audit_feature.log`
- `feature_statistics.csv`
- `feature_anomalies.csv`
- `preprocessing_comparison.csv`
- `summary.md`
- `feature_status`

The final decision on baseline trust and Step 2.6 should be made only after
checking the full norm/NaN/Inf scan and the representative A/B distribution.
