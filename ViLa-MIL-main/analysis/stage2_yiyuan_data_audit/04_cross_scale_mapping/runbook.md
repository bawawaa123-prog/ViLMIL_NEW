# Step 2.4 execution runbook

The mapping audit is read-only. It reads coordinate H5 arrays and Step 2.2
metadata, never WSI pixels, feature matrices, or source files. The mapping is
based on level-0 continuous bboxes and coordinate values, never H5 row order.

The script was syntax-checked and tested on one slide and ten slides. The
ten-slide test took approximately 6 seconds. Full processing is expected to
take approximately 9-10 minutes and writes a high-to-low CSV with roughly 3
million rows, so full execution is intentionally left to the user.

Run from the project root:

```bash
cd /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main
set -o pipefail

/opt/conda/envs/vila_mil_overlay_rt/bin/python \
  analysis/stage2_yiyuan_data_audit/scripts/audit_cross_scale_mapping.py \
  2>&1 | tee analysis/stage2_yiyuan_data_audit/04_cross_scale_mapping/audit_mapping.log
mapping_status=${PIPESTATUS[0]}
echo "mapping_status=${mapping_status}"
test "${mapping_status}" -eq 0 || exit "${mapping_status}"
```

Environment used for checks:

- `/opt/conda/envs/vila_mil_overlay_rt/bin/python`
- Python 3.12.6
- NumPy 2.5.0
- h5py 3.12.1
- OpenSlide Python 1.3.1

After completion, provide:

- `audit_mapping.log`
- `mapping_statistics.csv`
- `low_to_high_mapping.csv`
- `high_to_low_statistics.csv`
- `unmapped_regions.csv`
- `padding_severity.csv`
- `mapping_anomalies.csv`
- `summary.md`
- the `figures/` directory or its generated PNG names
- `mapping_status`

Do not start later model changes. Step 2.4 should be interpreted from the
full-run `summary.md`, especially the low-with-child ratio, high-with-parent
ratio, strict 16-child ratio, unmapped counts, anomaly records, and padding
severity buckets.
