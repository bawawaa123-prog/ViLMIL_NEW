#!/usr/bin/env bash
set -euo pipefail

# Stage 3.3.2 pilot: fold 0 only. E0 and E1 intentionally share every
# training/split/feature setting; routing is the only experimental variable.
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/vila_mil_overlay_rt/bin/python}"
COMMON=(
  main.py
  --data_root_dir data/yiyuan
  --data_folder_s features_biomedclip_5x
  --data_folder_l features_biomedclip_20x
  --task task_adenocarcinoma
  --split_dir splits/strict/task_adenocarcinoma_100_k5_s1
  --model_type ViLa_MIL_BiomedCLIP
  --mode transformer
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv
  --k 5 --k_start 0 --k_end 0
  --max_epochs 80 --lr 1e-4 --seed 1 --drop_out --opt adam --early_stopping
  --scale_mode high
)
MAPPING="analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings"

echo "[E0] High-only baseline"
"${PYTHON_BIN}" "${COMMON[@]}" \
  --exp_code stage332_pilot_E0_high_only \
  --results_dir analysis/stage3_model_design/03_low_context_routing/03_pilot_validation/results/E0 \
  2>&1 | tee analysis/stage3_model_design/03_low_context_routing/03_pilot_validation/results_E0.log

echo "[E1] High + Low-conditioned routing"
"${PYTHON_BIN}" "${COMMON[@]}" \
  --use_low_context_routing --mapping_path "${MAPPING}" \
  --exp_code stage332_pilot_E1_low_context_routing \
  --results_dir analysis/stage3_model_design/03_low_context_routing/03_pilot_validation/results/E1 \
  2>&1 | tee analysis/stage3_model_design/03_low_context_routing/03_pilot_validation/results_E1.log
