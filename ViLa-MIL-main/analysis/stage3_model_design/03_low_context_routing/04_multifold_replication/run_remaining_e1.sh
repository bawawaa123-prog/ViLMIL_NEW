#!/usr/bin/env bash
set -euo pipefail

# Train only E1 folds 1-4. Fold 0 is the completed pilot and is never
# overwritten. All settings match the fold-0 pilot.
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/vila_mil_overlay_rt/bin/python}"
BASE="analysis/stage3_model_design/03_low_context_routing/04_multifold_replication"
mkdir -p "${BASE}/logs" "${BASE}/results"
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
  --k 5 --max_epochs 80 --lr 1e-4 --seed 1 --drop_out --opt adam
  --early_stopping --scale_mode high
  --use_low_context_routing
  --mapping_path analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings
)

for FOLD in 1 2 3 4; do
  # Skip a completed fold so an interrupted run can be safely resumed.
  EXISTING_DIR="${BASE}/results/E1_fold${FOLD}"
  if find "${EXISTING_DIR}" -type f -name fold_summary.csv -print -quit 2>/dev/null | grep -q .; then
    echo "[E1] fold ${FOLD} already has fold_summary.csv; skipping"
    continue
  fi
  echo "[E1] training fold ${FOLD}"
  "${PYTHON_BIN}" "${COMMON[@]}" \
    --k_start "${FOLD}" --k_end "${FOLD}" \
    --exp_code "stage332_multifold_E1_fold${FOLD}" \
    --results_dir "${BASE}/results/E1_fold${FOLD}" \
    2>&1 | tee "${BASE}/logs/E1_fold${FOLD}.log"
done
