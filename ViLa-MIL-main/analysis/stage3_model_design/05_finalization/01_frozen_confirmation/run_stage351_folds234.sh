#!/usr/bin/env bash
set -euo pipefail

# Stage 3.5.1 frozen confirmation: Routed High vs Clean High on zero-based
# Fold 2, Fold 3, and Fold 4. Each invocation trains exactly one fold.
cd "$(dirname "$0")/../../../.."
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/vila_mil_overlay_rt/bin/python}"

BASE="analysis/stage3_model_design/05_finalization/01_frozen_confirmation"
MAPPING="analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings"
mkdir -p "$BASE/routed_control" "$BASE/clean_high"

COMMON_ARGS=(
  main.py
  --data_root_dir data/yiyuan
  --data_folder_s features_biomedclip_5x
  --data_folder_l features_biomedclip_20x
  --task task_adenocarcinoma
  --split_dir splits/strict/task_adenocarcinoma_100_k5_s1
  --model_type ViLa_MIL_BiomedCLIP
  --mode transformer
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv
  --k 5
  --seed 1
  --lr 1e-4 --reg 1e-5 --opt adam
  --drop_out
  --max_epochs 80 --patience 10 --early_stopping
  --scale_mode high
  --mapping_path "$MAPPING"
  --skip_test_evaluation
)

for FOLD in 2 3 4; do
  echo "[Stage 3.5.1] Arm A: Routed High control, zero-based Fold ${FOLD}"
  "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
    --k_start "$FOLD" --k_end "$FOLD" \
    --use_low_context_routing \
    --use_routing_stabilization \
    --exp_code "stage351_routed_control_fold${FOLD}" \
    --results_dir "$BASE/routed_control/fold${FOLD}" \
    2>&1 | tee "$BASE/routed_control/fold${FOLD}_train.log"
done

for FOLD in 2 3 4; do
  echo "[Stage 3.5.1] Arm B: Clean High candidate, zero-based Fold ${FOLD}"
  "$PYTHON_BIN" "${COMMON_ARGS[@]}" \
    --k_start "$FOLD" --k_end "$FOLD" \
    --exp_code "stage351_clean_high_fold${FOLD}" \
    --results_dir "$BASE/clean_high/fold${FOLD}" \
    2>&1 | tee "$BASE/clean_high/fold${FOLD}_train.log"
done

echo "[Stage 3.5.1] frozen confirmation training complete"
echo "[Stage 3.5.1] all six runs used --skip_test_evaluation"
