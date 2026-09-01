#!/usr/bin/env bash
set -euo pipefail

# Stage 3.5.0 matched two-arm ablation: Routed High control vs Clean High.
# The arms share the complete Stage 3.3.4/3.4.x training protocol. The only
# experimental variable is Low->High routing ON versus OFF.
cd "$(dirname "$0")/../../../.."
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/vila_mil_overlay_rt/bin/python}"

BASE="analysis/stage3_model_design/05_finalization/00_clean_high_baseline"
MAPPING="analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings"
mkdir -p "$BASE/routed_control" "$BASE/clean_high"

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
  --k 5 --k_start 0 --k_end 1
  --seed 1
  --lr 1e-4 --reg 1e-5 --opt adam
  --drop_out
  --max_epochs 80 --patience 10 --early_stopping
  --scale_mode high
  --mapping_path "$MAPPING"
  --skip_test_evaluation
)

echo "[Stage 3.5.0] Arm A: Routed High control, folds 0 and 1"
"$PYTHON_BIN" "${COMMON[@]}" \
  --use_low_context_routing \
  --use_routing_stabilization \
  --exp_code stage350_routed_control_folds01 \
  --results_dir "$BASE/routed_control" \
  2>&1 | tee "$BASE/routed_control/train_folds01.log"

echo "[Stage 3.5.0] Arm B: Clean High candidate, folds 0 and 1"
"$PYTHON_BIN" "${COMMON[@]}" \
  --exp_code stage350_clean_high_folds01 \
  --results_dir "$BASE/clean_high" \
  2>&1 | tee "$BASE/clean_high/train_folds01.log"

echo "[Stage 3.5.0] matched two-arm training complete"
echo "[Stage 3.5.0] both arms used --skip_test_evaluation"
