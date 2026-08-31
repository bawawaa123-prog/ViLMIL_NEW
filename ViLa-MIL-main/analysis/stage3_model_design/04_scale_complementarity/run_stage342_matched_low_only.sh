#!/usr/bin/env bash
set -euo pipefail

# Stage 3.4.2 matched Low-only control: strict5 Fold 0 and Fold 1.
# This is the Stage 3.4.1 training protocol with scale_mode=low as the only
# experimental change. Test evaluation is disabled for the entire stage.
cd "$(dirname "$0")/../../.."
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/vila_mil_overlay_rt/bin/python}"

BASE="analysis/stage3_model_design/04_scale_complementarity/stage342_matched_low_only"
mkdir -p "$BASE"

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
  --scale_mode low
  --use_low_context_routing
  --use_routing_stabilization
  --mapping_path analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings
  --skip_test_evaluation
  --exp_code stage342_matched_low_only_folds01
  --results_dir "$BASE"
)

echo "[Stage 3.4.2] matched Low-only control: folds 0 and 1"
echo "[Stage 3.4.2] validation-only mode: test evaluation disabled"
"$PYTHON_BIN" "${COMMON[@]}" 2>&1 | tee "$BASE/train_folds01.log"
