#!/usr/bin/env bash
set -euo pipefail

# Matched Stage 3.3.8 control: same Stage 3.4 warm-start and training setup,
# with global prototype conditioning disabled.
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/vila_mil_overlay_rt/bin/python}"

OUT="analysis/stage3_model_design/03_low_context_routing/07_global_proto_conditioning_control"
mkdir -p "$OUT"

"$PYTHON_BIN" main.py \
  --data_root_dir data/yiyuan \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --split_dir splits/strict/task_adenocarcinoma_100_k5_s1 \
  --task task_adenocarcinoma \
  --model_type ViLa_MIL_BiomedCLIP \
  --mode transformer \
  --scale_mode high \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --k 5 --k_start 0 --k_end 1 \
  --seed 1 \
  --lr 1e-4 --opt adam --drop_out \
  --max_epochs 80 --patience 10 --early_stopping \
  --exp_code stage338_high_only_warmstart_control_folds01 \
  --results_dir "$OUT" \
  --init_checkpoint_template 'analysis/stage3_model_design/03_low_context_routing/06_stabilization/results/S1_fold{fold}/stage334_S1_stabilized_fold{fold}_s1/s_{fold}_checkpoint.pt' \
  2>&1 | tee "$OUT/train_folds01.log"
