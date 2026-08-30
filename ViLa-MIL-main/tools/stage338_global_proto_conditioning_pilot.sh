#!/usr/bin/env bash
set -euo pipefail

# Stage 3.3.8 pilot: train Fold 0 and Fold 1 only.
# Keep data, split, seed, optimizer, and epoch settings aligned with the
# existing High-only pilot by overriding only experiment name and conditioning.
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/vila_mil_overlay_rt/bin/python}"

OUT="analysis/stage3_model_design/03_low_context_routing/07_global_proto_conditioning"
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
  --exp_code stage338_global_proto_conditioning_high_folds01 \
  --results_dir "$OUT" \
  --init_checkpoint_template 'analysis/stage3_model_design/03_low_context_routing/06_stabilization/results/S1_fold{fold}/stage334_S1_stabilized_fold{fold}_s1/s_{fold}_checkpoint.pt' \
  --use_global_proto_context \
  2>&1 | tee "$OUT/train_folds01.log"
