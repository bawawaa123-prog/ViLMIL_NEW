# Step 2.6 runbook

Run from the repository root with the same environment used for Stage 1:

```bash
cd /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main
PY=/opt/conda/envs/vila_mil_overlay_rt/bin/python
BASE=analysis/stage2_yiyuan_data_audit/06_scale_ablation
COMMON="--data_root_dir data/yiyuan --data_folder_s features_biomedclip_5x --data_folder_l features_biomedclip_20x --task task_adenocarcinoma --split_dir splits/strict/task_adenocarcinoma_100_k5_s1 --model_type ViLa_MIL_BiomedCLIP --mode transformer --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv --k 5 --k_start 0 --k_end 4 --max_epochs 80 --lr 1e-4 --seed 1 --drop_out --opt adam --exp_code adenocarcinoma_biomedclip_SCALE_strict5 --results_dir $BASE/SCALE --scale_mode MODE"
```

Run low-only and high-only separately. The literal `SCALE` and `MODE` must be replaced as follows:

```bash
# low-only: SCALE=low_only, MODE=low
# high-only: SCALE=high_only, MODE=high
# dual-scale: reuse the verified Stage 1 strict5 result; no retraining is needed.
```

The existing dual result has the matching configuration (`80` epochs, `lr=1e-4`, `seed=1`, `drop_out`, Adam, strict split, dual logits). Create a non-destructive link under the ablation output:

```bash
mkdir -p "$BASE/dual_scale"
ln -sfn "$(pwd)/trained_models/adenocarcinoma_strict5_new/adenocarcinoma_biomedclip_dual_strict5_new_s1" \
  "$BASE/dual_scale/adenocarcinoma_biomedclip_dual_scale_strict5_s1"
```

For a concrete command (low-only), use:

```bash
mkdir -p analysis/stage2_yiyuan_data_audit/06_scale_ablation/low_only
$PY main.py --data_root_dir data/yiyuan --data_folder_s features_biomedclip_5x --data_folder_l features_biomedclip_20x --task task_adenocarcinoma --split_dir splits/strict/task_adenocarcinoma_100_k5_s1 --model_type ViLa_MIL_BiomedCLIP --mode transformer --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv --k 5 --k_start 0 --k_end 4 --max_epochs 80 --lr 1e-4 --seed 1 --drop_out --opt adam --exp_code adenocarcinoma_biomedclip_low_only_strict5 --results_dir analysis/stage2_yiyuan_data_audit/06_scale_ablation/low_only --scale_mode low 2>&1 | tee analysis/stage2_yiyuan_data_audit/06_scale_ablation/low_only/train.log
```

For high-only, first create its log directory, then change `low_only/low` and the experiment code in the command to `high_only/high`:

```bash
mkdir -p analysis/stage2_yiyuan_data_audit/06_scale_ablation/high_only
```

Then summarize:

```bash
$PY analysis/stage2_yiyuan_data_audit/scripts/summarize_scale_ablation.py
```

Each trained single-scale run must return status 0 and contain `fold_summary.csv`, `result.csv`, and five `s_*_checkpoint.pt` files. The dual symlink must resolve to the Stage 1 `fold_summary.csv`. Do not point `--results_dir` at `trained_models/adenocarcinoma_strict5_new` for training.
