# Stage 3.4.1 matched dual-trained baseline

This pilot trains strict5 Fold 0 and Fold 1 with the Stage 3.3.4 stabilized
High-only setup. The only experimental change is `scale_mode=high` to
`scale_mode=dual`; dual forward and loss code are unchanged.

## Matched configuration

| Setting | High-only control | Dual pilot |
| --- | --- | --- |
| Dataset/task | `task_adenocarcinoma`, `dataset_csv/all_data.csv` | same |
| Splits | `splits/strict/task_adenocarcinoma_100_k5_s1`, folds 0/1 | same |
| Seed | `1` | same |
| Features | `data/yiyuan/features_biomedclip_5x` and `features_biomedclip_20x` | same |
| Prompt | `text_prompt/adenocarcinoma_dual_scale_prompt.csv` | same |
| Model | `ViLa_MIL_BiomedCLIP`, transformer, prototype number 16 | same |
| Routing | low-context routing + Stage 3.3.4 stabilization, mapping directory unchanged | same |
| Optimizer | Adam, base `lr=1e-4`, `reg=1e-5` weight decay | same |
| Regularization | dropout enabled, weighted sampling disabled | same |
| Schedule | max 80 epochs, validation early stopping, effective patience 10 | same |
| Scale mode | `high` | **`dual` (only experimental variable)** |
| Test evaluation | control may report test after training | disabled for pilot via `--skip_test_evaluation` |

The pilot output is isolated under
`analysis/stage3_model_design/04_scale_complementarity/stage341_matched_dual_baseline/`
and does not reuse Stage 3.3/3.4.0 output directories.

## Data flow

In `ViLa_MIL_BiomedCLIP.forward`, both branches are always computed:

```text
Low features -> low cross-attention/pooling -> logits_low
High features (with existing routing) -> high cross-attention/pooling -> logits_high
logits_dual = logits_low + logits_high
```

With `scale_mode=dual`, `logits = logits_dual`; the existing
`CrossEntropyLoss(logits, label)` is therefore the optimization loss. With
`scale_mode=high`, only `logits_high` is selected for that same loss. The
branch logits are not redefined and no auxiliary loss or fusion module is
introduced.

## Manual command

From `ViLa-MIL-main/`:

```bash
bash analysis/stage3_model_design/04_scale_complementarity/run_stage341_matched_dual_baseline.sh
```

Set `PYTHON_BIN` if the environment path differs. Do not run a full fold job
from this development turn; the command above is for manual server execution.

After both checkpoints exist, run the existing validation-only complementarity
diagnostic into a separate Stage 3.4.1 directory:

```bash
python tools/stage340_scale_complementarity.py \
  --checkpoint_template 'analysis/stage3_model_design/04_scale_complementarity/stage341_matched_dual_baseline/stage341_matched_dual_baseline_folds01_s1/s_{fold}_checkpoint.pt' \
  --data_root_dir data/yiyuan \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --mapping_path analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings \
  --splits_dir splits/strict/task_adenocarcinoma_100_k5_s1 \
  --task task_adenocarcinoma \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --folds 0 1 --scale_mode dual --seed 1 \
  --output_dir analysis/stage3_model_design/04_scale_complementarity/stage341_matched_dual_baseline/diagnostics
```

## Required post-run artifacts

For each fold, retain the validation-only `epoch_details.csv`,
`fold_summary.csv`, `experiment_*.txt`, `s_{fold}_checkpoint.pt`, and the
complete `train_folds01.log`. Report per-fold best/final validation loss,
AUC, accuracy, macro-F1, stopping epoch, and checkpoint path. For the
scale-complementarity comparison, also run the existing validation-only
Stage 3.4.0 analysis on these dual-trained checkpoints and provide its
`branch_metrics.csv`, `branch_predictions.csv`, `alpha_sweep.csv`, and
`complementarity_summary.csv`; do not use test metrics for model selection.
