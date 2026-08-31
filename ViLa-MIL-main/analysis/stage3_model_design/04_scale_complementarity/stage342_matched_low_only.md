# Stage 3.4.2 matched Low-only control and independent complementarity diagnosis

Stage 3.4.2 trains an independently initialized Low-only predictor on strict5
Fold 0 and Fold 1, then compares it with the original Stage 3.4.0
High-trained checkpoints on the same validation slides. It does not perform
joint dual training or introduce fusion, calibration, routing, or new losses.

## Matched Low-only training

`run_stage342_matched_low_only.sh` copies the Stage 3.4.1 protocol exactly:
the same task, strict split directory, seed 1, feature folders, prompt,
transformer model, prototype count, routing and stabilization flags, Adam,
`lr=1e-4`, `reg=1e-5`, dropout, 80-epoch cap, fixed patience 10, and
the existing validation-based early stopping (repository `EarlyStopping`
receives validation error, equivalent to maximizing validation accuracy; no
AUC selection). The only model/training variable changed is `--scale_mode
low`; `--skip_test_evaluation` is mandatory.

From `ViLa-MIL-main/`:

```bash
bash analysis/stage3_model_design/04_scale_complementarity/run_stage342_matched_low_only.sh
```

The checkpoints are written to:

```text
analysis/stage3_model_design/04_scale_complementarity/stage342_matched_low_only/stage342_matched_low_only_folds01_s1/s_{fold}_checkpoint.pt
```

## Independent validation-only diagnosis

The High template below deliberately points to the original Stage 3.4.0
High-trained Stage 3.3.4 stabilized checkpoints, not the Stage 3.4.1 dual
checkpoints:

```bash
python tools/stage342_independent_complementarity.py \
  --low_checkpoint_template 'analysis/stage3_model_design/04_scale_complementarity/stage342_matched_low_only/stage342_matched_low_only_folds01_s1/s_{fold}_checkpoint.pt' \
  --high_checkpoint_template 'analysis/stage3_model_design/03_low_context_routing/06_stabilization/results/S1_fold{fold}/stage334_S1_stabilized_fold{fold}_s1/s_{fold}_checkpoint.pt' \
  --data_root_dir data/yiyuan \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --mapping_path analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings \
  --splits_dir splits/strict/task_adenocarcinoma_100_k5_s1 \
  --task task_adenocarcinoma \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --folds 0 1 --seed 1 \
  --output_dir analysis/stage3_model_design/04_scale_complementarity/stage342_independent_complementarity
```

Each checkpoint is evaluated on `split=val`. Predictions are merged by
`slide_id`; the script asserts identical ID sets, one-to-one joins, and equal
labels before calculating results. The raw-logit sweep uses the established
definition `alpha * High + (1 - alpha) * Low` for alpha 0.0 through 1.0 and is
diagnostic only, never a learned or selected final fusion.

## Expected artifacts

The output directory contains `checkpoint_manifest.csv`, per-fold and
aggregate `branch_metrics.csv`, `complementarity_summary.csv`,
`alpha_sweep.csv`, `logit_statistics.csv`, and per-fold
`branch_predictions.csv`. The statistics report logit norm and
prediction-margin summaries for both branches.
