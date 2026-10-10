# Stage 3.5.1 frozen Clean High confirmation

Stage 3.5.1 confirms the frozen Stage 3.5.0 architecture on previously unused
zero-based Fold 2, Fold 3, and Fold 4. It keeps the matched two-arm design and
does not search or modify architecture.

## Arms

| Arm | `scale_mode` | `use_low_context_routing` | `use_routing_stabilization` | `use_global_proto_context` | `mapping_path` |
| --- | --- | --- | --- | --- | --- |
| Routed High control | `high` | enabled | enabled | disabled | Stage 3.3.4 mapping |
| Clean High candidate | `high` | disabled | disabled | disabled | same path argument |

Both arms use the same mapping path. Routing OFF remains a model/dataset
configuration already supported by the current repository; this stage does
not remove mapping plumbing or change forward code.

## Frozen protocol

Both arms use the exact Stage 3.5.0 settings:

- `task_adenocarcinoma`
- `features_biomedclip_5x` and `features_biomedclip_20x`
- `splits/strict/task_adenocarcinoma_100_k5_s1`
- `ViLa_MIL_BiomedCLIP`, transformer mode
- `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- seed `1`
- Adam, `lr=1e-4`, `reg=1e-5`
- dropout enabled
- max 80 epochs
- early stopping enabled, repository-forced patience `10`
- independent initialization, no warm start
- `--skip_test_evaluation`

The script uses an explicit shell loop `for FOLD in 2 3 4`. Each invocation
passes `--k_start FOLD --k_end FOLD`, so exactly one zero-based fold is run.
Fold 0/1 are never retrained and Stage 3.5.0 output is not overwritten.

The repository's `validate()` passes `val_error` to `EarlyStopping`, which
selects the checkpoint by validation error / accuracy. AUC is not a selection
criterion.

## Training command

From `ViLa-MIL-main/`:

```bash
bash analysis/stage3_model_design/05_finalization/01_frozen_confirmation/run_stage351_folds234.sh
```

This performs six long runs: Routed and Clean arms for Fold 2, 3, and 4.

## Output layout

```text
analysis/stage3_model_design/05_finalization/01_frozen_confirmation/
  routed_control/
    fold2/stage351_routed_control_fold2_s1/
    fold3/stage351_routed_control_fold3_s1/
    fold4/stage351_routed_control_fold4_s1/
  clean_high/
    fold2/stage351_clean_high_fold2_s1/
    fold3/stage351_clean_high_fold3_s1/
    fold4/stage351_clean_high_fold4_s1/
```

Each result directory should contain `s_FOLD_checkpoint.pt`,
`epoch_details.csv`, `fold_summary.csv`, experiment config, split records, and
its arm/fold log. Test columns must remain empty.

## Read-only summary command

After all six runs complete:

```bash
python tools/stage351_summarize_frozen_confirmation.py
```

The summary tool reads Stage 3.5.0 Fold 0/1 plus Stage 3.5.1 Fold 2/3/4
validation files. It takes Macro-F1 from the actual selected epoch: earliest
maximum `val_acc` in `epoch_details.csv`, matching strict-improvement tie
handling in `EarlyStopping`. It performs no model inference.

Generated aggregate files:

- `stage351_confirmation_folds234.csv`
- `stage351_full5fold_metrics.csv`
- `stage351_full5fold_summary.csv`
- `stage351_selected_epochs.csv`
- `stage351_summary.md`

Do not run formal test evaluation in this stage. Do not commit checkpoint,
feature, cache, or other large binary files; commit scripts, protocol,
experiment configs, validation CSVs, split records, and aggregate summaries.
