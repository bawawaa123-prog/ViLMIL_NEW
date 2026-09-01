# Stage 3.5.0 matched clean High final candidate

Stage 3.5.0 is a validation-only, matched two-arm ablation that asks whether
removing the unsupported Low-to-High routing mechanism preserves the strong
High predictor. It does not redesign the model, add fusion, or modify any
checkpoint.

## Arms

| Arm | `scale_mode` | `use_low_context_routing` | `use_routing_stabilization` | `use_global_proto_context` |
| --- | --- | --- | --- | --- |
| Routed High control | `high` | `True` | `True` | `False` |
| Clean High candidate | `high` | `False` | `False` | `False` |

The clean arm receives the same `mapping_path` argument and uses the same
dataset feature paths, but routing is disabled. In the current code,
`return_mapping` is false when routing is disabled and the model's High path
does not enter `HighRouter`; mapping therefore cannot affect Clean High logits.
No model or dataset code was changed for this ablation.

## Matched protocol

Both arms use exactly the same:

- task: `task_adenocarcinoma`
- features: `features_biomedclip_5x` and `features_biomedclip_20x`
- split: `splits/strict/task_adenocarcinoma_100_k5_s1`
- zero-based folds 0 and 1 (`--k_start 0 --k_end 1`)
- seed `1`
- model: `ViLa_MIL_BiomedCLIP`, transformer mode
- prompt: `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- Adam, `lr=1e-4`, `reg=1e-5`
- dropout enabled
- maximum 80 epochs
- early stopping enabled; repository forces patience to `10`
- checkpoint selection through the existing validation-error/accuracy path
- `--skip_test_evaluation`

The arms are independently initialized. No warm-start checkpoint is used.
The repository's `validate()` passes `val_error` to `EarlyStopping`, so this
matched experiment retains the existing validation-accuracy-based selection;
AUC is not used for checkpoint selection.

## Run command

From `ViLa-MIL-main/`:

```bash
bash analysis/stage3_model_design/05_finalization/00_clean_high_baseline/run_stage350_matched_clean_high.sh
```

Set `PYTHON_BIN` if the default environment path differs.

## Output layout

```text
analysis/stage3_model_design/05_finalization/00_clean_high_baseline/
  routed_control/
    stage350_routed_control_folds01_s1/
      s_0_checkpoint.pt
      s_1_checkpoint.pt
      epoch_details.csv
      fold_summary.csv
      experiment_stage350_routed_control_folds01.txt
      splits_0.csv
      splits_1.csv
    train_folds01.log
  clean_high/
    stage350_clean_high_folds01_s1/
      s_0_checkpoint.pt
      s_1_checkpoint.pt
      epoch_details.csv
      fold_summary.csv
      experiment_stage350_clean_high_folds01.txt
      splits_0.csv
      splits_1.csv
    train_folds01.log
```

The `fold_summary.csv` files contain validation columns; test columns remain
empty because test evaluation is disabled. Do not interpret results until the
two arms have been manually trained and their validation artifacts inspected.
