# Stage 3.3.2 Pilot Validation

## Purpose

This directory prepares, but does not execute, a single-fold pilot comparing:

- **E0**: `scale_mode=high`, `use_low_context_routing=False`
- **E1**: identical configuration with `use_low_context_routing=True` and the
  Stage 3.1 mapping directory enabled

Fold 0, seed 1, split, feature H5 files, optimizer, LR, dropout, early
stopping (`--early_stopping`, patience fixed by the current code to 10), text
prompt, and `scale_mode=high` are shared. Routing is the only
core variable. No Stage 3.4 fusion, semantic guidance, hard top-k, or new loss
was added.

## Diagnostics

`HighRouter` stores only scalar diagnostics after each forward:

- route mean/std/min/max
- residual norm and routed-vs-original high feature change norm
- mapped, unmapped, and total high counts

`utils/core_utils.py` writes these to E1's
`routing_diagnostics.jsonl` after optimizer steps 0-4 and every 25th step, and
for the first three validation bags. After backward it also records
`context_projection_grad_norm` and `route_score_grad_norm`, plus phase, epoch,
step, and loss. No patch-level tensors are serialized. E0 produces no routing
diagnostic file.

The first steps should show context projection gradients first; route MLP
gradients are recorded independently. Pilot review must check finite values,
non-zero residual/change after learning, and route statistics not remaining
at the neutral 0.5 indefinitely.

## Commands

Run from the repository root. The script is:

```bash
bash analysis/stage3_model_design/03_low_context_routing/03_pilot_validation/run_pilot.sh
```

Equivalent explicit commands:

```bash
/opt/conda/envs/vila_mil_overlay_rt/bin/python main.py \
  --data_root_dir data/yiyuan \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --task task_adenocarcinoma \
  --split_dir splits/strict/task_adenocarcinoma_100_k5_s1 \
  --model_type ViLa_MIL_BiomedCLIP --mode transformer \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --k 5 --k_start 0 --k_end 0 --max_epochs 80 \
  --lr 1e-4 --seed 1 --drop_out --opt adam --early_stopping --scale_mode high \
  --exp_code stage332_pilot_E0_high_only \
  --results_dir analysis/stage3_model_design/03_low_context_routing/03_pilot_validation/results/E0 \
  2>&1 | tee analysis/stage3_model_design/03_low_context_routing/03_pilot_validation/results_E0.log
```

```bash
/opt/conda/envs/vila_mil_overlay_rt/bin/python main.py \
  --data_root_dir data/yiyuan \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --task task_adenocarcinoma \
  --split_dir splits/strict/task_adenocarcinoma_100_k5_s1 \
  --model_type ViLa_MIL_BiomedCLIP --mode transformer \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --k 5 --k_start 0 --k_end 0 --max_epochs 80 \
  --lr 1e-4 --seed 1 --drop_out --opt adam --early_stopping --scale_mode high \
  --use_low_context_routing \
  --mapping_path analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings \
  --exp_code stage332_pilot_E1_low_context_routing \
  --results_dir analysis/stage3_model_design/03_low_context_routing/03_pilot_validation/results/E1 \
  2>&1 | tee analysis/stage3_model_design/03_low_context_routing/03_pilot_validation/results_E1.log
```

Outputs are isolated under `03_pilot_validation/results/E0` and `results/E1`.
E1 diagnostics are in `results/E1/.../routing_diagnostics.jsonl` (the exact
experiment subdirectory is printed by `main.py`); logs are
`results_E0.log` and `results_E1.log`.

## Smoke checks completed

- Python syntax check for modified modules and pilot scripts passed.
- Existing Stage 3.3.2 four-test routing suite passed.
- `collate_tranformer` mapping round-trip remains covered.
- Real slide `2460239-B2` routing forward smoke remains finite.
- Routing-disabled real baseline checkpoint strict load remains compatible.
- No pilot training was launched by Codex.

Stage 3.4 must wait until these E0/E1 artifacts are reviewed and the pilot
result is replicated sufficiently for a design decision.

## Pilot Results (fold 0, seed 1)

Both runs completed with `--early_stopping` and stopped at epoch 13. The
reported final checkpoint/evaluation was:

| Experiment | Test AUC | Test Accuracy | Test Macro-F1 | Val AUC | Val Accuracy | Val Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| E0 High-only | 0.9820 | 0.9381 | 0.9329 | 0.9725 | 0.9381 | 0.9325 |
| E1 High + routing | 0.9869 | 0.9485 | 0.9438 | 0.9734 | 0.9330 | 0.9267 |

Relative to E0 on this single fold, E1 improved test AUC by `+0.0049`, test
accuracy by `+0.0103`, and test Macro-F1 by `+0.0108`. Validation AUC was nearly
unchanged (`+0.0008`), while the final validation accuracy/F1 were lower. This
is encouraging evidence that the route can learn, not a statistically
validated generalization claim; multi-fold replication is required.

### Routing diagnostics observed

- Step 0: `context_projection_grad_norm=1.907`,
  `route_score_grad_norm=0`, residual norm `0`, route mean about `0.497`.
- Step 1 onward: route MLP gradients became non-zero; this is expected because
  the zero-initialized context projection gives the route-score branch no
  useful gradient on the exact initial step.
- By later training samples, route mean moved to roughly `0.18-0.22` with
  non-zero standard deviation, so it did not remain fixed at `0.5`.
- Routing residual/change norm became non-zero immediately after the first
  optimizer update and reached roughly `5e2-1.4e3` in sampled bags.
- All sampled diagnostics were finite. Sampled mapped ratios ranged from about
  `0.840` to `1.000`; unmapped high rows were retained explicitly.

The E1 log reports approximately 14.38 minutes for the fold; E0 took about
13.27 minutes. The vectorized context implementation is now used by future
runs. These timing values are not a controlled benchmark because they were
collected in separate processes.

### Current decision

Stage 3.3.2 pilot passed its learning/finite-gradient sanity check. Do not
implement Stage 3.4 solely from this result. Before Stage 3.4, preserve these
E0/E1 artifacts and preferably replicate the comparison across the remaining
folds or at least a pre-declared small multi-fold pilot.
