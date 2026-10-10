# Stage 3.5.1 Matched Five-fold Baseline Validation

Validation only. Fold indices below are zero-based. No inference or training was performed.

Checkpoint selection: first maximum validation ACC, effective patience=10. Validation Macro-F1 is read from val_f1 at that exact epoch, never from Test. AUC uses P(class 1=NonAdenocarcinoma). Macro-F1 uses sklearn average=macro, zero_division=0. Both validation classes are required. Standard deviation uses ddof=0.

PENDING: the complete three-arm comparison is unavailable.

| arm | fold | status | val_auc | val_acc | val_macro_f1 | selected_epoch | epochs_run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 0 | Complete | 0.972538 | 0.938144 | 0.932522 | 3.000000 | 13.000000 |
| E0 | 1 | Complete | 0.989070 | 0.963918 | 0.960240 | 24.000000 | 34.000000 |
| E0 | 2 | Complete | 0.972322 | 0.937824 | 0.930409 | 4.000000 | 14.000000 |
| E0 | 3 | Complete | 0.953770 | 0.927461 | 0.919411 | 9.000000 | 19.000000 |
| E0 | 4 | Complete | 0.977746 | 0.932990 | 0.925090 | 5.000000 | 15.000000 |
| E1 | 0 | Complete | 0.967330 | 0.917526 | 0.905974 | 3.000000 | 13.000000 |
| E1 | 1 | Complete | 0.975673 | 0.953608 | 0.947742 | 4.000000 | 14.000000 |
| E1 | 2 | Complete | 0.959795 | 0.932642 | 0.925436 | 10.000000 | 20.000000 |
| E1 | 3 | Complete | 0.962181 | 0.943005 | 0.937766 | 13.000000 | 23.000000 |
| E1 | 4 | Complete | 0.979877 | 0.927835 | 0.920202 | 4.000000 | 14.000000 |
| E2 | 0 | Pending | Pending | Pending | Pending | Pending | Pending |
| E2 | 1 | Pending | Pending | Pending | Pending | Pending | Pending |
| E2 | 2 | Pending | Pending | Pending | Pending | Pending | Pending |
| E2 | 3 | Pending | Pending | Pending | Pending | Pending | Pending |
| E2 | 4 | Pending | Pending | Pending | Pending | Pending | Pending |

## Fivefold statistics

| arm | metric | n | status | mean | std | best_fold | worst_fold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | val_auc | 5 | Fivefold | 0.973089 | 0.011412 | 1.000000 | 3.000000 |
| E0 | val_acc | 5 | Fivefold | 0.940067 | 0.012543 | 1.000000 | 3.000000 |
| E0 | val_macro_f1 | 5 | Fivefold | 0.933534 | 0.014105 | 1.000000 | 3.000000 |
| E1 | val_auc | 5 | Fivefold | 0.968971 | 0.007705 | 4.000000 | 2.000000 |
| E1 | val_acc | 5 | Fivefold | 0.934923 | 0.012430 | 1.000000 | 0.000000 |
| E1 | val_macro_f1 | 5 | Fivefold | 0.927424 | 0.014394 | 1.000000 | 0.000000 |
| E2 | val_auc | 0 | Pending (no fivefold statistic) | Pending | Pending | Pending | Pending |
| E2 | val_acc | 0 | Pending (no fivefold statistic) | Pending | Pending | Pending | Pending |
| E2 | val_macro_f1 | 0 | Pending (no fivefold statistic) | Pending | Pending | Pending | Pending |

## Paired fold differences

| comparison | fold | status | delta_val_auc | delta_val_acc | delta_val_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| E1-E0 | 0 | Complete | -0.005208 | -0.020619 | -0.026548 |
| E1-E0 | 1 | Complete | -0.013398 | -0.010309 | -0.012498 |
| E1-E0 | 2 | Complete | -0.012527 | -0.005181 | -0.004973 |
| E1-E0 | 3 | Complete | 0.008411 | 0.015544 | 0.018355 |
| E1-E0 | 4 | Complete | 0.002131 | -0.005155 | -0.004888 |
| E2-E0 | 0 | Pending | Pending | Pending | Pending |
| E2-E0 | 1 | Pending | Pending | Pending | Pending |
| E2-E0 | 2 | Pending | Pending | Pending | Pending |
| E2-E0 | 3 | Pending | Pending | Pending | Pending |
| E2-E0 | 4 | Pending | Pending | Pending | Pending |
| E1-E2 | 0 | Pending | Pending | Pending | Pending |
| E1-E2 | 1 | Pending | Pending | Pending | Pending |
| E1-E2 | 2 | Pending | Pending | Pending | Pending |
| E1-E2 | 3 | Pending | Pending | Pending | Pending |
| E1-E2 | 4 | Pending | Pending | Pending | Pending |

## Paired fivefold statistics

| comparison | metric | n_pairs | mean_delta | improvement_folds | max_degradation | worst_delta | worst_fold | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1-E0 | val_auc | 5 | -0.004118 | 2.000000 | 0.013398 | -0.013398 | 1.000000 | Fivefold |
| E1-E0 | val_acc | 5 | -0.005144 | 1.000000 | 0.020619 | -0.020619 | 0.000000 | Fivefold |
| E1-E0 | val_macro_f1 | 5 | -0.006110 | 1.000000 | 0.026548 | -0.026548 | 0.000000 | Fivefold |
| E2-E0 | val_auc | 0 | Pending | Pending | Pending | Pending | Pending | Pending |
| E2-E0 | val_acc | 0 | Pending | Pending | Pending | Pending | Pending | Pending |
| E2-E0 | val_macro_f1 | 0 | Pending | Pending | Pending | Pending | Pending | Pending |
| E1-E2 | val_auc | 0 | Pending | Pending | Pending | Pending | Pending | Pending |
| E1-E2 | val_acc | 0 | Pending | Pending | Pending | Pending | Pending | Pending |
| E1-E2 | val_macro_f1 | 0 | Pending | Pending | Pending | Pending | Pending | Pending |

## Provenance and limitations

Source paths/checkpoints/issues: fivefold_metrics.csv. Inputs/current sources: audit_manifest.json. Historical runs have no contemporaneous source/input hashes; semantic config, saved split, checkpoint schema, epoch sequence and final metrics were audited, but historical bitwise provenance cannot be proved.

Strict5 remains a provisional case-level split. Candidate naming groups cross splits; patient-level independence is unconfirmed. Feature directories retain nominal 5x/20x names although Stage2 measured approximately 2.5x/10x. No split or feature was changed.

Stage 3.5.2 can proceed with all 15 validated folds, matched schemas/configurations and E1 diagnostics. This is a validation comparison after prior development on the same folds; it is not an untouched test estimate. Do not tune hyperparameters or select checkpoints using historical Test metrics.

## Missing/invalid artifacts

| arm | fold | status | source | issue |
| --- | --- | --- | --- | --- |
| E2 | 0 | Pending | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_matched_fivefold/runs/E2/fold0/stage351_matched_E2_fold0_s1 | No training artifacts |
| E2 | 1 | Pending | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_matched_fivefold/runs/E2/fold1/stage351_matched_E2_fold1_s1 | No training artifacts |
| E2 | 2 | Pending | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_matched_fivefold/runs/E2/fold2/stage351_matched_E2_fold2_s1 | No training artifacts |
| E2 | 3 | Pending | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_matched_fivefold/runs/E2/fold3/stage351_matched_E2_fold3_s1 | No training artifacts |
| E2 | 4 | Pending | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_matched_fivefold/runs/E2/fold4/stage351_matched_E2_fold4_s1 | No training artifacts |
