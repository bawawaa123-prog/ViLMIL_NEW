# Stage 3.5.1 Matched Five-fold Baseline Validation

Validation only. Fold indices below are zero-based. No inference or training was performed.

Checkpoint selection: first maximum validation ACC, effective patience=10. Validation Macro-F1 is read from val_f1 at that exact epoch, never from Test. AUC uses P(class 1=NonAdenocarcinoma). Macro-F1 uses sklearn average=macro, zero_division=0. Both validation classes are required. Standard deviation uses ddof=0.

All three arms complete.

| arm | fold | status | val_auc | val_acc | val_macro_f1 | selected_epoch | epochs_run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 0 | Complete | 0.972538 | 0.938144 | 0.932522 | 3 | 13 |
| E0 | 1 | Complete | 0.989070 | 0.963918 | 0.960240 | 24 | 34 |
| E0 | 2 | Complete | 0.972322 | 0.937824 | 0.930409 | 4 | 14 |
| E0 | 3 | Complete | 0.953770 | 0.927461 | 0.919411 | 9 | 19 |
| E0 | 4 | Complete | 0.977746 | 0.932990 | 0.925090 | 5 | 15 |
| E1 | 0 | Complete | 0.967330 | 0.917526 | 0.905974 | 3 | 13 |
| E1 | 1 | Complete | 0.975673 | 0.953608 | 0.947742 | 4 | 14 |
| E1 | 2 | Complete | 0.959795 | 0.932642 | 0.925436 | 10 | 20 |
| E1 | 3 | Complete | 0.962181 | 0.943005 | 0.937766 | 13 | 23 |
| E1 | 4 | Complete | 0.979877 | 0.927835 | 0.920202 | 4 | 14 |
| E2 | 0 | Complete | 0.964134 | 0.922680 | 0.914800 | 12 | 22 |
| E2 | 1 | Complete | 0.975438 | 0.953608 | 0.948518 | 13 | 23 |
| E2 | 2 | Complete | 0.962062 | 0.932642 | 0.923722 | 8 | 18 |
| E2 | 3 | Complete | 0.952517 | 0.943005 | 0.937766 | 5 | 15 |
| E2 | 4 | Complete | 0.984375 | 0.932990 | 0.925638 | 3 | 13 |

## Fivefold statistics

| arm | metric | n | status | mean | std | best_fold | worst_fold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | val_auc | 5 | Fivefold | 0.973089 | 0.011412 | 1 | 3 |
| E0 | val_acc | 5 | Fivefold | 0.940067 | 0.012543 | 1 | 3 |
| E0 | val_macro_f1 | 5 | Fivefold | 0.933534 | 0.014105 | 1 | 3 |
| E1 | val_auc | 5 | Fivefold | 0.968971 | 0.007705 | 4 | 2 |
| E1 | val_acc | 5 | Fivefold | 0.934923 | 0.012430 | 1 | 0 |
| E1 | val_macro_f1 | 5 | Fivefold | 0.927424 | 0.014394 | 1 | 0 |
| E2 | val_auc | 5 | Fivefold | 0.967705 | 0.011072 | 4 | 3 |
| E2 | val_acc | 5 | Fivefold | 0.936985 | 0.010507 | 1 | 0 |
| E2 | val_macro_f1 | 5 | Fivefold | 0.930089 | 0.011770 | 1 | 0 |

## Paired fold differences

| comparison | fold | status | delta_val_auc | delta_val_acc | delta_val_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| E1-E0 | 0 | Complete | -0.005208 | -0.020619 | -0.026548 |
| E1-E0 | 1 | Complete | -0.013398 | -0.010309 | -0.012498 |
| E1-E0 | 2 | Complete | -0.012527 | -0.005181 | -0.004973 |
| E1-E0 | 3 | Complete | 0.008411 | 0.015544 | 0.018355 |
| E1-E0 | 4 | Complete | 0.002131 | -0.005155 | -0.004888 |
| E2-E0 | 0 | Complete | -0.008404 | -0.015464 | -0.017722 |
| E2-E0 | 1 | Complete | -0.013633 | -0.010309 | -0.011722 |
| E2-E0 | 2 | Complete | -0.010260 | -0.005181 | -0.006686 |
| E2-E0 | 3 | Complete | -0.001253 | 0.015544 | 0.018355 |
| E2-E0 | 4 | Complete | 0.006629 | 0.000000 | 0.000548 |
| E1-E2 | 0 | Complete | 0.003196 | -0.005155 | -0.008827 |
| E1-E2 | 1 | Complete | 0.000235 | 0.000000 | -0.000777 |
| E1-E2 | 2 | Complete | -0.002267 | 0.000000 | 0.001714 |
| E1-E2 | 3 | Complete | 0.009664 | 0.000000 | 0.000000 |
| E1-E2 | 4 | Complete | -0.004498 | -0.005155 | -0.005435 |

## Paired fivefold statistics

| comparison | metric | n_pairs | mean_delta | improvement_folds | max_degradation | worst_delta | worst_fold | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1-E0 | val_auc | 5 | -0.004118 | 2 | 0.013398 | -0.013398 | 1 | Fivefold |
| E1-E0 | val_acc | 5 | -0.005144 | 1 | 0.020619 | -0.020619 | 0 | Fivefold |
| E1-E0 | val_macro_f1 | 5 | -0.006110 | 1 | 0.026548 | -0.026548 | 0 | Fivefold |
| E2-E0 | val_auc | 5 | -0.005384 | 1 | 0.013633 | -0.013633 | 1 | Fivefold |
| E2-E0 | val_acc | 5 | -0.003082 | 1 | 0.015464 | -0.015464 | 0 | Fivefold |
| E2-E0 | val_macro_f1 | 5 | -0.003445 | 2 | 0.017722 | -0.017722 | 0 | Fivefold |
| E1-E2 | val_auc | 5 | 0.001266 | 3 | 0.004498 | -0.004498 | 4 | Fivefold |
| E1-E2 | val_acc | 5 | -0.002062 | 0 | 0.005155 | -0.005155 | 4 | Fivefold |
| E1-E2 | val_macro_f1 | 5 | -0.002665 | 1 | 0.008827 | -0.008827 | 0 | Fivefold |

## Provenance and limitations

Source paths/checkpoints/issues: fivefold_metrics.csv. Inputs/current sources: audit_manifest.json. Historical runs have no contemporaneous source/input hashes; semantic config, saved split, checkpoint schema, epoch sequence and final metrics were audited, but historical bitwise provenance cannot be proved.

Strict5 remains a provisional case-level split. Candidate naming groups cross splits; patient-level independence is unconfirmed. Feature directories retain nominal 5x/20x names although Stage2 measured approximately 2.5x/10x. No split or feature was changed.

Stage 3.5.2 can proceed with all 15 validated folds, matched schemas/configurations and E1 diagnostics. This is a validation comparison after prior development on the same folds; it is not an untouched test estimate. Do not tune hyperparameters or select checkpoints using historical Test metrics.
