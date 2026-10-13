# Stage 3.5.2 Routing diagnostics

完整计划已完成并核验。

默认完整五折/三seed诊断齐全，可进入证据审阅。

只使用 Validation，AUC 正类=1 NonAdenocarcinoma；阈值保持概率 argmax。Gate 饱和阈值 0.05/0.95，边界描述 |P1-P0|≤0.10。

## 干预配对结果

| fold | arm | mode | seed | status | checkpoint | split | n_slides | auc | accuracy | macro_f1 | delta_auc | delta_accuracy | delta_macro_f1 | mean_absolute_probability_change | changed_prediction_count | correct_to_wrong | wrong_to_correct | mean_signed_margin_change | mean_absolute_margin_change | mean_confidence_margin_change | mean_true_class_margin_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | E1 | normal | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | val | 194 | 0.967330 | 0.917526 | 0.905974 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 0 | E1 | context_shuffle | 352 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | val | 194 | 0.967093 | 0.917526 | 0.905974 | -0.000237 | 0.000000 | 0.000000 | 0.000659 | 0 | 0 | 0 | -0.000838 | 0.001318 | -0.000616 | -0.000558 |
| 0 | E1 | context_shuffle | 353 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | val | 194 | 0.967330 | 0.917526 | 0.905974 | 0.000000 | 0.000000 | 0.000000 | 0.000662 | 0 | 0 | 0 | -0.000737 | 0.001324 | -0.000479 | -0.000558 |
| 0 | E1 | context_shuffle | 354 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | val | 194 | 0.967330 | 0.917526 | 0.905974 | 0.000000 | 0.000000 | 0.000000 | 0.000670 | 0 | 0 | 0 | -0.000806 | 0.001339 | -0.000541 | -0.000646 |
| 0 | E1 | context_mean | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | val | 194 | 0.967211 | 0.917526 | 0.905974 | -0.000118 | 0.000000 | 0.000000 | 0.000655 | 0 | 0 | 0 | -0.000814 | 0.001311 | -0.000534 | -0.000564 |
| 0 | E1 | residual_off | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | val | 194 | 0.966974 | 0.917526 | 0.905974 | -0.000355 | 0.000000 | 0.000000 | 0.000954 | 0 | 0 | 0 | -0.000268 | 0.001908 | -0.000713 | -0.001173 |
| 1 | E1 | normal | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | val | 194 | 0.975673 | 0.953608 | 0.947742 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | E1 | context_shuffle | 352 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | val | 194 | 0.975320 | 0.953608 | 0.947742 | -0.000353 | 0.000000 | 0.000000 | 0.001755 | 0 | 0 | 0 | 0.001700 | 0.003511 | -0.001161 | -0.001250 |
| 1 | E1 | context_shuffle | 353 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | val | 194 | 0.975555 | 0.953608 | 0.947742 | -0.000118 | 0.000000 | 0.000000 | 0.001190 | 0 | 0 | 0 | 0.000721 | 0.002380 | -0.000597 | -0.001019 |
| 1 | E1 | context_shuffle | 354 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | val | 194 | 0.975438 | 0.953608 | 0.947742 | -0.000235 | 0.000000 | 0.000000 | 0.001516 | 0 | 0 | 0 | 0.001217 | 0.003032 | -0.000509 | -0.000583 |
| 1 | E1 | context_mean | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | val | 194 | 0.975555 | 0.953608 | 0.947742 | -0.000118 | 0.000000 | 0.000000 | 0.001246 | 0 | 0 | 0 | 0.001292 | 0.002491 | -0.000707 | -0.000774 |
| 1 | E1 | residual_off | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | val | 194 | 0.975320 | 0.953608 | 0.947742 | -0.000353 | 0.000000 | 0.000000 | 0.001786 | 0 | 0 | 0 | 0.001421 | 0.003571 | -0.001852 | -0.001927 |
| 2 | E1 | normal | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | val | 193 | 0.959795 | 0.932642 | 0.925436 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | E1 | context_shuffle | 352 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | val | 193 | 0.962062 | 0.932642 | 0.925436 | 0.002267 | 0.000000 | 0.000000 | 0.004616 | 0 | 0 | 0 | 0.006072 | 0.009231 | -0.003146 | -0.003849 |
| 2 | E1 | context_shuffle | 353 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | val | 193 | 0.961823 | 0.932642 | 0.925436 | 0.002028 | 0.000000 | 0.000000 | 0.005066 | 0 | 0 | 0 | 0.005608 | 0.010131 | -0.003591 | -0.002812 |
| 2 | E1 | context_shuffle | 354 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | val | 193 | 0.962897 | 0.932642 | 0.925436 | 0.003102 | 0.000000 | 0.000000 | 0.004697 | 0 | 0 | 0 | 0.005802 | 0.009394 | -0.003952 | -0.003431 |
| 2 | E1 | context_mean | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | val | 193 | 0.962777 | 0.932642 | 0.925436 | 0.002983 | 0.000000 | 0.000000 | 0.005031 | 0 | 0 | 0 | 0.008320 | 0.010062 | -0.002548 | -0.003123 |
| 2 | E1 | residual_off | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | val | 193 | 0.961942 | 0.932642 | 0.925436 | 0.002147 | 0.000000 | 0.000000 | 0.006813 | 0 | 0 | 0 | 0.012133 | 0.013626 | -0.003622 | -0.004482 |
| 3 | E1 | normal | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | val | 193 | 0.962181 | 0.943005 | 0.937766 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | E1 | context_shuffle | 352 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | val | 193 | 0.962658 | 0.943005 | 0.938164 | 0.000477 | 0.000000 | 0.000399 | 0.004992 | 2 | 1 | 1 | 0.002267 | 0.009983 | -0.002451 | -0.002283 |
| 3 | E1 | context_shuffle | 353 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | val | 193 | 0.963255 | 0.943005 | 0.938164 | 0.001074 | 0.000000 | 0.000399 | 0.006062 | 2 | 1 | 1 | 0.002157 | 0.012123 | -0.002280 | 0.000386 |
| 3 | E1 | context_shuffle | 354 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | val | 193 | 0.962181 | 0.943005 | 0.938164 | 0.000000 | 0.000000 | 0.000399 | 0.006692 | 2 | 1 | 1 | 0.002294 | 0.013384 | -0.003403 | -0.001405 |
| 3 | E1 | context_mean | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | val | 193 | 0.964209 | 0.943005 | 0.938164 | 0.002028 | 0.000000 | 0.000399 | 0.005340 | 2 | 1 | 1 | 0.004318 | 0.010680 | -0.001374 | -0.000219 |
| 3 | E1 | residual_off | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | val | 193 | 0.966237 | 0.937824 | 0.932328 | 0.004056 | -0.005181 | -0.005437 | 0.009801 | 3 | 2 | 1 | 0.008421 | 0.019603 | -0.004635 | -0.001580 |
| 4 | E1 | normal | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | val | 194 | 0.979877 | 0.927835 | 0.920202 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | E1 | context_shuffle | 352 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | val | 194 | 0.979403 | 0.927835 | 0.920202 | -0.000473 | 0.000000 | 0.000000 | 0.001550 | 0 | 0 | 0 | -0.001085 | 0.003101 | -0.000241 | -0.000033 |
| 4 | E1 | context_shuffle | 353 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | val | 194 | 0.979285 | 0.927835 | 0.920202 | -0.000592 | 0.000000 | 0.000000 | 0.002118 | 0 | 0 | 0 | -0.001944 | 0.004237 | 0.000777 | 0.000082 |
| 4 | E1 | context_shuffle | 354 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | val | 194 | 0.979403 | 0.927835 | 0.920202 | -0.000473 | 0.000000 | 0.000000 | 0.001708 | 0 | 0 | 0 | -0.001665 | 0.003415 | 0.000633 | 0.000386 |
| 4 | E1 | context_mean | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | val | 194 | 0.979522 | 0.927835 | 0.920202 | -0.000355 | 0.000000 | 0.000000 | 0.001684 | 0 | 0 | 0 | -0.002153 | 0.003367 | 0.000690 | 0.000470 |
| 4 | E1 | residual_off | 0 | Complete | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | val | 194 | 0.979522 | 0.922680 | 0.914800 | -0.000355 | -0.005155 | -0.005402 | 0.002671 | 1 | 1 | 0 | 0.002963 | 0.005342 | -0.000905 | -0.001513 |

## 训练方案与同权重干预

| fold | comparison | status | n_slides | auc | accuracy | macro_f1 | delta_auc | delta_accuracy | delta_macro_f1 | mean_absolute_probability_change | changed_prediction_count | correct_to_wrong | wrong_to_correct | mean_signed_margin_change | mean_absolute_margin_change | mean_confidence_margin_change | mean_true_class_margin_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | B0_to_B1 | Complete | 194 | 0.967330 | 0.917526 | 0.905974 | -0.005208 | -0.020619 | -0.026548 | 0.082923 | 12 | 8 | 4 | -0.161199 | 0.165847 | 0.016320 | -0.009296 |
| 0 | B1_to_B2 | Complete | 194 | 0.966974 | 0.917526 | 0.905974 | -0.000355 | 0.000000 | 0.000000 | 0.000954 | 0 | 0 | 0 | -0.000268 | 0.001908 | -0.000713 | -0.001173 |
| 0 | B0_to_B2 | Complete | 194 | 0.966974 | 0.917526 | 0.905974 | -0.005563 | -0.020619 | -0.026548 | 0.083005 | 12 | 8 | 4 | -0.161467 | 0.166010 | 0.015607 | -0.010468 |
| 1 | B0_to_B1 | Complete | 194 | 0.975673 | 0.953608 | 0.947742 | -0.013398 | -0.010309 | -0.012498 | 0.058492 | 6 | 4 | 2 | -0.089292 | 0.116984 | -0.062414 | -0.058374 |
| 1 | B1_to_B2 | Complete | 194 | 0.975320 | 0.953608 | 0.947742 | -0.000353 | 0.000000 | 0.000000 | 0.001786 | 0 | 0 | 0 | 0.001421 | 0.003571 | -0.001852 | -0.001927 |
| 1 | B0_to_B2 | Complete | 194 | 0.975320 | 0.953608 | 0.947742 | -0.013750 | -0.010309 | -0.012498 | 0.058680 | 6 | 4 | 2 | -0.087871 | 0.117360 | -0.064266 | -0.060301 |
| 2 | B0_to_B1 | Complete | 193 | 0.959795 | 0.932642 | 0.925436 | -0.012527 | -0.005181 | -0.004973 | 0.059120 | 7 | 4 | 3 | 0.005908 | 0.118240 | -0.046909 | -0.062045 |
| 2 | B1_to_B2 | Complete | 193 | 0.961942 | 0.932642 | 0.925436 | 0.002147 | 0.000000 | 0.000000 | 0.006813 | 0 | 0 | 0 | 0.012133 | 0.013626 | -0.003622 | -0.004482 |
| 2 | B0_to_B2 | Complete | 193 | 0.961942 | 0.932642 | 0.925436 | -0.010379 | -0.005181 | -0.004973 | 0.061914 | 7 | 4 | 3 | 0.018041 | 0.123827 | -0.050531 | -0.066527 |
| 3 | B0_to_B1 | Complete | 193 | 0.962181 | 0.943005 | 0.937766 | 0.008411 | 0.015544 | 0.018355 | 0.038016 | 7 | 2 | 5 | 0.041687 | 0.076032 | 0.028608 | 0.048104 |
| 3 | B1_to_B2 | Complete | 193 | 0.966237 | 0.937824 | 0.932328 | 0.004056 | -0.005181 | -0.005437 | 0.009801 | 3 | 2 | 1 | 0.008421 | 0.019603 | -0.004635 | -0.001580 |
| 3 | B0_to_B2 | Complete | 193 | 0.966237 | 0.937824 | 0.932328 | 0.012467 | 0.010363 | 0.012918 | 0.039911 | 6 | 2 | 4 | 0.050108 | 0.079822 | 0.023973 | 0.046524 |
| 4 | B0_to_B1 | Complete | 194 | 0.979877 | 0.927835 | 0.920202 | 0.002131 | -0.005155 | -0.004888 | 0.039025 | 5 | 3 | 2 | 0.058585 | 0.078050 | -0.023093 | -0.022530 |
| 4 | B1_to_B2 | Complete | 194 | 0.979522 | 0.922680 | 0.914800 | -0.000355 | -0.005155 | -0.005402 | 0.002671 | 1 | 1 | 0 | 0.002963 | 0.005342 | -0.000905 | -0.001513 |
| 4 | B0_to_B2 | Complete | 194 | 0.979522 | 0.922680 | 0.914800 | 0.001776 | -0.010309 | -0.010290 | 0.040322 | 6 | 4 | 2 | 0.061548 | 0.080644 | -0.023998 | -0.024042 |

## 五折干预统计

| mode | seed | metric | n_folds | mean | std | status |
| --- | --- | --- | --- | --- | --- | --- |
| context_mean | 0 | delta_auc | 5 | 0.000884 | 0.001361 | Fivefold |
| context_mean | 0 | delta_accuracy | 5 | 0.000000 | 0.000000 | Fivefold |
| context_mean | 0 | delta_macro_f1 | 5 | 0.000080 | 0.000160 | Fivefold |
| context_mean | 0 | mean_absolute_probability_change | 5 | 0.002791 | 0.001984 | Fivefold |
| context_mean | 0 | changed_prediction_count | 5 | 0.400000 | 0.800000 | Fivefold |
| context_shuffle | 352 | delta_auc | 5 | 0.000336 | 0.001020 | Fivefold |
| context_shuffle | 352 | delta_accuracy | 5 | 0.000000 | 0.000000 | Fivefold |
| context_shuffle | 352 | delta_macro_f1 | 5 | 0.000080 | 0.000160 | Fivefold |
| context_shuffle | 352 | mean_absolute_probability_change | 5 | 0.002714 | 0.001749 | Fivefold |
| context_shuffle | 352 | changed_prediction_count | 5 | 0.400000 | 0.800000 | Fivefold |
| context_shuffle | 353 | delta_auc | 5 | 0.000479 | 0.000947 | Fivefold |
| context_shuffle | 353 | delta_accuracy | 5 | 0.000000 | 0.000000 | Fivefold |
| context_shuffle | 353 | delta_macro_f1 | 5 | 0.000080 | 0.000160 | Fivefold |
| context_shuffle | 353 | mean_absolute_probability_change | 5 | 0.003020 | 0.002152 | Fivefold |
| context_shuffle | 353 | changed_prediction_count | 5 | 0.400000 | 0.800000 | Fivefold |
| context_shuffle | 354 | delta_auc | 5 | 0.000479 | 0.001323 | Fivefold |
| context_shuffle | 354 | delta_accuracy | 5 | 0.000000 | 0.000000 | Fivefold |
| context_shuffle | 354 | delta_macro_f1 | 5 | 0.000080 | 0.000160 | Fivefold |
| context_shuffle | 354 | mean_absolute_probability_change | 5 | 0.003056 | 0.002271 | Fivefold |
| context_shuffle | 354 | changed_prediction_count | 5 | 0.400000 | 0.800000 | Fivefold |
| normal | 0 | delta_auc | 5 | 0.000000 | 0.000000 | Fivefold |
| normal | 0 | delta_accuracy | 5 | 0.000000 | 0.000000 | Fivefold |
| normal | 0 | delta_macro_f1 | 5 | 0.000000 | 0.000000 | Fivefold |
| normal | 0 | mean_absolute_probability_change | 5 | 0.000000 | 0.000000 | Fivefold |
| normal | 0 | changed_prediction_count | 5 | 0.000000 | 0.000000 | Fivefold |
| residual_off | 0 | delta_auc | 5 | 0.001028 | 0.001798 | Fivefold |
| residual_off | 0 | delta_accuracy | 5 | -0.002067 | 0.002532 | Fivefold |
| residual_off | 0 | delta_macro_f1 | 5 | -0.002168 | 0.002655 | Fivefold |
| residual_off | 0 | mean_absolute_probability_change | 5 | 0.004405 | 0.003367 | Fivefold |
| residual_off | 0 | changed_prediction_count | 5 | 0.800000 | 1.166190 | Fivefold |

## 完成状态

| fold | arm | mode | seed | status | issue | checkpoint | val_auc | val_acc | val_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | E0 | normal | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/clean_high/stage350_clean_high_folds01_s1/s_0_checkpoint.pt | 0.972538 | 0.938144 | 0.932522 |
| 0 | E1 | normal | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | 0.967330 | 0.917526 | 0.905974 |
| 0 | E1 | context_shuffle | 352 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | 0.967093 | 0.917526 | 0.905974 |
| 0 | E1 | context_shuffle | 353 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | 0.967330 | 0.917526 | 0.905974 |
| 0 | E1 | context_shuffle | 354 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | 0.967330 | 0.917526 | 0.905974 |
| 0 | E1 | context_mean | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | 0.967211 | 0.917526 | 0.905974 |
| 0 | E1 | residual_off | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_0_checkpoint.pt | 0.966974 | 0.917526 | 0.905974 |
| 1 | E0 | normal | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/clean_high/stage350_clean_high_folds01_s1/s_1_checkpoint.pt | 0.989070 | 0.963918 | 0.960240 |
| 1 | E1 | normal | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | 0.975673 | 0.953608 | 0.947742 |
| 1 | E1 | context_shuffle | 352 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | 0.975320 | 0.953608 | 0.947742 |
| 1 | E1 | context_shuffle | 353 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | 0.975555 | 0.953608 | 0.947742 |
| 1 | E1 | context_shuffle | 354 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | 0.975438 | 0.953608 | 0.947742 |
| 1 | E1 | context_mean | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | 0.975555 | 0.953608 | 0.947742 |
| 1 | E1 | residual_off | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/s_1_checkpoint.pt | 0.975320 | 0.953608 | 0.947742 |
| 2 | E0 | normal | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/clean_high/fold2/stage351_clean_high_fold2_s1/s_2_checkpoint.pt | 0.972322 | 0.937824 | 0.930409 |
| 2 | E1 | normal | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | 0.959795 | 0.932642 | 0.925436 |
| 2 | E1 | context_shuffle | 352 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | 0.962062 | 0.932642 | 0.925436 |
| 2 | E1 | context_shuffle | 353 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | 0.961823 | 0.932642 | 0.925436 |
| 2 | E1 | context_shuffle | 354 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | 0.962897 | 0.932642 | 0.925436 |
| 2 | E1 | context_mean | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | 0.962777 | 0.932642 | 0.925436 |
| 2 | E1 | residual_off | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold2/stage351_routed_control_fold2_s1/s_2_checkpoint.pt | 0.961942 | 0.932642 | 0.925436 |
| 3 | E0 | normal | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/clean_high/fold3/stage351_clean_high_fold3_s1/s_3_checkpoint.pt | 0.953770 | 0.927461 | 0.919411 |
| 3 | E1 | normal | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | 0.962181 | 0.943005 | 0.937766 |
| 3 | E1 | context_shuffle | 352 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | 0.962658 | 0.943005 | 0.938164 |
| 3 | E1 | context_shuffle | 353 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | 0.963255 | 0.943005 | 0.938164 |
| 3 | E1 | context_shuffle | 354 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | 0.962181 | 0.943005 | 0.938164 |
| 3 | E1 | context_mean | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | 0.964209 | 0.943005 | 0.938164 |
| 3 | E1 | residual_off | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold3/stage351_routed_control_fold3_s1/s_3_checkpoint.pt | 0.966237 | 0.937824 | 0.932328 |
| 4 | E0 | normal | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/clean_high/fold4/stage351_clean_high_fold4_s1/s_4_checkpoint.pt | 0.977746 | 0.932990 | 0.925090 |
| 4 | E1 | normal | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | 0.979877 | 0.927835 | 0.920202 |
| 4 | E1 | context_shuffle | 352 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | 0.979403 | 0.927835 | 0.920202 |
| 4 | E1 | context_shuffle | 353 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | 0.979285 | 0.927835 | 0.920202 |
| 4 | E1 | context_shuffle | 354 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | 0.979403 | 0.927835 | 0.920202 |
| 4 | E1 | context_mean | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | 0.979522 | 0.927835 | 0.920202 |
| 4 | E1 | residual_off | 0 | Complete |  | /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main/analysis/stage3_model_design/05_finalization/01_frozen_confirmation/routed_control/fold4/stage351_routed_control_fold4_s1/s_4_checkpoint.pt | 0.979522 | 0.922680 | 0.914800 |

## 判定边界

- Q1 空间依赖：当前请求范围有相应实验输出，需结合连续量审阅，不能自动确认为机制。仅当 Normal 与 Shuffle/Mean 的配对结果齐全才解释空间敏感性。小差异不证明 Low 无信息或 Router 未用 Low。
- Q2 Patch 区分性：当前请求范围有相应实验输出，需结合连续量审阅，不能自动确认为机制。结合有效 mapped Gate 方差、饱和比例、父节点内/间统计和 Low 空间统计；重叠父组不是互斥 ANOVA 分组。
- Q3 Residual 贡献：当前请求范围有相应实验输出，需结合连续量审阅，不能自动确认为机制。结合逐 patch 与 WSI 范数、cap 触发、B1_to_B2 预测变化。
- Q4 共适应：当前请求范围有相应实验输出，需结合连续量审阅，不能自动确认为机制。B0_to_B2 与共享训练参数差异是线索；独立训练及 Checkpoint epoch 不同，不能确定唯一因果。
- Q5 Fold 差异：当前请求范围有相应实验输出，需结合连续量审阅，不能自动确认为机制。查看 correctness_group_statistics、prediction_changes、class_metrics 与 ranking_changes；Fold3/4 不按结果挑选阈值。
- Q6 下一步：当前请求范围有相应实验输出，需结合连续量审阅，不能自动确认为机制。按配对敏感性和病例证据排序待验证瓶颈，不实现新结构、不按诊断结果改训练超参数。

数据表保存具体证据，报告不自动将相关性判为因果。Gate 不是病灶概率。五折训练集重叠、单 seed，未进行独立样本显著性检验。

case-level strict5 尚未证明 patient-level 独立；名义 5x/20x 实际约 2.5x/10x；验证折参与开发。Checkpoint 身份按路径/大小/mtime 核对，未全量 hash 大权重或特征。
