# Step 2.6: Low-only / High-only / Dual-scale 消融

三组使用相同 strict5 case-level split、seed=1、max 80 epochs、Adam、lr=1e-4、dropout、prompt 与现有 BiomedCLIP features；三组均启用 early stopping，代码固定 patience=10，监控 validation loss。`dual_scale` 复用了已验证的 Stage 1 baseline，不覆盖原 checkpoint。

早停实际 epoch：low-only 为 fold 1--5 = 26/16/15/14/14；high-only = 13/34/14/19/15；Stage 1 dual baseline = 26/18/15/13/12。

## Fold results

### test_auc

| metric   | fold   |   dual_scale |   high_only |   low_only |   dual_minus_low |   dual_minus_high |   high_minus_low |
|:---------|:-------|-------------:|------------:|-----------:|-----------------:|------------------:|-----------------:|
| test_auc | 1      |     0.984612 |    0.982008 |   0.938447 |        0.0461648 |        0.00260417 |        0.0435606 |
| test_auc | 2      |     0.95348  |    0.958688 |   0.882812 |        0.0706676 |       -0.00520833 |        0.0758759 |
| test_auc | 3      |     0.980844 |    0.973792 |   0.959102 |        0.0217417 |        0.00705136 |        0.0146903 |
| test_auc | 4      |     0.973753 |    0.955858 |   0.907898 |        0.0658554 |        0.0178955  |        0.0479599 |
| test_auc | 5      |     0.963493 |    0.958125 |   0.921498 |        0.0419948 |        0.00536865 |        0.0366261 |
| test_auc | mean   |     0.971236 |    0.965694 |   0.921952 |        0.0492848 |        0.00554227 |        0.0437426 |

### test_acc

| metric   | fold   |   dual_scale |   high_only |   low_only |   dual_minus_low |   dual_minus_high |   high_minus_low |
|:---------|:-------|-------------:|------------:|-----------:|-----------------:|------------------:|-----------------:|
| test_acc | 1      |     0.92268  |    0.938144 |   0.902062 |        0.0206186 |       -0.0154639  |        0.0360825 |
| test_acc | 2      |     0.907216 |    0.912371 |   0.809278 |        0.0979381 |       -0.00515464 |        0.103093  |
| test_acc | 3      |     0.958763 |    0.963918 |   0.886598 |        0.0721649 |       -0.00515464 |        0.0773196 |
| test_acc | 4      |     0.943005 |    0.906736 |   0.829016 |        0.11399   |        0.0362694  |        0.0777202 |
| test_acc | 5      |     0.911917 |    0.932642 |   0.854922 |        0.0569948 |       -0.0207254  |        0.0777202 |
| test_acc | mean   |     0.928716 |    0.930762 |   0.856375 |        0.0723412 |       -0.00204583 |        0.0743871 |

### test_f1

| metric   | fold   |   dual_scale |   high_only |   low_only |   dual_minus_low |   dual_minus_high |   high_minus_low |
|:---------|:-------|-------------:|------------:|-----------:|-----------------:|------------------:|-----------------:|
| test_f1  | 1      |     0.915375 |    0.932949 |   0.889677 |        0.0256979 |       -0.0175743  |        0.0432723 |
| test_f1  | 2      |     0.895886 |    0.905303 |   0.797386 |        0.0984993 |       -0.00941785 |        0.107917  |
| test_f1  | 3      |     0.953727 |    0.959664 |   0.873698 |        0.080029  |       -0.00593687 |        0.0859658 |
| test_f1  | 4      |     0.935964 |    0.897122 |   0.799547 |        0.136417  |        0.038842   |        0.0975748 |
| test_f1  | 5      |     0.900252 |    0.926922 |   0.832216 |        0.0680367 |       -0.0266693  |        0.094706  |
| test_f1  | mean   |     0.920241 |    0.924392 |   0.838505 |        0.081736  |       -0.00415126 |        0.0858872 |

## Interpretation

- **Low-only**：test AUC `0.9220 ± 0.0260`，Accuracy `0.8564 ± 0.0346`，Macro-F1 `0.8385 ± 0.0377`。
- **High-only**：test AUC `0.9657 ± 0.0103`，Accuracy `0.9308 ± 0.0204`，Macro-F1 `0.9244 ± 0.0221`。
- **Dual-scale**：test AUC `0.9712 ± 0.0114`，Accuracy `0.9287 ± 0.0194`，Macro-F1 `0.9202 ± 0.0218`。
- High-only 是主要性能来源：相对 low-only，平均 AUC/Accuracy/F1 分别高 `0.0437/0.0744/0.0859`。
- Dual 相对 low-only 在 5/5 folds、全部三项指标上提升；平均 AUC/Accuracy/F1 提升 `0.0493/0.0723/0.0817`。
- Dual 相对 high-only：AUC 平均提升 `0.0055`，但 Accuracy 下降 `0.0020`、Macro-F1 下降 `0.0042`。AUC 仅 4/5 folds 提升，fold 2 下降 `0.0052`；Accuracy 和 F1 仅 fold 4 提升，其余 4 folds 下降。
- 因而 dual **不稳定地全面优于 high-only**。当前结果最支持“High 基本主导，Low 提供有限且指标依赖的补充；简单 `logits_low + logits_high` 存在融合/校准问题”，而不是“两尺度已证明明显互补”。
- 当前评估代码没有输出 sensitivity/specificity，因此本次没有虚构这两项指标。

## Decision

**Step 2.6 通过。** 不需要重新生成 coordinates/features，也不需要否定现有 Stage 1 baseline；但 Stage 1 dual 不是 high-only 的全面稳定改进。由于 Step 2.4 已确认 low/high 存在可靠空间对应，且 low 相对 low-only 的信息在 dual 中仍带来 AUC 小幅增益，下一阶段值得做受控的 spatially aligned cross-scale reasoning；应以 high-only 作为强基线，并重点比较融合校准、尺度权重和逐 patch 对齐带来的增益，避免直接假设简单 logits 相加有效。
