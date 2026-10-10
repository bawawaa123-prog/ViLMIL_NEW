# Stage 3.5.1 — Matched Five-fold Baseline Validation

本目录只建立 E0/E1/E2 的匹配验证体系，不修改模型、Prompt、损失、超参数、strict5 或特征。所有正式训练和批量验证均由研究者手动运行。脚本默认 dry-run，只有明确传入 `--execute` 才执行训练/推理。

## 核查结论（2026-10-09）

实际服务器状态比任务背景更新：E0、E1 的 Fold 0～4 均已完成，不需要再次训练。

| 实验 | Fold 0/1 来源 | Fold 2/3/4 来源 | 当前状态 |
|---|---|---|---|
| E0 Clean High | `../00_clean_high_baseline/clean_high/stage350_clean_high_folds01_s1/` | `../01_frozen_confirmation/clean_high/fold{f}/stage351_clean_high_fold{f}_s1/` | 五折通过产物核查 |
| E1 Stabilized Routed High | `../00_clean_high_baseline/routed_control/stage350_routed_control_folds01_s1/` | `../01_frozen_confirmation/routed_control/fold{f}/stage351_routed_control_fold{f}_s1/` | 五折通过产物核查 |
| E2 Clean Dual | 无本阶段批准复用的结果 | 无 | Fold 0～4 Pending |

历史 Stage2 Dual 不作为 E2。Stage3.4.1 Dual 使用 Routing ON，也不是 Clean Dual，不混用。

当前验证集五折结果（标准差 ddof=0）：

| 实验 | Val AUC | Val ACC | Val Macro-F1 |
|---|---|---|---|
| E0 | 0.973089 ± 0.011412 | 0.940067 ± 0.012543 | 0.933534 ± 0.014105 |
| E1 | 0.968971 ± 0.007705 | 0.934923 ± 0.012430 | 0.927424 ± 0.014394 |
| E2 | Pending | Pending | Pending |

E1−E0 的平均差分别为 −0.004118/−0.005144/−0.006110；提升折数为 AUC 2/5、ACC 1/5、Macro-F1 1/5。最大单折退化分别为 0.013398/0.020619/0.026548。当前结果不支持稳定化 Routing 稳定改善 Clean High，但三组结论必须等待 E2。

## 实际代码和协议

核查范围：Stage3.5.0 protocol/训练脚本及两折产物、现有 Stage3.5.1 frozen-confirmation 脚本与六次训练产物；`main.py`、`utils/core_utils.py`、`utils/utils.py`、`datasets/dataset_generic.py`、`models/model_ViLa_MIL_BiomedCLIP.py`、`models/cross_scale_modules.py`；Stage2 身份、倍率及尺度消融记录；Stage3.4 尺度互补性和 matched-dual 记录。

| 项目 | 三组一致的设置 |
|---|---|
| 数据/CSV | `data/yiyuan` / `dataset_csv/all_data.csv`，968 slides |
| strict5 | `splits/strict/task_adenocarcinoma_100_k5_s1`，原文件不变 |
| 特征 | `features_biomedclip_5x` / `features_biomedclip_20x`，512 维 |
| 模型/Prompt | `ViLa_MIL_BiomedCLIP` / `text_prompt/adenocarcinoma_dual_scale_prompt.csv` |
| 原型/隐藏维度 | 16 / 192 |
| 种子/初始化 | 每折重新 seed=1；独立初始化，无 warm start |
| 优化器 | Adam，base lr=1e-4，weight decay=1e-5 |
| Prompt lr | 原代码自动派生 **1e-3**，不是 base lr；文本编码器冻结 |
| 其他策略 | CE，weighted sampling OFF，保留 `--drop_out` 和原数据加载行为 |
| Epoch/早停 | max 80；实际强制 patience=10，stop_epoch=0 |
| Checkpoint | 最小验证错误率，即验证 ACC 首次达到最高值的 epoch |
| Test | `--skip_test_evaluation`；输出 Test 字段必须为空 |

| 实验 | scale_mode | low_context_routing | routing_stabilization | global_proto_context |
|---|---|---|---|---|
| E0 | high | OFF | OFF | OFF |
| E1 | high | ON | ON，rho=0.10 | OFF |
| E2 | dual | OFF | OFF | OFF |

代码注意事项：

- `--k_start` 与 `--k_end` 均为零基索引，`k_end` 包含终点；单折传入相同 start/end。
- EarlyStopping 的参数和日志虽然叫 `val_loss`，`validate()` 实际传入的是 **val_error**。相同 ACC 不覆盖旧 Checkpoint；不是选最低 CE，也不是最高 AUC。旧 Stage2 summary 中“监控 validation loss”的文字不能替代实际代码。
- `epoch_details.csv` 保存每个 epoch 的 `val_f1`。用最早的最大 `val_acc` 选出记录，再核对 `fold_summary.csv` 的最终 val_auc/val_acc，可获取选中 Checkpoint 的验证 Macro-F1，无需新推理。
- 原训练代码创建 ReduceLROnPlateau，但没有调用 `scheduler.step()`；本阶段保留该实际行为。
- `--drop_out` 保持 ON，但当前 BiomedCLIP 构造分支未将该参数传入模型，不能仅凭 help 中的文字断言新增 MIL dropout 为 0.25。保留模型实际行为，包括原预训练文本模块训练态行为，不在本阶段修复。
- Routing 初始化保存/恢复 **CPU RNG**；新模块由 CPU 创建。测试运行真实模型构造器和真实 Router，仅替换外部预训练加载/文本包装，确认 E0/E1/E2 共有参数和后续 RNG 一致。该测试不证明历史训练当时的初始参数逐位一致，也不代替完整 BiomedCLIP 前向验证。
- 模型在 high 模式仍计算 Low 分支；E0 不通过 Low Routing 或 global context 改变 High logits，保留原实现，不做计算优化。

## 产物核查和限制

`preflight.py`/`common.py` 核对完整配置中已保存的字段、Checkpoint 非空及模型 schema、原型形状、划分副本、训练结束日志、epoch 连续性、patience=10/80 epoch 完成证据、选中 epoch 与最终 AUC/ACC 一致、Test 字段为空及 E1 diagnostics 存在。跨组比较共有 Checkpoint 参数名称与形状。

历史 `experiment_*.txt` 没记录 reg、特征路径、Prompt 路径、部分优化器参数等；这些来自原训练脚本、代码默认值和现有日志/权重证据。旧实验没有当时的代码/输入 hash，因此不能追溯证明逐位复现或输入从未改变。新运行保存完整 argv、解释器、协议及代码/CSV/Prompt/splits 的 SHA256。保留特征路径，不对巨大特征重新提取或全量 hash。

Checkpoint schema 检查使用 `torch.load(..., weights_only=True, mmap=True)`，只查看名称/形状，避免读取整套冻结预训练张量。它不是历史权重的完整模型加载测试。独立复评工具使用 `load_state_dict(..., strict=True)`，不忽略缺失/多余权重。

划分检查覆盖 968 slides；每折 train/val/test 不重叠、无字面 case_id 重叠、两类均存在、五个 test fold 构成完整分区。沿用 Stage2 已列出的命名候选组，目前每折跨集合候选组数为 39/43/45/45/41；这些不是已确认患者身份。医院尚未确认编号语义，不能称为已证明 patient-level 独立。不得自行重划分。

倍率沿用旧目录名；Stage2 实测约 **2.5×/10×**，不重提取特征。已有五折曾用于研究开发，三组比较是开发后的 validation 证据，不是未接触测试集的独立泛化估计。

指定环境当前导入 SciPy 时会出现 NumPy 2.5.0 超出 SciPy 声明支持范围的警告。当前 import、指标测试、入口解析均可运行；没有安装或升级依赖。未进行正式训练或批量推理，不保证所有训练时数值路径已实测。

## 文件作用

- `activate_env.sh`：初始化 Conda、激活并验证 `vila_mil_overlay_rt`。
- `run_E0_remaining_folds.sh` / `run_E1_remaining_folds.sh`：只检查/补 Fold 2/3/4；验证完成的历史折会跳过；不重跑 Fold 0/1。
- `run_E2_fivefold.sh`：独立 Fold 0～4 Clean Dual。
- `run_experiment.py`：统一参数、前置检查、前台日志、互斥输出目录、失败返回非零。
- `common.py`：固定协议、来源白名单、输入/产物检查和追溯记录。
- `preflight.py`：只读审计，不创建模型、不推理。
- `aggregate_fivefold.py`：选中 epoch 的验证指标、五折均值/标准差/最佳最差折、三组配对差值、提升折数与最大退化。
- `evaluate_validation.py`：必要时手动严格复评原验证集，输出逐样本预测和 Macro-F1；不读取 Test loader。
- `test_matched_fivefold.py`：CPU 回归测试，包括 RNG 公平性、选中 epoch、概率列顺序、结果保护及失败路径。

## 手动运行命令

以下路径均为当前服务器绝对路径。Shell 包装器也会自行激活相同环境。缺少 `--execute` 时只检查和打印命令，绝不会启动训练。

```bash
cd /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main
source /opt/conda/etc/profile.d/conda.sh
conda activate vila_mil_overlay_rt
python -c 'import sys; print(sys.executable)'

# 可先运行只读核查
python analysis/stage3_model_design/05_finalization/01_matched_fivefold/preflight.py

# E0：当前三个折均完整，会报告 SKIP，无需重训
bash analysis/stage3_model_design/05_finalization/01_matched_fivefold/run_E0_remaining_folds.sh --execute

# E1：当前三个折均完整，会报告 SKIP，无需重训
bash analysis/stage3_model_design/05_finalization/01_matched_fivefold/run_E1_remaining_folds.sh --execute

# E2：实际需要手动执行的五折正式训练
bash analysis/stage3_model_design/05_finalization/01_matched_fivefold/run_E2_fivefold.sh --execute
```

所有任务前台顺序执行，无 nohup/tmux/后台启动。成功退出前会审计训练产物。非零退出或中断会保留日志与 lock，不自动续训。发现部分结果、配置不符、原有日志、重复来源或缺少完成标记即报错，不能简单跳过。运行前会先核查全部计划折，已知部分失败不会先启动其他折。若中断，不删除旧产物；人工确认原因后另建显式实验版本并更新来源清单，当前工具不猜测应该覆盖哪个版本。

新训练输出：

```text
01_matched_fivefold/runs/E2/fold{f}/
  run_manifest.json
  train.log
  completion.json                    # 仅训练和结果核查全部成功后生成
  stage351_matched_E2_fold{f}_s1/
    s_{f}_checkpoint.pt
    splits_{f}.csv
    experiment_stage351_matched_E2_fold{f}.txt
    epoch_details.csv
    fold_summary.csv
    summary_partial_{f}_{f}.csv
    result_partial_{f}_{f}.csv
    split_{f}_results.pkl             # validation-only 模式为空结果，不当作预测
    {f}/                             # TensorBoard
```

若未来某 E0/E1 待补折完全不存在，新结果放入对应 `runs/E0` / `runs/E1`，E1 另外保留 `routing_diagnostics.jsonl`。旧结果一律留在原目录；不会移动或复制 Checkpoint。

## 训练后汇总（通常不需要推理）

```bash
cd /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main
source /opt/conda/etc/profile.d/conda.sh
conda activate vila_mil_overlay_rt
python analysis/stage3_model_design/05_finalization/01_matched_fivefold/aggregate_fivefold.py --require-complete
```

每次生成新 `reports/YYYYMMDD_HHMMSS_microseconds/`，不覆盖历史报告。输出 `fivefold_metrics.csv`、`fivefold_statistics.csv`、`paired_differences.csv`、`paired_summary.csv`、`fivefold_summary.md`、`split_audit.csv` 和 `audit_manifest.json`。缺失/部分结果列明来源和问题；未完成五折不报告五折均值。`--require-complete` 在任何一组不完整时返回 2，同时留下缺失报告；普通模式允许 Pending，但遇到无效产物仍返回 2。

标签固定为 Adenocarcinoma=0、NonAdenocarcinoma=1；AUC 使用 `p_1`，预测取概率 argmax；Macro-F1 保持 sklearn `average='macro', zero_division=0` 的默认类别并集，要求验证集中实际含两类。不用 Test F1，不从 Test 选择 epoch。

仅当 `val_f1` 缺失或需要独立复核时，手动运行下面的批量验证。当前 E0/E1 的日志已足够，无须执行。输出目录必须新建且没有对应折结果；只评估部分折可用 `--arms E0 --folds 0`。

```bash
cd /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main
source /opt/conda/etc/profile.d/conda.sh
conda activate vila_mil_overlay_rt
python analysis/stage3_model_design/05_finalization/01_matched_fivefold/evaluate_validation.py \
  --arms E0 E1 E2 --folds 0 1 2 3 4 \
  --output-dir analysis/stage3_model_design/05_finalization/01_matched_fivefold/validation_replay_v1 \
  --execute
python analysis/stage3_model_design/05_finalization/01_matched_fivefold/aggregate_fivefold.py \
  --validation-dir analysis/stage3_model_design/05_finalization/01_matched_fivefold/validation_replay_v1 \
  --require-complete
```

复评保存每折 `validation_predictions.csv` 和 `validation_metrics.json`。汇总会重新核对验证样本/标签、概率、严格加载标记、选中 epoch、CSV/Prompt/split/epoch/prediction hash、Checkpoint 路径/大小/mtime，并从预测重算指标；已有指标若不一致则报错，不能用复评值掩盖来源错误。Checkpoint 文件本体不全量 hash，不能排除保留大小和时间戳的人为替换。

## 检查及 Stage3.5.2 入口条件

已执行指定环境的 Python 编译、Shell 语法、`main.py --help`、三组 dry-run、当前 15 个 arm/fold 状态汇总，以及 CPU 单元测试。未启动训练、真实模型批量加载或批量推理；RNG 测试对外部预训练加载使用 stub。

进入 Stage3.5.2 前应满足：15 个 arm/fold 的状态均 Complete；配置/共享 schema/划分一致；选中 epoch 与最终验证指标一致；三项验证指标与配对差值齐全；E1 五折 diagnostics 保留；患者身份和倍率限制写入研究记录。然后分析当前 Routing 的失效机制，不以验证均值更高作为进入下一阶段的硬性门槛，也不提前修改超参数或设计新模型。
