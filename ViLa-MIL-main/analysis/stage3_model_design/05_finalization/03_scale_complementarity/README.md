# Stage 3.5.3 — Independent Low/High Complementarity

**代码与轻量检查已准备，正式Low五折训练和Validation推理Pending，由研究者手动执行。** 没有运行Git、正式训练或真实WSI批量推理；现有核心模型、权重、Prompt、数据、split和历史CSV均未修改。

Stage352正式解释已补齐：`../02_routing_diagnostics/stage352_final_interpretation.md`。详细来源审计见 `reports/preparation_20261013/source_audit.json`；研究定义和局限见 `protocol.md`。

## 已核查的模型

| 模型 | 存在的Fold | 本阶段可用性 |
|---|---|---|
| Stage351 Clean High E0 | 0–4 | 全部复用；High预测直接沿用核验通过的Stage352 E0 |
| Stage342 Independent Low | 0、1 | Router/Stabilization配置及schema与Clean协议不同，历史保留，不纳入正式配对 |
| Stage2 Low | 0–4 | 权重实际存在；追溯记录不足，不把历史Test结果或未核定协议直接混入 |
| Stage341 Dual-trained | 0、1 | 非独立Low，不能替代 |

因此本阶段匹配Low可用Fold为**暂无**；准备训练Fold **0、1、2、3、4**，High无需重训。旧Test评估本身不等于Low权重无效；详细拒绝依据和追溯边界已记录。

## 新增文件

| 文件 | 功能 |
|---|---|
| stage353_common.py | 复用Stage351/352来源审计、命令、指标/配对；新Low协议及hash核验 |
| preflight.py | 只读核查；可生成明确Pending的准备报告 |
| train_low.py / run_low_only_training.sh | E0协议下独立Low训练，默认dry-run；防覆盖/部分结果自动续训 |
| run_independent_inference.py | 只推理独立Low，复制已核验High；默认dry-run，逐折/root完成hash |
| analyze_complementarity.py | G1–G4、Oracle硬分类上限、固定Margin分层和Routing关联；纯函数，不执行旧融合搜索 |
| aggregate_stage353.py | 来源完整性→性能/互补/病例/关联/五折统计→匿名报告与复核清单 |
| prepare_case_visualizations.py | 本地资源清单/既有Routing散点；可显式运行已选病例Attention、Top Patch坐标/拼图和Mapping对照 |
| finalize_stage352.py | 只读旧表生成正式Stage352解释；已经执行，无需再次运行 |
| tests/test_stage353.py | CPU合成预测、来源、分组、隐私、完整性、正常forward Hook测试 |
| protocol.md / README.md | 冻结协议、完整命令、预期产物与验收 |

## 运行顺序

以下命令均从当前服务器仓库执行。先初始化指定环境：

```bash
cd /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main
source /opt/conda/etc/profile.d/conda.sh
conda activate vila_mil_overlay_rt
python -c 'import sys; print(sys.executable)'
```

**1. 只读检查与训练dry-run。** 不构造真实模型、不创建正式训练结果：

```bash
python analysis/stage3_model_design/05_finalization/03_scale_complementarity/preflight.py
bash analysis/stage3_model_design/05_finalization/03_scale_complementarity/run_low_only_training.sh \
  --folds 0 1 2 3 4 \
  --output-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/runs/low_matched_v1
```

**2. 正式匹配Low五折训练（耗时任务，仅研究者执行）。** 沿用E0训练协议，无Router/Global/Warm Start：

```bash
bash analysis/stage3_model_design/05_finalization/03_scale_complementarity/run_low_only_training.sh \
  --folds 0 1 2 3 4 \
  --output-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/runs/low_matched_v1 \
  --execute
```

逐Fold输出：`runs/low_matched_v1/fold{f}/stage353_low_fold{f}_s1/s_{f}_checkpoint.pt`、epoch_details/fold_summary/splits，以及fold父目录的train.log、run_manifest.json、completion.json。可以指定Fold子集，但五折验收不能缺折。已核验完成Fold会跳过；部分/Invalid来源报错、不续训不覆盖。需要新版本时显式改所有后续 `--low-run-dir`。

**3. 训练完成后检查，再运行Validation推理（耗时任务，仅研究者执行）。** 完整来源检查不读取Test预测：

```bash
python analysis/stage3_model_design/05_finalization/03_scale_complementarity/preflight.py \
  --low-run-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/runs/low_matched_v1 \
  --require-complete

python analysis/stage3_model_design/05_finalization/03_scale_complementarity/run_independent_inference.py \
  --folds 0 1 2 3 4 \
  --low-run-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/runs/low_matched_v1 \
  --stage352-run-dir analysis/stage3_model_design/05_finalization/02_routing_diagnostics/runs/stage352_full_v1 \
  --stage352-report-dir analysis/stage3_model_design/05_finalization/02_routing_diagnostics/reports/20261010_130747_580789 \
  --output-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/runs/independent_val_v1 \
  --execute
```

只做Low的五折Validation前向，High原CSV按字节复制。预计968次Low Slide前向，时长由Patch数/硬件决定，不捏造分钟数。输出每Fold的high_predictions.csv、low_predictions.csv、completion.json，根manifest/completion/run.log。若Low尚Pending，加`--execute`会在启动前报错；不加可显示计划。

**4. 完整汇总。** 仅读已有CSV，默认dry-run；输出必须是新目录：

```bash
python analysis/stage3_model_design/05_finalization/03_scale_complementarity/aggregate_stage353.py \
  --run-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/runs/independent_val_v1 \
  --output-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/reports/stage353_full_v1 \
  --require-complete \
  --execute
```

预期输出：

```text
reports/stage353_full_v1/
  independent_scale_metrics.csv              # 每Fold H/L性能、类别召回与混淆矩阵
  paired_scale_differences.csv               # L-H指标差
  fivefold_scale_statistics.csv              # 完整五折才有mean/std(ddof=0)
  complementarity_summary.csv                # G1–G4、G3机会、G2风险、Oracle上限
  fivefold_complementarity_statistics.csv
  case_level_complementarity.csv             # 匿名token、概率、margin、分组
  case_group_statistics.csv
  conditional_complementarity.csv            # 固定High margin/类别描述，非选择器
  routing_complementarity_association.csv     # 病例与E1/空间统计关联
  routing_group_statistics.csv
  case_review_selection.csv                  # 所有G3和固定G2对照
  pathology_review_template.csv              # 空白专家复核模板；初始Pending
  experiment_status.csv
  experiment_manifest.json
  stage353_summary.md
  private/                                   # 仅本地，不公开
    slide_id_mapping.csv
    id_salt.bin
```

不用Test、不搜索阈值、不训练融合选择器。Oracle只表示已知标签的硬分类纠错上限。全部正式推理/报告保留输入与来源追溯；改变文件、缺失ID、标签/概率/来源不一致则拒绝。

**5. 病例复核图。** 推荐先生成既有Routing散点和本地资源清单（不重新推理）：

```bash
python analysis/stage3_model_design/05_finalization/03_scale_complementarity/prepare_case_visualizations.py \
  --report-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/reports/stage353_full_v1 \
  --output-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/reports/stage353_case_review_v1 \
  --folds 0 1 2 3 4 \
  --execute
```

如需匹配独立H/L的新注意力与Top Patch拼图，由研究者在**另一新目录**运行下列命令（已选病例推理，可能耗时）：

```bash
python analysis/stage3_model_design/05_finalization/03_scale_complementarity/prepare_case_visualizations.py \
  --report-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/reports/stage353_full_v1 \
  --output-dir analysis/stage3_model_design/05_finalization/03_scale_complementarity/reports/stage353_case_attention_v1 \
  --folds 0 1 2 3 4 \
  --top-k 10 \
  --extract-attention \
  --execute
```

所有G3与字典序类/Fold匹配G2对照；正常forward Hook，概率核对；Top坐标、最多1024点attention散点、既有Patch PNG拼图、High→Low多parent对照。缺Patch图片标Pending，不重新提取特征或批量读WSI。目录含本地受控资料与真实病理图像，不能直接公开。Attention、Gate不是病灶真值，病理形态结论需专家复核；没有复核时保留Pending。

若Patch PNG的ICC配置超过Pillow元数据解压限额，拼图读取会仅在内存副本中移除iCCP元数据后重试；原PNG、像素数据与全局Pillow限制不修改，逐Patch的`icc_profile_omitted`记录在`private/top_patch_resources.csv`。这里沿用原RGB转换，不执行ICC色彩校正。其他图片解码错误仍报错。失败目录保持原样，再次运行须指定新输出目录；不必重训或重新汇总。

## 轻量测试

```bash
/opt/conda/envs/vila_mil_overlay_rt/bin/python -m compileall -q \
  analysis/stage3_model_design/05_finalization/03_scale_complementarity
bash -n analysis/stage3_model_design/05_finalization/03_scale_complementarity/run_low_only_training.sh
/opt/conda/envs/vila_mil_overlay_rt/bin/python -m unittest discover \
  -s analysis/stage3_model_design/05_finalization/03_scale_complementarity/tests \
  -p 'test_*.py' -v
```

测试只用CPU合成数据，外部预训练加载被stub；没有下载依赖或真实Slide前向。现有环境的NumPy/SciPy支持范围警告保持原样，不改依赖。

## 当前状态与验收

Stage352解释Complete；High来源/预测Complete；工具准备Complete。Low新五折训练、Low预测、真实G1–G4/Oracle/Routing关联均**Pending**，专家形态复核**Pending**。不能把准备报告写成Stage353五折完成。

计算验收要求：Low0–4全部完成且协议/来源/复评通过；H/L严格配对；全部统计和完整五折mean/std齐全；`--require-complete`通过。研究验收要求：从真实结果回答Q1–Q6，分析G3稳定条件与G2风险，区分Routing关联和因果；缺专家证据明确说明。进入Stage3.6前先确认什么条件下Low有稳定优势，再提出待独立验证的创新假说。
