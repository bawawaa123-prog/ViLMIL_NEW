# Stage 3.5.2 — Low-to-High Routing Failure Mechanism Diagnosis

实现已准备；正式五折推理、空间扰动与统计图均由研究者手动执行。默认dry-run；没有执行Git操作、训练、真实Validation批量推理或数据重生成。核心模型和历史工具文件没有修改。

完整核查、分组/范数/随机种子定义和判定边界见 [protocol.md](protocol.md)。Stage351最新报告 `../01_matched_fivefold/reports/20261009_203945_934130/` 是基线，不是Stage352机制结果。

## 文件作用

| 文件 | 作用 |
|---|---|
| bridge.py | 复用Stage351来源/预检、Stage335 E1严格加载；为E0/E2关闭Router构造；固定种子/阈值 |
| telemetry.py | 只读Hook：Gate饱和、空间组、patch/WSI残差统计；默认推理不变 |
| run_diagnostics.py | 按Fold同一E1权重多模式推理，E0正常对照，可选E2；写日志、完成标记与manifest |
| run_stage352.sh | 激活指定Conda并设置工作目录 |
| preflight.py | 只读源/配置检查；检查中断结果；可写明确Pending的准备报告 |
| analysis_utils.py | 复用Stage336配对指标，增加正确性变化、margin、类别与排序统计 |
| aggregate_diagnostics.py | 核验完整job，输出统计/五折报告；不读取Test或启动推理 |
| plot_diagnostics.py | 统计图与固定有界坐标散点，不读取WSI或原始病理图像 |
| tests/test_diagnostics.py、tests/test_pipeline.py | CPU合成数据、Hook一致性、多父节点/无效行、配对标签、完整性/覆盖保护，以及合成CSV到统计图全流程 |

当前缺失：全部Stage352正式干预结果，以及据此回答Q1–Q6的机制证据。默认35个折遍历（E0五次+E1三十次），约6776次Slide前向；耗时取决于每张Slide的patch数和GPU，不预估未经测量的具体分钟数。

## 准备检查与正式执行

每段命令均在当前服务器执行，所有Python都使用`vila_mil_overlay_rt`。不需要安装、升级或创建环境。

```bash
cd /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main
source /opt/conda/etc/profile.d/conda.sh
conda activate vila_mil_overlay_rt
python -c 'import sys; print(sys.executable)'
python analysis/stage3_model_design/05_finalization/02_routing_diagnostics/preflight.py

# 默认dry-run：显示计划，不构造真实模型，不创建结果目录
bash analysis/stage3_model_design/05_finalization/02_routing_diagnostics/run_stage352.sh

# 正式五折：目录必须不存在；三固定Shuffle种子全部保留
bash analysis/stage3_model_design/05_finalization/02_routing_diagnostics/run_stage352.sh \
  --folds 0 1 2 3 4 \
  --modes normal context_shuffle context_mean residual_off \
  --shuffle-seeds 352 353 354 \
  --output-dir analysis/stage3_model_design/05_finalization/02_routing_diagnostics/runs/stage352_full_v1 \
  --execute
```

如需事先运行预定Fold3/4子集，使用独立目录，不能用它替代完整五折结果：

```bash
bash analysis/stage3_model_design/05_finalization/02_routing_diagnostics/run_stage352.sh \
  --folds 3 4 --output-dir analysis/stage3_model_design/05_finalization/02_routing_diagnostics/runs/stage352_folds34_v1 \
  --execute
```

可加`--include-e2`加入五折Clean Dual正常推理，仍以E0/E1为主；不是必需任务。可用`--modes normal residual_off`做显式子集；Normal强制存在，缺少Shuffle/Mean不能回答完整空间依赖问题。不同子集/版本不要混拼来源。

## 结果和安全检查

```text
runs/stage352_full_v1/
  experiment_manifest.json            # 冻结计划、来源、阈值、seed、输入/代码hash
  run.log
  completion.json                    # 所有计划完成后才生成
  fold{f}/
    parameter_differences.csv         # E0/E1共享训练参数距离，非因果证明
    E0/normal_seed0/
      predictions.csv
      completion.json
    E1/{mode}_seed{seed}/
      predictions.csv
      gate_statistics.csv
      residual_statistics.csv
      histograms.csv                 # 全部合法mapped patch计数
      completion.json                # 当前job完整CSV的hash
      patch_samples.csv              # 仅Normal，有界标量/坐标
      region_samples.csv             # 仅Normal，有界Low区域标量/坐标；无合法组时缺省
      low_context_statistics.csv     # 仅Normal，复用Stage337
```

仅保存Slide ID用于对齐，不导出case_id、患者资料、WSI、patch图像或高维特征矩阵。根/job完成标记与独立目录阻止覆盖。失败保留`.running.lock`和部分文件，不能再次在原目录执行。

检查中断或完成状态：

```bash
python analysis/stage3_model_design/05_finalization/02_routing_diagnostics/preflight.py \
  --run-dir analysis/stage3_model_design/05_finalization/02_routing_diagnostics/runs/stage352_full_v1
```

Partial/Invalid/Pending计划返回非零，保留原始结果；本工具不自动恢复、补写完成标记、覆盖或猜测续跑。要继续时先人工排查，再用新输出版本运行。已有完整job可由汇总显式局部描述，但未完成job不会被采用。

## 生成汇总

```bash
cd /private/ljh-data/shared/ViLMIL_NEW/ViLa-MIL-main
source /opt/conda/etc/profile.d/conda.sh
conda activate vila_mil_overlay_rt
python analysis/stage3_model_design/05_finalization/02_routing_diagnostics/aggregate_diagnostics.py \
  --run-dir analysis/stage3_model_design/05_finalization/02_routing_diagnostics/runs/stage352_full_v1 \
  --output-dir analysis/stage3_model_design/05_finalization/02_routing_diagnostics/reports/stage352_full_v1 \
  --require-complete --execute
```

输出目录必须不存在。去掉`--execute`仅核验。汇总保存：

- `routing_intervention_metrics.csv`：每Fold/模式/seed的AUC、ACC、Macro-F1及相对Normal变化、改变Slide数、概率和margin变化、纠错/损害数。
- `routing_gate_statistics.csv`、`routing_residual_statistics.csv`：逐Slide统计，带来源/模式/seed。
- `prediction_changes.csv`：逐Slide干预及B0/B1/B2对照；E0/E1对照行包含E1 Normal Gate/Residual统计。
- `fold_comparison.csv`：独立E0 vs正常E1、同权重E1 vs关闭Residual、E0 vs关闭Residual。
- `correctness_group_statistics.csv`：四类正确性变化、数量/比例、类别/边界分层和统计均值。
- `ranking_changes.csv`、`class_metrics.csv`：Fold4及其他折的排序/硬分类区别；AUC正类为1 NonAdenocarcinoma。
- `parameter_differences.csv`、`low_context_statistics.csv`：共适应/上下文线索。
- `fivefold_intervention_summary.csv`：按固定seed分别汇总五折，ddof=0；缺折不填五折均值。
- `experiment_status.csv`、`experiment_manifest.json`、`stage352_summary.md`：状态、来源和证据表/解释边界。

Normal复评必须对齐Stage351选中Checkpoint的三项Validation指标，否则当前job报错。共享配置/schema、文件hash、逐Slide标签及预测概率也会核对。大Checkpoint仅使用路径/大小/mtime身份校验，未全量hash；保留这一追溯限制。

## 生成统计图

```bash
python analysis/stage3_model_design/05_finalization/02_routing_diagnostics/plot_diagnostics.py \
  --run-dir analysis/stage3_model_design/05_finalization/02_routing_diagnostics/runs/stage352_full_v1 \
  --report-dir analysis/stage3_model_design/05_finalization/02_routing_diagnostics/reports/stage352_full_v1 \
  --output-dir analysis/stage3_model_design/05_finalization/02_routing_diagnostics/reports/stage352_figures_v1 \
  --execute
```

生成Gate/Residual比例直方图、各Fold Gate分布、干预性能差值图和每折预定的首个Slide有界空间散点。散点采用原存储坐标，不是WSI热图或真实病灶标注；Gate不能解释为病灶概率。图表数量有界，不会批量读WSI。去掉`--execute`只检查。

## 测试与已知限制

```bash
python -m compileall -q analysis/stage3_model_design/05_finalization/02_routing_diagnostics
bash -n analysis/stage3_model_design/05_finalization/02_routing_diagnostics/run_stage352.sh
python -m unittest discover -s analysis/stage3_model_design/05_finalization/02_routing_diagnostics/tests -p 'test_*.py' -v
```

轻量测试用CPU合成数据；真实生产forward/Router/Low Context路径配合替换的外部文本/预训练加载测试，未下载模型、未做真实Slide批量推理。指定环境仍有已知NumPy/SciPy版本范围警告，不擅自改依赖。

跨集合潜在患者身份重叠、实际倍率与目录名不同、验证折已用于开发、历史hash不足、单seed/交叉验证非独立等限制延续。以同Checkpoint干预的直接证据优先，独立训练参数/病例统计只能作为相关线索，不能自动归因为全局Gate、错误Low信息或唯一共适应原因。
