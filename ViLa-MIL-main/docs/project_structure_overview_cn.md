# ViLa-MIL 项目结构总览（中文）

本文档用于在新会话开始时，帮助大模型快速理解当前项目结构、两个版本的关系、你当前已经完成的训练资产，以及外部验证相关目录的位置。

## 1. 项目整体定位

当前仓库主目录为：

- `/private/ljh-data/shared/ViLMIL`

核心代码仓库位于：

- `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main`

这个项目已经不是单一的“论文原版 ViLa-MIL”，而是一个同时包含以下两条主线的工作仓库：

- 原版 `ViLa-MIL`
  - 使用原始 `CLIP RN50`
  - 主要模型文件：`models/model_ViLa_MIL.py`
  - 主要特征提取脚本：`feature_extraction/patch_extraction.py`
- `BiomedCLIP` 版 `ViLa-MIL`
  - 使用 `BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
  - 主要模型文件：`models/model_ViLa_MIL_BiomedCLIP.py`
  - 主要特征提取脚本：`feature_extraction/patch_extraction_biomedclip.py`

你当前实际在使用的主线，是 `BiomedCLIP` 版 `ViLa-MIL`。

## 2. 你当前工作的主任务

你已经完成了以下关键工作：

- 基于本地私有数据，完成了 `BiomedCLIP` 版 `ViLa-MIL` 的五折交叉训练
- 训练结果保存在：
  - `ViLa-MIL-main/trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`
- 该目录内包含：
  - `s_0_checkpoint.pt` 到 `s_4_checkpoint.pt`：5 个 fold 的模型权重
  - `splits_0.csv` 到 `splits_4.csv`：训练时使用的 5 折划分文件
  - `summary.csv`、`result.csv`、`fold_summary.csv`、`epoch_details.csv`：训练统计结果

你当前的后续任务，不是重新训练，而是：

- 使用这套本地私有数据训练得到的五折权重
- 去评估两个外部数据集：
  - `jiangxi_external`
  - `heyuan_external`

## 3. 仓库顶层结构说明

`ViLa-MIL-main` 下当前最重要的目录和文件如下：

- `main.py`
  - 训练入口
- `eval.py`
  - 评估入口
- `models/`
  - 核心模型实现
- `datasets/`
  - 数据集读取逻辑，负责按 `slide_id` 读取双尺度 `.h5` 特征
- `feature_extraction/`
  - patch 特征提取相关脚本
- `dataset_csv/`
  - 数据集的 CSV 定义文件
- `text_prompt/`
  - 双尺度文本提示词
- `trained_models/`
  - 训练产物
- `eval_results/`
  - 评估产物
- `docs/`
  - 文档说明
- `tools/`
  - 辅助分析、报告生成、检查脚本
- `create_patches_fp*.py`、`patch_generation_*`
  - patch 坐标生成和 patch 裁剪流程

## 4. 原版 ViLa-MIL 与 BiomedCLIP 版的关系

这两个版本共用同一套整体框架：

- 都是双尺度 MIL
- 都需要低倍和高倍两个尺度的 patch 特征
- 都使用文本 prompt 参与视觉-语言对齐
- 都通过 `main.py` 训练、`eval.py` 评估

主要区别在编码器和特征维度：

- 原版 `ViLa-MIL`
  - 图像编码器：`CLIP RN50`
  - 特征维度：`1024`
  - 模型文件：`models/model_ViLa_MIL.py`
- `BiomedCLIP` 版
  - 图像编码器：`BiomedCLIP`
  - 文本编码器：`PubMedBERT`
  - 特征维度：`512`
  - 模型文件：`models/model_ViLa_MIL_BiomedCLIP.py`

训练入口通过 `--model_type` 区分：

- `ViLa_MIL`
- `ViLa_MIL_BiomedCLIP`

你当前使用的是：

- `--model_type ViLa_MIL_BiomedCLIP`

## 5. 核心运行链路

当前项目的主流程可以理解为 4 段：

1. 原始 WSI 或已有 patch/坐标准备
2. 双尺度 patch 特征提取
3. 五折训练
4. 外部数据评估

对你现在最重要的是后 2 段，因为训练已经完成，外部评估也已有结果目录。

## 6. 训练入口与行为

训练入口文件：

- `ViLa-MIL-main/main.py`

它的职责：

- 解析训练参数
- 读取文本 prompt
- 构建 `Generic_MIL_Dataset`
- 根据 `splits_*.csv` 读取每个 fold 的训练/验证/测试集
- 调用 `utils/core_utils.py` 中的训练逻辑
- 输出每折权重与汇总结果

当前对你这个任务最关键的事实：

- `main.py` 支持 `model_type=ViLa_MIL_BiomedCLIP`
- `task_adenocarcinoma` 对应标签映射：
  - `Adenocarcinoma -> 0`
  - `NonAdenocarcinoma -> 1`
- 训练目录会命名成：
  - `results_dir/exp_code_s{seed}`

你当前训练好的目录就是：

- `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`

说明它对应：

- `exp_code = adenocarcinoma_biomedclip_dual_strict5`
- `seed = 1`

## 7. 评估入口与行为

评估入口文件：

- `ViLa-MIL-main/eval.py`

它的职责：

- 加载训练目录下的 `s_i_checkpoint.pt`
- 读取同目录下的 `splits_i.csv`
- 根据 `--csv_path` 指定要评估的数据集 CSV
- 对每个可用 fold 分别评估
- 输出每折结果、均值汇总、误判样本、耗时统计

当前 `eval.py` 的一个重要修改点是：

- 在 `task_adenocarcinoma` 下，它允许通过 `--csv_path` 切换评估数据集

这意味着：

- 训练仍然可以用内部私有数据
- 评估时可以直接切到 `all_data_jiangxi.csv` 或 `all_data_heyuan.csv`
- 同时继续复用训练时的五折权重和五折 split 文件

## 8. 数据读取契约

数据读取核心文件：

- `ViLa-MIL-main/datasets/dataset_generic.py`

其中 `Generic_MIL_Dataset.__getitem__()` 的关键逻辑是：

- 根据 CSV 中的 `slide_id`
- 去低倍特征目录读取：
  - `data_dir_s/<slide_id>.h5`
- 去高倍特征目录读取：
  - `data_dir_l/<slide_id>.h5`

每个 `.h5` 文件内部至少需要两个字段：

- `features`
- `coords`

也就是说，运行时真正依赖的是：

- CSV 中的 `slide_id`
- 双尺度特征目录里存在同名的 `.h5`

而不是直接在评估阶段再读原始 `.svs`。

## 9. CSV 格式

当前内部训练集和两个外部评估 CSV 的列结构一致，都是：

- `case_id`
- `slide_id`
- `label`

对应文件：

- `ViLa-MIL-main/dataset_csv/all_data.csv`
- `ViLa-MIL-main/dataset_csv/all_data_jiangxi.csv`
- `ViLa-MIL-main/dataset_csv/all_data_heyuan.csv`

因此，对新会话来说，判断一个新数据集能否直接接入评估，最重要就是检查：

- CSV 是否仍是这三列
- `slide_id` 是否能在双尺度特征目录中找到对应 `.h5`
- `label` 是否仍使用：
  - `Adenocarcinoma`
  - `NonAdenocarcinoma`

## 10. 文本提示词

当前腺癌任务使用的 prompt 文件是：

- `ViLa-MIL-main/text_prompt/adenocarcinoma_dual_scale_prompt.csv`

`main.py` 和 `eval.py` 都会读取这个文件，并拼接成：

- 前半段：低倍 prompt
- 后半段：高倍 prompt

模型内部会再把这两段拆回低倍/高倍文本特征，分别与两路图像特征做对齐。

## 11. BiomedCLIP 版的核心文件

如果新会话只想快速抓住 `BiomedCLIP` 主线，最优先看这些文件：

- `main.py`
  - 训练入口
- `eval.py`
  - 评估入口
- `models/model_ViLa_MIL_BiomedCLIP.py`
  - BiomedCLIP 版模型主体
- `feature_extraction/patch_extraction_biomedclip.py`
  - BiomedCLIP 特征提取入口
- `feature_extraction/patch_extraction_utils_biomedclip.py`
  - BiomedCLIP 特征提取细节
- `datasets/dataset_generic.py`
  - `.h5` 特征读取逻辑

## 12. 当前训练产物目录

你当前最重要的训练目录是：

- `ViLa-MIL-main/trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`

这个目录可以视为“当前主模型资产目录”。

里面最关键的文件分为三类：

- 模型权重
  - `s_0_checkpoint.pt`
  - `s_1_checkpoint.pt`
  - `s_2_checkpoint.pt`
  - `s_3_checkpoint.pt`
  - `s_4_checkpoint.pt`
- 折划分文件
  - `splits_0.csv`
  - `splits_1.csv`
  - `splits_2.csv`
  - `splits_3.csv`
  - `splits_4.csv`
- 训练统计
  - `summary.csv`
  - `result.csv`
  - `fold_summary.csv`
  - `epoch_details.csv`

对外部评估来说，最关键的是前两类。

## 13. 外部数据集位置

当前外部数据集位于：

- 江西外部集：
  - `ViLa-MIL-main/data/jiangxi`
- 河源外部集：
  - `ViLa-MIL-main/data/heyuan`

你已经说明“不需要查看具体数据内容，只需要知道格式即可”。从目录结构上看，这两个外部集都已经具备可评估所需的双尺度特征目录。

### 13.1 江西外部集

目录下可见的重要结构包括：

- `wsi/`
- `patches_coords_2048/`
- `patches_coords_20x/`
- `patches_5x/`
- `patches_20x/`
- `features_biomedclip_5x/`
- `features_biomedclip_20x/`

这说明江西外部集不仅有原始 WSI，也已经走完了：

- 坐标生成
- patch 裁剪
- BiomedCLIP 双尺度特征提取

### 13.2 河源外部集

目录下可见的重要结构包括：

- `wsi/`
- `patches_coords_5x/`
- `patches_coords_20x/`
- `patches_5x/`
- `patches_20x/`
- `features_biomedclip_5x/`
- `features_biomedclip_20x/`
- `reseg/`

这同样说明河源外部集已经有可直接用于评估的双尺度 `.h5` 特征。

## 14. 外部评估对应 CSV

两个外部集分别对应：

- `ViLa-MIL-main/dataset_csv/all_data_jiangxi.csv`
- `ViLa-MIL-main/dataset_csv/all_data_heyuan.csv`

评估时的逻辑是：

- `--csv_path` 指向这两个 CSV 之一
- `--data_root_dir` 指向对应外部数据集根目录
- `--data_folder_s` 和 `--data_folder_l` 指向对应的双尺度特征目录
- `--models_exp_code` 继续使用训练目录 `adenocarcinoma_biomedclip_dual_strict5_s1`

## 15. 当前外部评估结果目录

你已经有现成的外部评估结果目录：

- 河源结果：
  - `ViLa-MIL-main/eval_results/heyuan/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_heyuan_external`
- 江西结果：
  - `ViLa-MIL-main/eval_results/jiangxi/EVAL_jiangxi_external_biomedclip_2048_512_to256`

从结果目录结构看，典型输出包括：

- `fold_0.csv` 到 `fold_4.csv`
  - 每折逐样本预测结果
- `fold_0_misclassified.csv` 到 `fold_4_misclassified.csv`
  - 每折误判样本
- `summary.csv`
  - 各折指标汇总
- `result.csv`
  - 平均值与方差/标准差
- `timing_details.csv`
  - 耗时统计

江西目录额外还有：

- `ensemble_5fold_predictions.csv`
- `ensemble_5fold_report.csv`
- `result_calibrated.csv`
- `summary_calibrated.csv`

说明江西外部评估还做过五折集成或校准后的扩展分析。

## 16. 项目里的历史/补充文档

当前仓库里已经有一些与 BiomedCLIP 相关的说明文档，可作为补充阅读：

- `BIOMEDCLIP_INTEGRATION_GUIDE.md`
  - BiomedCLIP 集成说明
- `docs/biomedclip_vs_clip.md`
  - 原版 CLIP 与 BiomedCLIP 的流程差异
- `docs/dual_scale_prompt_usage.md`
  - 双尺度 prompt 的使用方式
- `docs/vilamil_biomedclip_timeline_cn.md`
  - BiomedCLIP 版流程时序图中文解读
- `SESSION_DIALOGUE_SUMMARY.md`
  - 某次会话中关于江西外部验证的压缩总结

这些文档能帮助补充细节，但如果只是为了快速进入任务，本文件应作为第一入口。

## 17. 新会话建议的最短阅读顺序

如果以后开启新会话，希望大模型最快进入状态，建议按下面顺序阅读：

1. 本文档：
   - `docs/project_structure_overview_cn.md`
2. 如果本次工作与外部验证直接相关：
   - `eval.py`
   - `datasets/dataset_generic.py`
3. 如果本次工作与 BiomedCLIP 模型修改相关：
   - `models/model_ViLa_MIL_BiomedCLIP.py`
4. 如果本次工作与特征或数据预处理相关：
   - `feature_extraction/patch_extraction_biomedclip.py`
   - `feature_extraction/patch_extraction_utils_biomedclip.py`
5. 如果需要对比原版：
   - `models/model_ViLa_MIL.py`
   - `docs/biomedclip_vs_clip.md`

## 18. 对后续会话最关键的结论

可以把当前项目简单记成下面这句话：

- 这是一个同时包含原版 `ViLa-MIL` 和 `BiomedCLIP` 版 `ViLa-MIL` 的工作仓库，但当前实际主线是 `BiomedCLIP` 版；本地私有数据五折训练已完成，主模型目录是 `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`；当前重点是拿这套五折权重去评估外部数据集 `jiangxi_external` 和 `heyuan_external`，它们分别使用 `all_data_jiangxi.csv`、`all_data_heyuan.csv` 和各自根目录下已准备好的 `features_biomedclip_5x`、`features_biomedclip_20x` 双尺度特征。

## 19. 当前会话确认过的关键路径

- 仓库根目录：
  - `/private/ljh-data/shared/ViLMIL`
- 核心代码目录：
  - `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main`
- 当前主训练目录：
  - `ViLa-MIL-main/trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`
- 外部 CSV：
  - `ViLa-MIL-main/dataset_csv/all_data_jiangxi.csv`
  - `ViLa-MIL-main/dataset_csv/all_data_heyuan.csv`
- 外部数据根目录：
  - `ViLa-MIL-main/data/jiangxi`
  - `ViLa-MIL-main/data/heyuan`
- 当前外部评估结果：
  - `ViLa-MIL-main/eval_results/jiangxi/EVAL_jiangxi_external_biomedclip_2048_512_to256`
  - `ViLa-MIL-main/eval_results/heyuan/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_heyuan_external`

## 20. 后续维护建议

如果后面继续增加新的外部队列或新的训练版本，建议继续沿用当前约定：

- 训练目录统一放在 `trained_models/`
- 外部评估目录按数据源分子目录：
  - `eval_results/heyuan/`
  - `eval_results/jiangxi/`
- 每新增一个重要实验，就在 `docs/` 下补一份简短中文说明

这样以后无论是新会话还是换模型，都能快速定位当前主线。
