# 广东医外部队列外部验证全流程（参照河源流程改写）

本文档基于当前仓库中的真实脚本状态，为广东医外部队列整理一套可直接执行的完整外部验证流程。

目标是：

1. 使用 `dataset_csv/all_data_guangdongyi.csv` 作为广东医外部队列 CSV
2. 使用 `data/gdmuah/wsi` 作为广东医 SVS 根目录
3. 按河源外部验证的双尺度流程重新生成组织分割、patch 坐标、patch PNG 和 BiomedCLIP 特征
4. 复用现有五折权重 `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1` 对广东医队列做外部评估
5. 生成外部验证报告，并为后续热图 / top patch 分析保留标准输出目录

这份手册尽量与 [heyuan_rerun_with_new_preset_cn.md](./heyuan_rerun_with_new_preset_cn.md) 的风格和执行顺序保持一致，但结合广东医当前目录结构做了针对性调整。

---

## 1. 当前已确认的广东医输入信息

截至 2026-06-09，本仓库内已确认：

- 广东医 CSV：
  - `dataset_csv/all_data_guangdongyi.csv`
- 广东医 WSI 根目录：
  - `data/gdmuah/wsi`
- WSI 格式：
  - `.svs`
- 文本提示：
  - `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- 五折权重：
  - `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`

额外核对结果：

- `all_data_guangdongyi.csv` 当前共有 `335` 条记录
- `data/gdmuah/wsi` 当前共有 `343` 个 `.svs` 文件

这说明正式跑流程前，建议先做一次“CSV 与 WSI 对齐检查”，确认：

1. CSV 中的 `slide_id` 是否都能在 `wsi/` 下找到对应 `.svs`
2. `wsi/` 中是否存在不参与本次外部验证的额外文件

---

## 2. 这次流程的总体策略

沿用河源外部验证的双尺度定义：

- `5x` 支路：
  - `patch_level=2`
  - 直接切 `256x256`
- `20x` 支路：
  - `patch_level=1`
  - 直接切 `256x256`

也就是说，这里仍然采用当前仓库真实脚本行为：

- 坐标在指定 level 上生成
- patch 在指定 level 上直接裁出 `256x256`
- 不再假设“先读更大区域再缩放到 256”

这与河源新版流程保持一致。

关于"严格五折交叉验证划分"，这里需要特别说明：

- 这份文档当前走的是"外部验证"流程，而不是"在广东医队列上重新训练 / 重新做内部交叉验证"
- 因此，**这里故意没有加入对 `dataset_csv/all_data_guangdongyi.csv` 再做一次严格五折划分的步骤**
- 原因是外部验证阶段会直接复用既有五折模型权重 `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`
- 在评估时使用 `eval.py --split all`，让广东医外部队列的全部样本都参与每个 fold 的推理
- 也就是说：**当前外部验证流程不需要重新对广东医 CSV 生成 `splits_0.csv ... splits_4.csv`**

只有在下面两种场景中，才需要额外加入"严格五折划分"步骤：

1. 你准备在广东医队列上重新训练模型
2. 你准备在广东医队列内部做性能估计，而不是把它纯粹当作独立外部测试集

后文会补充一个"可选步骤"，专门用于这两类情况。

---

## 3. 本次建议使用的核心目录

仓库根目录：

- `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main`

本次关键输入：

- 广东医 WSI 根目录：
  - `data/gdmuah/wsi`
- 广东医 CSV：
  - `dataset_csv/all_data_guangdongyi.csv`
- 预设：
  - `presets/gdmuah.csv`
- 文本提示：
  - `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- 五折权重：
  - `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`

本次建议新输出目录：

- `data/gdmuah/patches_coords_5x`
- `data/gdmuah/patches_coords_20x`
- `data/gdmuah/patches_5x`
- `data/gdmuah/patches_20x`
- `data/gdmuah/features_biomedclip_5x`
- `data/gdmuah/features_biomedclip_20x`
- `eval_results/gdmuah`

说明：

- 我这里不额外套 `external_eval/` 子目录，而是直接与 `data/heyuan` 的现有组织方式保持一致。
- 如果你后面还要做不同 preset 的对照实验，可以在 `data/gdmuah_rerun_v2/`，或在 `eval_results/gdmuah/` 下按实验名继续扩展。

---

## 4. 运行前约定

以下命令默认在仓库根目录执行：

```bash
cd /private/ljh-data/shared/ViLMIL/ViLa-MIL-main
```

---

## 5. Step 0：创建输出目录

```bash
mkdir -p data/gdmuah/patches_coords_5x
mkdir -p data/gdmuah/patches_coords_20x
mkdir -p data/gdmuah/patches_5x
mkdir -p data/gdmuah/patches_20x
mkdir -p data/gdmuah/features_biomedclip_5x
mkdir -p data/gdmuah/features_biomedclip_20x
mkdir -p eval_results/gdmuah
```

---

## 6. Step 0.5：先做 CSV 与 WSI 的一致性检查

建议先跑这一轮预检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_guangdongyi.csv \
  --dirs_root data/gdmuah/wsi \
  --output_path eval_results/gdmuah/gdmuah_csv_vs_wsi.csv
```

这一步会帮你确认：

- `slide_id` 是否都能在 `wsi/` 下找到对应文件名
- `wsi/` 中是否有额外的 `.svs` 不在 CSV 中

如果这里出现：

- `missing_in_dirs > 0`

那说明有 CSV 样本在 WSI 目录中找不到原图，后续坐标生成时会直接缺失。

如果这里出现：

- `extra_in_dirs > 0`

通常不是大问题，只说明目录里有额外文件没参与本次评估。

---

## 7. Step 1：使用广东医专属 `gdmuah.csv` preset

当前仓库已经新增：

- `presets/gdmuah.csv`

这份 preset 是基于广东医代表性 `svs` 样本做的第一版定制参数，不再直接复用 `heyuan.csv`。

当前参数为：

- `seg_level=-1`
- `mthresh=7`
- `close=5`
- `use_otsu=TRUE`
- `a_t=12`
- `a_h=3`
- `max_n_holes=10`

它的定位是：

1. 保留 `heyuan.csv` 的 Otsu 分割主策略
2. 比 `heyuan.csv` 稍微增强连通性，减少广东医部分样本上的组织断裂
3. 又不像 `jiangxi.csv` 那样对小碎片过于宽松

因此，广东医第一轮外部验证建议直接使用这份 `gdmuah.csv`。

---

## 8. Step 2：重新生成 5x 坐标和分割结果

这里继续使用 `create_patches_fp_heyuan.py`。

虽然脚本名里带 `heyuan`，但它本质上已经是一个可复用的外部队列坐标生成入口，只要你通过参数改掉 `--source`、`--csv` 和 `--save-dir` 就可以复用到广东医。

```bash
python create_patches_fp_heyuan.py \
  --source data/gdmuah/wsi \
  --csv dataset_csv/all_data_guangdongyi.csv \
  --save-dir data/gdmuah/patches_coords_5x \
  --preset gdmuah.csv \
  --patch-size 256 \
  --step-size 256 \
  --patch-level 2 \
  --seg \
  --patch \
  --stitch
```

输出重点：

- `data/gdmuah/patches_coords_5x/patches_256/*.h5`
- `data/gdmuah/patches_coords_5x/masks/*.jpg`
- `data/gdmuah/patches_coords_5x/graph_256/*.jpg`

跑完后建议先抽查：

- `masks/*.jpg`
- `graph_256/*.jpg`

重点确认：

1. 组织区域是否被正确圈出
2. 背景是否被大面积误保留
3. patch 分布是否与组织区域匹配

---

## 9. Step 3：重新生成 20x 坐标和分割结果

```bash
python create_patches_fp_heyuan.py \
  --source data/gdmuah/wsi \
  --csv dataset_csv/all_data_guangdongyi.csv \
  --save-dir data/gdmuah/patches_coords_20x \
  --preset gdmuah.csv \
  --patch-size 256 \
  --step-size 256 \
  --patch-level 1 \
  --seg \
  --patch \
  --stitch
```

输出重点：

- `data/gdmuah/patches_coords_20x/patches_256/*.h5`
- `data/gdmuah/patches_coords_20x/masks/*.jpg`
- `data/gdmuah/patches_coords_20x/graph_256/*.jpg`

说明：

- 这里的 `patch-level=1` 是高倍支路定义
- 当前脚本行为仍然是“在 level 1 上直接生成并后续直接切 `256x256` 坐标”

---

## 10. Step 4：检查 5x / 20x 坐标生成是否完整

建议分别检查 5x 和 20x 的 `h5` 是否覆盖全部 CSV 样本。

5x 检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_guangdongyi.csv \
  --dirs_root data/gdmuah/patches_coords_5x/patches_256 \
  --output_path eval_results/gdmuah/gdmuah_csv_vs_coords5x.csv
```

20x 检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_guangdongyi.csv \
  --dirs_root data/gdmuah/patches_coords_20x/patches_256 \
  --output_path eval_results/gdmuah/gdmuah_csv_vs_coords20x.csv
```

如果这里出现缺失样本，优先回头检查：

1. 原始 `wsi/` 是否缺图
2. 某些 SVS 是否无法被 openslide 正常读取
3. 该 slide 的组织分割是否失败

---

## 11. Step 5：用 `patch_generation_5x.py` 裁剪 5x PNG

```bash
python patch_generation_5x.py \
  --source data/gdmuah/wsi \
  --csv dataset_csv/all_data_guangdongyi.csv \
  --coords-root data/gdmuah/patches_coords_5x \
  --patch-size 256 \
  --patch-level 2 \
  --output-root data/gdmuah/patches_5x \
  --workers 8 \
  --skip-existing
```

这一步的真实行为是：

- 从 `patches_coords_5x/patches_256/*.h5` 读坐标
- 在 `level=2` 上直接切 `256x256`
- 不做 resize

输出通常为：

- `data/gdmuah/patches_5x/<slide_id>/*.png`

---

## 12. Step 6：用 `patch_generation_20x.py` 裁剪 20x PNG

```bash
python patch_generation_20x.py \
  --source data/gdmuah/wsi \
  --csv dataset_csv/all_data_guangdongyi.csv \
  --coords-root data/gdmuah/patches_coords_20x \
  --patch-size 256 \
  --patch-level 1 \
  --output-root data/gdmuah/patches_20x \
  --workers 8 \
  --skip-existing
```

这一步的真实行为是：

- 从 `patches_coords_20x/patches_256/*.h5` 读坐标
- 在 `level=1` 上直接切 `256x256`
- 不做 resize

仍然不需要：

- `--read-size 1024`
- `--out-size 256`

因为当前这份脚本已经与 `patch_generation_5x.py` 保持统一的接口风格。

---

## 13. Step 7：检查 patch 裁剪结果是否完整

5x patch 目录检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_guangdongyi.csv \
  --dirs_root data/gdmuah/patches_5x \
  --output_path eval_results/gdmuah/gdmuah_csv_vs_patches5x.csv
```

20x patch 目录检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_guangdongyi.csv \
  --dirs_root data/gdmuah/patches_20x \
  --output_path eval_results/gdmuah/gdmuah_csv_vs_patches20x.csv
```

同时建议随机打开几个 slide 的 patch 子目录，确认：

1. patch 数量不是 0
2. patch 里不是大面积空白背景
3. 5x 和 20x 的视野差异符合预期

---

## 14. Step 8：提取 5x BiomedCLIP 特征

```bash
python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/gdmuah/patches_5x \
  --library_path data/gdmuah/features_biomedclip_5x \
  --batch_size 32 \
  --dataset adenocarcinoma
```

输出：

- `data/gdmuah/features_biomedclip_5x/*.h5`

---

## 15. Step 9：提取 20x BiomedCLIP 特征

```bash
python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/gdmuah/patches_20x \
  --library_path data/gdmuah/features_biomedclip_20x \
  --batch_size 32 \
  --dataset adenocarcinoma
```

输出：

- `data/gdmuah/features_biomedclip_20x/*.h5`

---

## 16. Step 10：检查特征文件是否完整

5x 特征检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_guangdongyi.csv \
  --dirs_root data/gdmuah/features_biomedclip_5x \
  --output_path eval_results/gdmuah/gdmuah_csv_vs_feat5x.csv
```

20x 特征检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_guangdongyi.csv \
  --dirs_root data/gdmuah/features_biomedclip_20x \
  --output_path eval_results/gdmuah/gdmuah_csv_vs_feat20x.csv
```

这一步很重要，因为外部评估最终是按特征文件是否齐全来决定能否完整推理。

---

## 17. Step 10.5（可选）：如果你要在广东医上重新训练 / 做内部验证，先生成严格五折划分

这一步**不属于当前外部验证主流程**，只有当你准备：

- 在广东医队列上重新训练模型
- 或者在广东医队列内部做严格五折交叉验证

时才需要执行。

当前仓库已经有严格划分脚本：

- `create_splits_strict_cv.py`

建议命令如下：

```bash
python create_splits_strict_cv.py \
  --csv_path dataset_csv/all_data_guangdongyi.csv \
  --save_dir splits/adenocarcinoma_gdmuah_strict5 \
  --k 5 \
  --seed 1 \
  --group_col case_id \
  --slide_col slide_id \
  --label_col label
```

说明：

- 这个脚本会按 `case_id` 分组后做严格五折，避免同一病例泄漏到 train / val / test
- 每个样本会且仅会有一次进入 test fold
- 生成结果包括：
  - `splits_0.csv` 到 `splits_4.csv`
  - `splits_i_descriptor.csv`
  - `strict_fold_assignments.csv`

前提条件：

- `dataset_csv/all_data_guangdongyi.csv` 必须包含 `case_id`、`slide_id`、`label` 三列
- 如果你的 CSV 当前不是这三个列名，需要先统一列名，或在命令里改对应参数

如果后续你真的要在广东医上训练 / 评估，那么训练命令和评估命令里的 `--split_dir` / `--splits_dir` 就应切换到：

- `splits/adenocarcinoma_gdmuah_strict5`

但对**当前这份外部验证流程**而言，这一步仍然是可选项，不应混入主流程。

---

## 18. Step 11：复用现有五折权重做广东医外部评估

```bash
python eval.py \
  --drop_out \
  --k 5 \
  --k_start 0 \
  --k_end 5 \
  --split all \
  --task task_adenocarcinoma \
  --results_dir trained_models \
  --models_exp_code adenocarcinoma_biomedclip_dual_strict5_s1 \
  --save_exp_code adenocarcinoma_biomedclip_dual_strict5_s1_gdmuah_external \
  --model_type ViLa_MIL_BiomedCLIP \
  --mode transformer \
  --splits_dir trained_models/adenocarcinoma_biomedclip_dual_strict5_s1 \
  --data_root_dir data/gdmuah \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --csv_path dataset_csv/all_data_guangdongyi.csv \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --prototype_number 16
```

说明：

- 外部广东医队列应使用 `--split all`
- 不要使用训练折的 `test` split 去筛外部 CSV
- 这里默认沿用河源同一套腺癌双尺度权重

输出目录通常为：

- `eval_results/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_gdmuah_external`

---

## 19. Step 12：把评估结果归档到 `eval_results/gdmuah`

```bash
mv eval_results/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_gdmuah_external \
   eval_results/gdmuah/
```

归档后的目录为：

- `eval_results/gdmuah/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_gdmuah_external`

---

## 20. Step 13：生成广东医外部评估报告图

```bash
python tools/generate_external_report_no_threshold.py \
  --external_dir eval_results/gdmuah/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_gdmuah_external \
  --plot_name GDMUAH \
  --negative_label Adenocarcinoma \
  --positive_label NonAdenocarcinoma
```

说明：

- 当前项目的标签映射里：
  - `Adenocarcinoma`
  - `NonAdenocarcinoma`
- 报告图里负类 / 正类命名应与已有实验保持一致

---

## 21. 可选：如果评估结果理想，再生成热图和 top patch

如果广东医外部评估结果稳定，再继续做可解释性分析更合适。

示例：

```bash
python biomedclip_generate_attention_heatmaps.py \
  --csv_path dataset_csv/all_data_guangdongyi.csv \
  --data_root_dir data/gdmuah \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --wsi_root data/gdmuah/wsi \
  --checkpoint trained_models/adenocarcinoma_biomedclip_dual_strict5_s1/s_2_checkpoint.pt \
  --results_dir eval_results/gdmuah/heatmap_top_patch_fold2_paper \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --prototype_number 16 \
  --task task_adenocarcinoma \
  --heatmap_style paper \
  --heatmap_cmap paper \
  --heatmap_alpha 0.78 \
  --wsi_suffix .svs
```

导出 top-k patch：

```bash
python export_top_patches_from_attention.py \
  --attention_h5_dir eval_results/gdmuah/heatmap_top_patch_fold2_paper/attention_h5 \
  --patches_5x_dir data/gdmuah/patches_5x \
  --patches_20x_dir data/gdmuah/patches_20x \
  --output_dir eval_results/gdmuah/overlay_heatmaps/top_patches \
  --top_k 10 \
  --slides_csv dataset_csv/all_data_guangdongyi.csv
```

---

## 21. 推荐执行顺序

建议按下面顺序推进：

1. 先跑 Step 0.5，确认 CSV 与 WSI 的匹配关系
2. 然后跑 Step 2 和 Step 3，先得到 5x / 20x 的 mask 和坐标
3. 抽查 `masks/` 和 `graph_256/` 的质量，再继续裁 patch
4. 跑 Step 5 和 Step 6 生成 5x / 20x patch
5. 跑 Step 8 和 Step 9 提取双尺度 BiomedCLIP 特征
6. 跑 Step 11 做广东医外部评估
7. 跑 Step 13 生成汇总报告图
8. 只有当评估结果可信后，再跑热图和 top patch

---

## 22. 建议重点关注的风险点

### 22.1 CSV 与 WSI 数量不完全一致

当前已知：

- CSV: 335
- WSI: 343

这意味着：

- 目录里很可能有额外 SVS 不参与本次评估
- 也不排除个别 CSV slide_id 与实际文件名不完全一致

所以 Step 0.5 必跑。

### 22.2 `gdmuah.csv` 仍然可能需要二次微调

当前已经给广东医单独建立了 `gdmuah.csv`，但如果你看到：

- mask 过度保留背景
- mask 漏掉大量组织
- graph 中 patch 分布明显异常

那就建议在现有 `presets/gdmuah.csv` 基础上继续微调：

- `seg_level`
- `sthresh`
- `mthresh`
- `close`
- `a_t`
- `a_h`

### 22.3 外部评估必须用 `--split all`

这是外部队列最容易出错的地方之一。

你现在这套广东医是完整外部队列，不属于训练五折中的任意一个内部测试折，所以应该：

- 使用完整 CSV
- 使用 `--split all`

---

## 23. 最终你应当拿到的结果

完整跑完后，至少应有以下产物：

- 坐标与分割结果：
  - `data/gdmuah/patches_coords_5x`
  - `data/gdmuah/patches_coords_20x`
- patch PNG：
  - `data/gdmuah/patches_5x`
  - `data/gdmuah/patches_20x`
- BiomedCLIP 特征：
  - `data/gdmuah/features_biomedclip_5x`
  - `data/gdmuah/features_biomedclip_20x`
- 外部评估目录：
  - `eval_results/gdmuah/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_gdmuah_external`
- 各类预检查结果：
  - `eval_results/gdmuah/gdmuah_csv_vs_wsi.csv`
  - `eval_results/gdmuah/gdmuah_csv_vs_coords5x.csv`
  - `eval_results/gdmuah/gdmuah_csv_vs_coords20x.csv`
  - `eval_results/gdmuah/gdmuah_csv_vs_patches5x.csv`
  - `eval_results/gdmuah/gdmuah_csv_vs_patches20x.csv`
  - `eval_results/gdmuah/gdmuah_csv_vs_feat5x.csv`
  - `eval_results/gdmuah/gdmuah_csv_vs_feat20x.csv`

---

## 24. 查看本手册

```bash
sed -n '1,260p' docs/gdmuah_external_validation_full_pipeline_cn.md
```
