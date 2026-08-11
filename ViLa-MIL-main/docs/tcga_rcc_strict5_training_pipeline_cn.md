# TCGA-RCC 严格五折训练与结果查看流程

本文档基于当前仓库中的真实脚本状态，整理一套可直接执行的 `TCGA-RCC` 双尺度 BiomedCLIP 训练流程。

目标是：

1. 使用 `data/TCGA-RCC/wsi` 作为 WSI 根目录
2. 使用 `dataset_csv/TCGA_RCC_subtyping.csv` 作为训练 CSV
3. 重新生成 5x / 20x 坐标、patch PNG 和 BiomedCLIP 特征
4. 按 `case_id` 做严格五折交叉划分，避免同一病例的多张 slide 泄漏到不同集合
5. 在五折训练完成后，查看每折与整体平均结果

---

## 1. 当前已确认的输入信息

- WSI 根目录：
  - `data/TCGA-RCC/wsi`
- 训练 CSV：
  - `dataset_csv/TCGA_RCC_subtyping.csv`
- 推荐分割 preset：
  - `presets/tcga.csv`
- 当前代码兼容的双尺度 prompt：
  - `text_prompt/TCGA_RCC_two_scale_text_prompt_structured.csv`

截至当前仓库状态：

- `TCGA_RCC_subtyping.csv` 共 `939` 条 slide 记录
- 唯一 `case_id` 数量为 `897`
- 存在 `28` 个 `case_id` 对应多张 slide
- 单个 case 最多有 `9` 张 slide

这意味着：

- 不能把 slide 当作彼此独立样本去随机划分
- 必须按 `case_id` 分组做严格五折，否则会出现病例泄漏

---

## 2. 本次流程的总体策略

沿用当前仓库的双尺度定义：

- `5x` 支路：
  - `patch_level=2`
  - 直接裁 `256x256`
- `20x` 支路：
  - `patch_level=1`
  - 直接裁 `256x256`

严格五折部分使用：

- `create_splits_strict_cv.py`
- `group_col=case_id`
- `slide_col=slide_id`
- `label_col=label`

它的规则是：

- 第 `i` 折的 `test = fold i`
- 第 `i` 折的 `val = fold (i + 1) % 5`
- 剩余折为 `train`

因此：

- 每个 `case_id` 只会在一个 test fold 中出现一次
- train / val / test 在每折内严格不重叠

---

## 3. 建议使用的目录

仓库根目录：

- `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main`

本次建议输出目录：

- `data/TCGA-RCC/patches_coords_5x`
- `data/TCGA-RCC/patches_coords_20x`
- `data/TCGA-RCC/patches_5x`
- `data/TCGA-RCC/patches_20x`
- `data/TCGA-RCC/features_biomedclip_5x`
- `data/TCGA-RCC/features_biomedclip_20x`
- `splits/tcga_rcc_strict5_case_seed1`
- `results/tcga_rcc_strict5`

---

## 4. 运行前约定

以下命令默认在仓库根目录执行：

```bash
cd /private/ljh-data/shared/ViLMIL/ViLa-MIL-main
```

---

## 5. Step 0：创建输出目录

```bash
mkdir -p data/TCGA-RCC/patches_coords_5x
mkdir -p data/TCGA-RCC/patches_coords_20x
mkdir -p data/TCGA-RCC/patches_5x
mkdir -p data/TCGA-RCC/patches_20x
mkdir -p data/TCGA-RCC/features_biomedclip_5x
mkdir -p data/TCGA-RCC/features_biomedclip_20x
mkdir -p splits/tcga_rcc_strict5_case_seed1
mkdir -p results/tcga_rcc_strict5
```

---

## 6. Step 0.5：先做 CSV 与 WSI 一致性检查

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/TCGA_RCC_subtyping.csv \
  --dirs_root data/TCGA-RCC/wsi \
  --output_path results/tcga_rcc_strict5/tcga_rcc_csv_vs_wsi.csv
```

重点关注：

- `missing_in_dirs`
- `extra_in_dirs`

如果 `missing_in_dirs > 0`，后续某些 slide 会直接无法生成坐标或特征。

---

## 7. Step 1：生成 5x 坐标和分割结果

```bash
python create_patches_fp_heyuan.py \
  --source data/TCGA-RCC/wsi \
  --csv dataset_csv/TCGA_RCC_subtyping.csv \
  --save-dir data/TCGA-RCC/patches_coords_5x \
  --preset tcga.csv \
  --patch-size 256 \
  --step-size 256 \
  --patch-level 2 \
  --seg \
  --patch \
  --stitch
```

输出重点：

- `data/TCGA-RCC/patches_coords_5x/patches_256/*.h5`
- `data/TCGA-RCC/patches_coords_5x/masks/*.jpg`
- `data/TCGA-RCC/patches_coords_5x/graph_256/*.jpg`

建议先抽查 10 到 20 张：

- 组织区域是否被正确圈出
- 背景是否被过度保留
- patch 分布是否贴合组织

---

## 8. Step 2：生成 20x 坐标和分割结果

```bash
python create_patches_fp_heyuan.py \
  --source data/TCGA-RCC/wsi \
  --csv dataset_csv/TCGA_RCC_subtyping.csv \
  --save-dir data/TCGA-RCC/patches_coords_20x \
  --preset tcga.csv \
  --patch-size 256 \
  --step-size 256 \
  --patch-level 1 \
  --seg \
  --patch \
  --stitch
```

输出重点：

- `data/TCGA-RCC/patches_coords_20x/patches_256/*.h5`
- `data/TCGA-RCC/patches_coords_20x/masks/*.jpg`
- `data/TCGA-RCC/patches_coords_20x/graph_256/*.jpg`

---

## 9. Step 3：裁剪 5x PNG patch

```bash
python patch_generation_5x.py \
  --source data/TCGA-RCC/wsi \
  --csv dataset_csv/TCGA_RCC_subtyping.csv \
  --coords-root data/TCGA-RCC/patches_coords_5x \
  --patch-size 256 \
  --patch-level 2 \
  --output-root data/TCGA-RCC/patches_5x \
  --workers 8 \
  --skip-existing
```

---

## 10. Step 4：裁剪 20x PNG patch

```bash
python patch_generation_20x.py \
  --source data/TCGA-RCC/wsi \
  --csv dataset_csv/TCGA_RCC_subtyping.csv \
  --coords-root data/TCGA-RCC/patches_coords_20x \
  --patch-size 256 \
  --patch-level 1 \
  --output-root data/TCGA-RCC/patches_20x \
  --workers 8 \
  --skip-existing
```

---

## 11. Step 5：提取 5x BiomedCLIP 特征

```bash
python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/TCGA-RCC/patches_5x \
  --library_path data/TCGA-RCC/features_biomedclip_5x \
  --batch_size 32 \
  --dataset tcga_rcc
```

输出：

- `data/TCGA-RCC/features_biomedclip_5x/*.h5`

---

## 12. Step 6：提取 20x BiomedCLIP 特征

```bash
python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/TCGA-RCC/patches_20x \
  --library_path data/TCGA-RCC/features_biomedclip_20x \
  --batch_size 32 \
  --dataset tcga_rcc
```

输出：

- `data/TCGA-RCC/features_biomedclip_20x/*.h5`

---

## 13. Step 7：生成按 case 分组的严格五折划分

```bash
python create_splits_strict_cv.py \
  --csv_path dataset_csv/TCGA_RCC_subtyping.csv \
  --save_dir splits/tcga_rcc_strict5_case_seed1 \
  --k 5 \
  --seed 1 \
  --group_col case_id \
  --slide_col slide_id \
  --label_col label \
  --strict_single_label_per_group
```

输出目录包含：

- `splits_0.csv` 到 `splits_4.csv`
- `splits_0_bool.csv` 到 `splits_4_bool.csv`
- `splits_0_descriptor.csv` 到 `splits_4_descriptor.csv`
- `strict_fold_assignments.csv`

建议重点看：

- `strict_fold_assignments.csv`
- `splits_i_descriptor.csv`

确认每折类别分布没有明显失衡。

---

## 14. Step 8：开始严格五折训练

```bash
python main.py \
  --data_root_dir data/TCGA-RCC \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --model_type ViLa_MIL_BiomedCLIP \
  --mode transformer \
  --task task_tcga_rcc_subtyping \
  --text_prompt_path text_prompt/TCGA_RCC_two_scale_text_prompt_structured.csv \
  --prototype_number 16 \
  --split_dir splits/tcga_rcc_strict5_case_seed1 \
  --results_dir results/tcga_rcc_strict5 \
  --exp_code tcga_rcc_biomedclip_dual_strict5 \
  --k 5 \
  --seed 1 \
  --lr 1e-4 \
  --early_stopping \
  --drop_out
```

训练输出目录会是：

- `results/tcga_rcc_strict5/tcga_rcc_biomedclip_dual_strict5_s1`

其中典型输出包括：

- `s_0_checkpoint.pt` 到 `s_4_checkpoint.pt`
- `splits_0.csv` 到 `splits_4.csv`
- `fold_summary.csv`
- `epoch_details.csv`
- `summary.csv`
- `result.csv`

---

## 15. Step 9：先看训练阶段直接产出的结果

训练结束后，优先看：

- `results/tcga_rcc_strict5/tcga_rcc_biomedclip_dual_strict5_s1/fold_summary.csv`
- `results/tcga_rcc_strict5/tcga_rcc_biomedclip_dual_strict5_s1/result.csv`

其中：

- `fold_summary.csv` 是每折 test / val 指标
- `result.csv` 是五折平均值和标准差

---

## 16. Step 10：用保存下来的五折 checkpoint 重新评估 test fold

这一步不是必须的，但我建议保留，因为它会额外生成每折预测明细和误分类清单。

```bash
python eval.py \
  --drop_out \
  --k 5 \
  --k_start 0 \
  --k_end 5 \
  --split test \
  --task task_tcga_rcc_subtyping \
  --results_dir results/tcga_rcc_strict5 \
  --models_exp_code tcga_rcc_biomedclip_dual_strict5_s1 \
  --save_exp_code tcga_rcc_biomedclip_dual_strict5_eval \
  --model_type ViLa_MIL_BiomedCLIP \
  --mode transformer \
  --splits_dir splits/tcga_rcc_strict5_case_seed1 \
  --data_root_dir data/TCGA-RCC \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --text_prompt_path text_prompt/TCGA_RCC_two_scale_text_prompt_structured.csv \
  --prototype_number 16
```

输出目录通常为：

- `eval_results/EVAL_tcga_rcc_biomedclip_dual_strict5_eval`

典型文件包括：

- `fold_0.csv` 到 `fold_4.csv`
- `fold_0_misclassified.csv` 到 `fold_4_misclassified.csv`
- `summary.csv`
- `result.csv`
- `timing_details.csv`

---

## 17. 结果查看建议

如果你只想快速看五折结果，优先看：

- `results/tcga_rcc_strict5/tcga_rcc_biomedclip_dual_strict5_s1/result.csv`

如果你想看每折预测明细，优先看：

- `eval_results/EVAL_tcga_rcc_biomedclip_dual_strict5_eval/fold_*.csv`

如果你想看误分类 slide：

- `eval_results/EVAL_tcga_rcc_biomedclip_dual_strict5_eval/fold_*_misclassified.csv`

---

## 18. 容易踩坑的地方

1. `TCGA_RCC_subtyping.csv` 里存在一个 `case_id` 对应多张 slide 的情况，所以 split 一定要按 `case_id` 分组。
2. 当前仓库里的旧文件 `text_prompt/TCGA_RCC_two_scale_text_prompt.csv` 是单列旧格式，不建议直接用于 `main.py` 和 `eval.py`。
3. 本文档使用的是新增的 `text_prompt/TCGA_RCC_two_scale_text_prompt_structured.csv`，它与当前 parser 兼容。
4. `main.py` 中 `--k_end` 是包含语义，但本流程直接跑全部五折，因此训练阶段不需要传 `--k_start/--k_end`。
5. `eval.py` 中 `--k_end` 更接近 Python `range` 的结束位，所以五折评估用 `--k_start 0 --k_end 5`。
6. `TCGA_RCC_subtyping.csv` 中有少量 `case_id` 写成类似 `TCGA-UW-A7GX, TCGA-UW-A7GX` 的重复字符串形式；当前不影响按原 CSV 做 case 级 split，但如果你后续手动清洗 `case_id`，记得重新生成 strict splits。
7. `presets/tcga.csv` 是一个合理起点，但仍建议先抽样核对 masks 和 graph 图，再批量跑完整队列。
