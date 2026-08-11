# 河源外部队列重跑手册（level 2 / level 1 均直接切 256x256）

本文档基于当前仓库里的真实代码状态重写，目标是：

1. 用 `presets/heyuan.csv` 重新生成河源外部队列的组织分割和 patch 坐标
2. 在 `level=2` 和 `level=1` 两条支路上，都直接裁剪 `256x256` patch
3. 重新提取 BiomedCLIP 双尺度特征
4. 复用现有五折权重重新做河源外部评估

这份新版手册不再假设 `patch_generation_20x.py` 使用 `read_size=1024 -> out_size=256`。你当前恢复出来的原始 `patch_generation_20x.py`，实际也是直接在 `level=1` 读取 `256x256`。

---

## 1. 当前两个裁 patch 脚本的真实行为

### 1.1 `patch_generation_5x.py`

这个脚本现在是一个通用命令行脚本，适合直接用于河源外部队列。

它的核心逻辑是：

- 从 `--coords-root/patches_<patch_size>/<slide_id>.h5` 读取坐标
- 对每个坐标执行：
  - `read_region((x, y), patch_level, (patch_size, patch_size))`
- 直接保存为 PNG，不做额外 resize

对于河源这次重跑，如果你用：

- `--patch-size 256`
- `--patch-level 2`

那么它的含义就是：

- 在 `level=2` 上直接切 `256x256`
- 输出文件名通常为：
  - `<slide_id>_<x>_<y>.png`

这个脚本的特点是：

- 已支持命令行参数
- 已支持 `--workers`
- 已支持 `--skip-existing`
- 可直接对接 `create_patches_fp_heyuan.py` 的输出目录

### 1.2 `patch_generation_20x.py`

它当前的真实逻辑是：

- 从 `--coords-root/patches_<patch_size>/<slide_id>.h5` 读取坐标
- 对每个坐标执行：
  - `read_region((x, y), patch_level, (patch_size, patch_size))`
- 直接保存为 PNG

也就是说，这个脚本当前做的是：

- 在指定的 `patch_level` 上直接切指定大小的 patch
- 对于河源这次重跑，使用：
  - `--patch-size 256`
  - `--patch-level 1`
- 不是先读 `1024x1024` 再缩成 `256x256`

输出文件名通常为：

- `<slide_id>_<x>_<y>.png`

它现在也已经支持：

- 命令行参数
- `--workers`
- `--skip-existing`

现在 `patch_generation_5x.py` 和 `patch_generation_20x.py` 已统一为同样的命名风格：

- `<slide_id>_<x>_<y>.png`

---

## 2. 哪一种更合理

如果你的目标是：

- 外部队列的低倍支路和高倍支路都保持“按坐标直接切 patch”
- `level=2` 和 `level=1` 都切真实的 `256x256`
- 不希望高倍支路再混入更大视野再缩放的隐式定义

那么当前这套“两个 level 都直接切 `256x256`”更合理。

原因是：

1. 它和 `create_patches_fp_heyuan.py` 生成坐标时的 `patch_size=256` 语义一致。
2. 两条支路的差异只来自 WSI level，不再额外混入 `1024 -> 256` 的视野变化。
3. 结果更容易解释：`level=2` 是低倍 256 patch，`level=1` 是高倍 256 patch。
4. 后续如果要核对 patch 数量、坐标对应关系、热图映射，都会更直接。

只有一种情况例外：

- 如果你要严格复现某个历史实验，而那个历史实验的高倍 patch 定义本来就是“先读更大区域，再缩成 256”，那就应该保留旧定义。

但那属于“复现实验定义”，不是“哪种更清晰、更自然”。
就你现在这次河源外部队列重跑的目标而言，我建议采用：

- `5x`: `level=2` 直接切 `256x256`
- `20x`: `level=1` 直接切 `256x256`

---

## 3. 本次会用到的核心目录

仓库根目录：

- `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main`

本次关键输入：

- 河源 WSI 根目录：
  - `data/heyuan/wsi`
- 河源 CSV：
  - `dataset_csv/all_data_heyuan.csv`
- 河源预设：
  - `presets/heyuan.csv`
- 文本提示：
  - `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- 五折权重：
  - `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`

本次建议新输出目录：

- `data/heyuan/reseg/patches_coords_5x`
- `data/heyuan/reseg/patches_coords_20x`
- `data/heyuan/reseg/patches_5x`
- `data/heyuan/reseg/patches_20x`
- `data/heyuan/reseg/features_biomedclip_5x`
- `data/heyuan/reseg/features_biomedclip_20x`

---

## 4. 运行前约定

以下命令默认在仓库根目录执行：

```bash
cd /private/ljh-data/shared/ViLMIL/ViLa-MIL-main
```

---

## 5. Step 0：创建输出目录

```bash
mkdir -p data/heyuan/reseg/patches_coords_5x
mkdir -p data/heyuan/reseg/patches_coords_20x
mkdir -p data/heyuan/reseg/patches_5x
mkdir -p data/heyuan/reseg/patches_20x
mkdir -p data/heyuan/reseg/features_biomedclip_5x
mkdir -p data/heyuan/reseg/features_biomedclip_20x
mkdir -p eval_results/heyuan
```

---

## 6. Step 1：重新生成 5x 坐标和分割结果

```bash
python create_patches_fp_heyuan.py \
  --source data/heyuan/wsi \
  --csv dataset_csv/all_data_heyuan.csv \
  --save-dir data/heyuan/reseg/patches_coords_5x \
  --preset heyuan.csv \
  --patch-size 256 \
  --step-size 256 \
  --patch-level 2 \
  --seg \
  --patch \
  --stitch
```

输出重点：

- `data/heyuan/reseg/patches_coords_5x/patches_256/*.h5`
- `data/heyuan/reseg/patches_coords_5x/masks/*.jpg`
- `data/heyuan/reseg/patches_coords_5x/graph_256/*.jpg`

---

## 7. Step 2：重新生成 20x 坐标和分割结果

```bash
python create_patches_fp_heyuan.py \
  --source data/heyuan/wsi \
  --csv dataset_csv/all_data_heyuan.csv \
  --save-dir data/heyuan/reseg/patches_coords_20x \
  --preset heyuan.csv \
  --patch-size 256 \
  --step-size 256 \
  --patch-level 1 \
  --seg \
  --patch \
  --stitch
```

输出重点：

- `data/heyuan/reseg/patches_coords_20x/patches_256/*.h5`
- `data/heyuan/reseg/patches_coords_20x/masks/*.jpg`
- `data/heyuan/reseg/patches_coords_20x/graph_256/*.jpg`

---

## 8. Step 3：用 `patch_generation_5x.py` 裁剪 5x PNG

```bash
python patch_generation_5x.py \
  --source data/heyuan/wsi \
  --csv dataset_csv/all_data_heyuan.csv \
  --coords-root data/heyuan/reseg/patches_coords_5x \
  --patch-size 256 \
  --patch-level 2 \
  --output-root data/heyuan/reseg/patches_5x \
  --workers 8 \
  --skip-existing
```

这一步的真实行为是：

- 从 `patches_coords_5x/patches_256/*.h5` 读坐标
- 在 `level=2` 上直接切 `256x256`
- 不做 resize

---

## 9. Step 4：用 `patch_generation_20x.py` 裁剪 20x PNG

```bash
python patch_generation_20x.py \
  --source data/heyuan/wsi \
  --csv dataset_csv/all_data_heyuan.csv \
  --coords-root data/heyuan/reseg/patches_coords_20x \
  --patch-size 256 \
  --patch-level 1 \
  --output-root data/heyuan/reseg/patches_20x \
  --workers 8 \
  --skip-existing
```

这一步的真实行为是：

- 从 `patches_coords_20x/patches_256/*.h5` 读坐标
- 在 `level=1` 上直接切 `256x256`
- 不做 resize

这一步不再出现：

- `--read-size 1024`
- `--out-size 256`

因为当前这份脚本已经改成和 `patch_generation_5x.py` 同风格的命令行接口，但仍保留“直接裁 `256x256`”的行为。

---

## 10. Step 5：提取 5x BiomedCLIP 特征

```bash
python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/heyuan/reseg/patches_5x \
  --library_path data/heyuan/reseg/features_biomedclip_5x \
  --batch_size 32 \
  --dataset adenocarcinoma
```

输出：

- `data/heyuan/reseg/features_biomedclip_5x/*.h5`

---

## 11. Step 6：提取 20x BiomedCLIP 特征

```bash
python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/heyuan/reseg/patches_20x \
  --library_path data/heyuan/reseg/features_biomedclip_20x \
  --batch_size 32 \
  --dataset adenocarcinoma
```

输出：

- `data/heyuan/reseg/features_biomedclip_20x/*.h5`

---

## 12. Step 7：复用现有五折权重做河源外部评估

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
  --save_exp_code adenocarcinoma_biomedclip_dual_strict5_s1_heyuan_external_reseg \
  --model_type ViLa_MIL_BiomedCLIP \
  --mode transformer \
  --splits_dir trained_models/adenocarcinoma_biomedclip_dual_strict5_s1 \
  --data_root_dir data/heyuan/reseg \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --csv_path dataset_csv/all_data_heyuan.csv \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --prototype_number 16
```

说明：

- 外部河源队列应使用 `--split all`
- 不要再用训练折的 `test` split 去筛外部 CSV

输出目录通常为：

- `eval_results/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_heyuan_external_reseg`

---

## 13. Step 8：把评估结果归档到 `eval_results/heyuan`

```bash
mv eval_results/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_heyuan_external_reseg \
   eval_results/heyuan/
```

---

## 14. Step 9：生成新的河源外部评估报告图

```bash
python tools/generate_external_report_no_threshold.py \
  --external_dir eval_results/heyuan/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_heyuan_external_reseg \
  --plot_name Heyuan \
  --negative_label Adenocarcinoma \
  --positive_label NonAdenocarcinoma
```

---

## 15. 可选：评估改善后再生成热图和 top patch

如果新的外部评估明显改善，再继续生成热图和 top patch 更合适。

示例：

```bash
python biomedclip_generate_attention_heatmaps.py \
  --csv_path dataset_csv/all_data_heyuan.csv \
  --data_root_dir data/heyuan/reseg \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --wsi_root data/heyuan/wsi \
  --checkpoint trained_models/adenocarcinoma_biomedclip_dual_strict5_s1/s_2_checkpoint.pt \
  --results_dir eval_results/heyuan/heatmap_top_patch_fold2_paper_reseg \
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
  --attention_h5_dir eval_results/heyuan/heatmap_top_patch_fold2_paper_reseg/attention_h5 \
  --patches_5x_dir data/heyuan/reseg/patches_5x \
  --patches_20x_dir data/heyuan/reseg/patches_20x \
  --output_dir eval_results/heyuan/overlay_heatmaps_reseg/top_patches \
  --top_k 10 \
  --slides_csv dataset_csv/all_data_heyuan.csv
```

---

## 16. 推荐执行顺序

建议按下面顺序推进：

1. 先跑 Step 1 和 Step 2，检查新的 `masks/` 和 `graph_256/`
2. 再跑 Step 3 和 Step 4，抽查 patch 数量和视野是否符合预期
3. 然后跑 Step 5 和 Step 6 提特征
4. 最后跑 Step 7 到 Step 9 做外部评估和汇总图
5. 只有评估结果改善后，再跑热图和 top patch

---

## 17. 这次重写后的关键结论

这次文档统一按当前真实代码写成：

- `5x` 支路：
  - `patch_level=2`
  - 直接切 `256x256`
- `20x` 支路：
  - `patch_level=1`
  - 直接切 `256x256`

不再写成：

- 先读 `1024x1024`
- 再缩成 `256x256`

因为这和你当前恢复出来的 `patch_generation_20x.py` 实际行为不一致。

---

## 18. 查看本手册

```bash
sed -n '1,260p' docs/heyuan_rerun_with_new_preset_cn.md
```
