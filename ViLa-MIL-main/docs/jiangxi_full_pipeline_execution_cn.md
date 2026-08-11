# 江西外部队列全链路执行手册

本文档基于当前仓库的真实代码状态编写，目标是把江西外部队列从：

1. 组织分割与双尺度坐标生成
2. 双尺度 patch 裁剪
3. BiomedCLIP 双尺度特征提取
4. 五折外部评估

完整串起来，并把每一步所需的：

- 环境
- 输入
- 输出
- 执行命令

统一汇总到一处，便于后续直接照着执行。

---

## 1. 当前采用的江西方案

### 1.1 数据格式现实

训练集 `yiyuan` 的 WSI 是多层 `svs`。

江西外部集 `jiangxi` 的 WSI 当前是单层 `tif / OME-TIFF`。

因此，江西不能直接复用训练集的：

- 高倍：`level=1, patch_size=256`
- 低倍：`level=2, patch_size=256`

因为江西 TIFF 当前没有对应的多层金字塔 level。

### 1.2 当前采用的等效尺度定义

为尽量逼近训练时两条分支在 `level0` 上看到的物理视野，当前江西方案固定为：

- 高倍支路：
  - `patch_size=1024`
  - `step_size=1024`
  - `patch_level=0`
  - 后续裁 patch 时缩成 `256 x 256`
- 低倍支路：
  - `patch_size=4096`
  - `step_size=4096`
  - `patch_level=0`
  - 后续裁 patch 时缩成 `256 x 256`

### 1.3 当前采用的正式江西 preset

当前正式使用：

- `presets/jiangxi.csv`

其内容目前为：

```csv
seg_level,sthresh,mthresh,close,use_otsu,a_t,a_h,max_n_holes,vis_level,line_thickness,white_thresh,black_thresh,use_padding,contour_fn,keep_ids,exclude_ids
-1,6,7,6,TRUE,8,2,12,-1,100,5,50,TRUE,four_pt,none,none
```

这是基于江西 5 张 pilot A/B 对比后选定的当前版本。

---

## 2. 环境与路径

### 2.1 Python 环境

统一使用：

- `/home/ljh/anaconda3/envs/vila_mil/bin/python`

### 2.2 仓库根目录

```bash
cd /private/ljh-data/shared/ViLMIL/ViLa-MIL-main
```

### 2.3 关键输入

- 江西 WSI 根目录：
  - `data/jiangxi/wsi`
- 江西 CSV：
  - `dataset_csv/all_data_jiangxi.csv`
- 文本提示：
  - `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- 五折训练权重目录：
  - `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`

### 2.4 建议输出目录

本手册统一使用新的全量输出目录：

- `data/jiangxi/reseg_v2_full/patches_coords_20x_eq1024`
- `data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096`
- `data/jiangxi/reseg_v2_full/patches_20x_eq1024_to256`
- `data/jiangxi/reseg_v2_full/patches_5x_eq4096_to256`
- `data/jiangxi/reseg_v2_full/features_biomedclip_20x_eq1024_to256`
- `data/jiangxi/reseg_v2_full/features_biomedclip_5x_eq4096_to256`
- `eval_results/jiangxi/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_jiangxi_rerun_v2`

---

## 3. Step 0：创建输出目录

### 输入

- 无

### 输出

- 创建全量输出目录结构

### 命令

```bash
mkdir -p data/jiangxi/reseg_v2_full/patches_coords_20x_eq1024
mkdir -p data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096
mkdir -p data/jiangxi/reseg_v2_full/patches_20x_eq1024_to256
mkdir -p data/jiangxi/reseg_v2_full/patches_5x_eq4096_to256
mkdir -p data/jiangxi/reseg_v2_full/features_biomedclip_20x_eq1024_to256
mkdir -p data/jiangxi/reseg_v2_full/features_biomedclip_5x_eq4096_to256
mkdir -p eval_results/jiangxi
```

---

## 4. Step 1：生成高倍支路坐标（1024 等效高倍）

### 脚本

- `create_patches_fp_jiangxi.py`

### 输入

- WSI：
  - `data/jiangxi/wsi`
- CSV：
  - `dataset_csv/all_data_jiangxi.csv`
- preset：
  - `presets/jiangxi.csv`

### 输出

- 坐标 H5：
  - `data/jiangxi/reseg_v2_full/patches_coords_20x_eq1024/patches_1024/*.h5`
- 分割掩膜：
  - `data/jiangxi/reseg_v2_full/patches_coords_20x_eq1024/masks/*.jpg`
- 仅轮廓图：
  - `data/jiangxi/reseg_v2_full/patches_coords_20x_eq1024/only_masks/*.png`
- 流程记录：
  - `data/jiangxi/reseg_v2_full/patches_coords_20x_eq1024/process_list_autogen.csv`

### 命令

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python create_patches_fp_jiangxi.py \
  --source data/jiangxi/wsi \
  --csv dataset_csv/all_data_jiangxi.csv \
  --save-dir data/jiangxi/reseg_v2_full/patches_coords_20x_eq1024 \
  --preset jiangxi.csv \
  --patch-size 1024 \
  --step-size 1024 \
  --patch-level 0 \
  --seg \
  --patch
```

---

## 5. Step 2：生成低倍支路坐标（4096 等效低倍）

### 脚本

- `create_patches_fp_jiangxi.py`

### 输入

- WSI：
  - `data/jiangxi/wsi`
- CSV：
  - `dataset_csv/all_data_jiangxi.csv`
- preset：
  - `presets/jiangxi.csv`

### 输出

- 坐标 H5：
  - `data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096/patches_4096/*.h5`
- 分割掩膜：
  - `data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096/masks/*.jpg`
- 仅轮廓图：
  - `data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096/only_masks/*.png`
- 流程记录：
  - `data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096/process_list_autogen.csv`

### 命令

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python create_patches_fp_jiangxi.py \
  --source data/jiangxi/wsi \
  --csv dataset_csv/all_data_jiangxi.csv \
  --save-dir data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096 \
  --preset jiangxi.csv \
  --patch-size 4096 \
  --step-size 4096 \
  --patch-level 0 \
  --seg \
  --patch
```

---

## 6. Step 3：裁剪高倍 patch（1024 读图，缩到 256）

### 脚本

- `patch_generation_20x_jiangxi.py`

### 输入

- WSI：
  - `data/jiangxi/wsi`
- CSV：
  - `dataset_csv/all_data_jiangxi.csv`
- 坐标目录：
  - `data/jiangxi/reseg_v2_full/patches_coords_20x_eq1024`

### 输出

- PNG patch：
  - `data/jiangxi/reseg_v2_full/patches_20x_eq1024_to256/<slide_id>/*.png`

### 命令

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python patch_generation_20x_jiangxi.py \
  --source data/jiangxi/wsi \
  --csv dataset_csv/all_data_jiangxi.csv \
  --coords-root data/jiangxi/reseg_v2_full/patches_coords_20x_eq1024 \
  --coords-size 1024 \
  --patch-level 0 \
  --out-size 256 \
  --output-root data/jiangxi/reseg_v2_full/patches_20x_eq1024_to256 \
  --workers 8 \
  --skip-existing
```

---

## 7. Step 4：裁剪低倍 patch（4096 读图，缩到 256）

### 脚本

- `patch_generation_5x_jiangxi.py`

### 输入

- WSI：
  - `data/jiangxi/wsi`
- CSV：
  - `dataset_csv/all_data_jiangxi.csv`
- 坐标目录：
  - `data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096`

### 输出

- PNG patch：
  - `data/jiangxi/reseg_v2_full/patches_5x_eq4096_to256/<slide_id>/*.png`

### 命令

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python patch_generation_5x_jiangxi.py \
  --source data/jiangxi/wsi \
  --csv dataset_csv/all_data_jiangxi.csv \
  --coords-root data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096 \
  --coords-size 4096 \
  --patch-level 0 \
  --out-size 256 \
  --output-root data/jiangxi/reseg_v2_full/patches_5x_eq4096_to256 \
  --workers 4 \
  --skip-existing
```

说明：

- 低倍支路每个 patch 更大，单张切片的 IO 负担更重。
- 建议低倍 `workers` 先从 `4` 开始，不要默认开太高。

---

## 8. Step 5：提取高倍 BiomedCLIP 特征

### 脚本

- `feature_extraction/patch_extraction_biomedclip.py`

### 输入

- patch 目录：
  - `data/jiangxi/reseg_v2_full/patches_20x_eq1024_to256`

### 输出

- H5 特征：
  - `data/jiangxi/reseg_v2_full/features_biomedclip_20x_eq1024_to256/*.h5`

### 命令

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/jiangxi/reseg_v2_full/patches_20x_eq1024_to256 \
  --library_path data/jiangxi/reseg_v2_full/features_biomedclip_20x_eq1024_to256 \
  --batch_size 32 \
  --dataset adenocarcinoma
```

---

## 9. Step 6：提取低倍 BiomedCLIP 特征

### 脚本

- `feature_extraction/patch_extraction_biomedclip.py`

### 输入

- patch 目录：
  - `data/jiangxi/reseg_v2_full/patches_5x_eq4096_to256`

### 输出

- H5 特征：
  - `data/jiangxi/reseg_v2_full/features_biomedclip_5x_eq4096_to256/*.h5`

### 命令

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/jiangxi/reseg_v2_full/patches_5x_eq4096_to256 \
  --library_path data/jiangxi/reseg_v2_full/features_biomedclip_5x_eq4096_to256 \
  --batch_size 32 \
  --dataset adenocarcinoma
```

---

## 10. Step 7：检查特征目录是否齐全

### 输入

- 高倍特征目录
- 低倍特征目录
- 江西 CSV

### 输出

- 人工确认：
  - 两个特征目录下 `.h5` 数量是否和 CSV 中 slide 数量一致
  - 是否存在某一支路缺失

### 建议检查命令

```bash
find data/jiangxi/reseg_v2_full/features_biomedclip_20x_eq1024_to256 -name '*.h5' | wc -l
find data/jiangxi/reseg_v2_full/features_biomedclip_5x_eq4096_to256 -name '*.h5' | wc -l
wc -l dataset_csv/all_data_jiangxi.csv
```

说明：

- `wc -l` 的 CSV 结果会包含表头，所以 slide 数量应为 `行数 - 1`

---

## 11. Step 8：执行江西外部评估

### 脚本

- `eval.py`

### 输入

- CSV：
  - `dataset_csv/all_data_jiangxi.csv`
- 高倍特征目录：
  - `data/jiangxi/reseg_v2_full/features_biomedclip_20x_eq1024_to256`
- 低倍特征目录：
  - `data/jiangxi/reseg_v2_full/features_biomedclip_5x_eq4096_to256`
- 五折训练目录：
  - `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`
- 文本提示：
  - `text_prompt/adenocarcinoma_dual_scale_prompt.csv`

### 输出

- 评估结果目录：
  - `eval_results/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_jiangxi_rerun_v2`

目录内典型输出包括：

- `fold_0.csv` 到 `fold_4.csv`
- `fold_0_misclassified.csv` 到 `fold_4_misclassified.csv`
- `summary.csv`
- `result.csv`
- `timing_details.csv`

### 命令

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python eval.py \
  --data_root_dir data/jiangxi/reseg_v2_full \
  --data_folder_s features_biomedclip_5x_eq4096_to256 \
  --data_folder_l features_biomedclip_20x_eq1024_to256 \
  --results_dir trained_models \
  --models_exp_code adenocarcinoma_biomedclip_dual_strict5_s1 \
  --save_exp_code adenocarcinoma_biomedclip_dual_strict5_s1_jiangxi_rerun_v2 \
  --model_type ViLa_MIL_BiomedCLIP \
  --task task_adenocarcinoma \
  --split all \
  --k 5 \
  --csv_path dataset_csv/all_data_jiangxi.csv \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --prototype_number 16
```

说明：

- 这里 `--results_dir` 必须指向 `trained_models`
- 因为 `eval.py` 内部会拼成：
  - `args.models_dir = os.path.join(args.results_dir, args.models_exp_code)`
- 所以最终加载的是：
  - `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`

---

## 12. Step 9：如需先做单折测试

如果想先快速验证评估链路，而不是一上来跑 5 折，可以先跑 `fold 0`：

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python eval.py \
  --data_root_dir data/jiangxi/reseg_v2_full \
  --data_folder_s features_biomedclip_5x_eq4096_to256 \
  --data_folder_l features_biomedclip_20x_eq1024_to256 \
  --results_dir trained_models \
  --models_exp_code adenocarcinoma_biomedclip_dual_strict5_s1 \
  --save_exp_code adenocarcinoma_biomedclip_dual_strict5_s1_jiangxi_rerun_v2_fold0 \
  --model_type ViLa_MIL_BiomedCLIP \
  --task task_adenocarcinoma \
  --split all \
  --k 5 \
  --fold 0 \
  --csv_path dataset_csv/all_data_jiangxi.csv \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --prototype_number 16
```

---

## 13. Step 10：可选，生成江西 Overlay 热力图与 top-k patch

建议只在以下条件满足后再执行：

- 江西外部评估已经跑通
- 你已经确认当前 rerun 结果值得可视化

### 脚本

- `jiangxi_generate_attention_heatmaps_and_topk.py`

### 输入

- 江西 CSV：
  - `dataset_csv/all_data_jiangxi.csv`
- 江西 rerun 特征根目录：
  - `data/jiangxi/reseg_v2_full`
- 低倍特征目录：
  - `features_biomedclip_5x_eq4096_to256`
- 高倍特征目录：
  - `features_biomedclip_20x_eq1024_to256`
- WSI 根目录：
  - `data/jiangxi/wsi`
- 单折 checkpoint：
  - 例如 `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1/s_2_checkpoint.pt`
- 文本提示：
  - `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- 低倍 patch 根目录：
  - `data/jiangxi/reseg_v2_full/patches_5x_eq4096_to256`
- 高倍 patch 根目录：
  - `data/jiangxi/reseg_v2_full/patches_20x_eq1024_to256`

### 输出

- 注意力 H5：
  - `eval_results/jiangxi/heatmap_top_patch_fold2_paper_rerun_v2/attention_h5/*.h5`
- 热图：
  - `eval_results/jiangxi/heatmap_top_patch_fold2_paper_rerun_v2/overlay_heatmaps/5x/*.png`
  - `eval_results/jiangxi/heatmap_top_patch_fold2_paper_rerun_v2/overlay_heatmaps/20x/*.png`
- 预测汇总：
  - `eval_results/jiangxi/heatmap_top_patch_fold2_paper_rerun_v2/predictions.csv`
- top-k patch：
  - `eval_results/jiangxi/heatmap_top_patch_fold2_paper_rerun_v2/top_patches/...`
- top-k 清单：
  - `eval_results/jiangxi/heatmap_top_patch_fold2_paper_rerun_v2/top_patches/top_patches_manifest.csv`

### 命令

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python jiangxi_generate_attention_heatmaps_and_topk.py \
  --csv_path dataset_csv/all_data_jiangxi.csv \
  --data_root_dir data/jiangxi/reseg_v2_full \
  --data_folder_s features_biomedclip_5x_eq4096_to256 \
  --data_folder_l features_biomedclip_20x_eq1024_to256 \
  --wsi_root data/jiangxi/wsi \
  --checkpoint trained_models/adenocarcinoma_biomedclip_dual_strict5_s1/s_2_checkpoint.pt \
  --results_dir eval_results/jiangxi/heatmap_top_patch_fold2_paper_rerun_v2 \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --prototype_number 16 \
  --task task_adenocarcinoma \
  --heatmap_cmap soft_paper \
  --heatmap_alpha 0.72 \
  --wsi_suffix .tif \
  --patch_size_5x_level0 4096 \
  --patch_size_20x_level0 1024 \
  --show_colorbar \
  --export_top_k 10 \
  --patches_5x_dir data/jiangxi/reseg_v2_full/patches_5x_eq4096_to256 \
  --patches_20x_dir data/jiangxi/reseg_v2_full/patches_20x_eq1024_to256
```

说明：

- 这里默认选 `fold 2` 只是示例。
- 如果你更想看某个表现更好的 fold，可以把 checkpoint 改成对应的：
  - `s_0_checkpoint.pt`
  - `s_1_checkpoint.pt`
  - `s_2_checkpoint.pt`
  - `s_3_checkpoint.pt`
  - `s_4_checkpoint.pt`
- 这个江西专用脚本已经内置 top-k patch 导出逻辑，不需要再单独运行额外脚本。
- 相比旧版 `biomedclip_generate_attention_heatmaps.py`，它已专门兼容江西单层 TIFF：
  - 不再错误使用 `level=2/1`
  - 改为使用 `level=0`
  - 并显式指定：
    - 5x 视野大小 `4096`
    - 20x 视野大小 `1024`
- 它也已兼容江西 patch 文件名：
  - `<slide_id>_<x>_<y>.png`
  - `<slide_id>_256_<x>_<y>.png`

---

## 14. 推荐执行顺序

建议严格按下面顺序执行：

1. Step 0：建目录
2. Step 1：高倍 coords
3. Step 2：低倍 coords
4. Step 3：高倍 patch
5. Step 4：低倍 patch
6. Step 5：高倍特征
7. Step 6：低倍特征
8. Step 7：检查特征齐全
9. Step 8：5 折全量评估

如果担心链路中间出错，建议额外插入：

1. Step 9：先跑单折 `fold 0`
2. 确认无误后，再跑 Step 8 全量评估

如果后续还要做可视化，再继续：

3. Step 10：热图与 top-k patch

---

## 15. 每一步产物速查

### 15.1 坐标生成阶段

- 高倍 coords：
  - `data/jiangxi/reseg_v2_full/patches_coords_20x_eq1024/patches_1024/*.h5`
- 低倍 coords：
  - `data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096/patches_4096/*.h5`

### 15.2 patch 裁剪阶段

- 高倍 PNG：
  - `data/jiangxi/reseg_v2_full/patches_20x_eq1024_to256/<slide_id>/*.png`
- 低倍 PNG：
  - `data/jiangxi/reseg_v2_full/patches_5x_eq4096_to256/<slide_id>/*.png`

### 15.3 特征提取阶段

- 高倍特征：
  - `data/jiangxi/reseg_v2_full/features_biomedclip_20x_eq1024_to256/*.h5`
- 低倍特征：
  - `data/jiangxi/reseg_v2_full/features_biomedclip_5x_eq4096_to256/*.h5`

### 15.4 评估阶段

- 结果目录：
  - `eval_results/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_jiangxi_rerun_v2`

### 15.5 热图阶段

- 结果目录：
  - `eval_results/jiangxi/heatmap_top_patch_fold2_paper_rerun_v2`

### 15.6 top-k patch 阶段

- 结果目录：
  - `eval_results/jiangxi/heatmap_top_patch_fold2_paper_rerun_v2/top_patches`

---

## 16. 当前方案的边界说明

这份手册对应的是当前仓库里的现实约束：

- 江西 WSI 仍是单层 TIFF
- 没有把江西重新转换成带 4x/16x 金字塔的代理 SVS

所以当前方案不是“完全复刻训练集的 level1/level2 机制”，而是：

- 在 `level0` 上用更大的读取窗口
- 尽量逼近训练时两条分支的物理视野

当前推荐继续先按这份手册执行。

如果后续你要进一步追求与训练集定义完全一致，再考虑单独做：

- 江西 TIFF -> 多层 pyramidal WSI 的转换流程

但这不属于当前这份全链路执行手册的范围。
