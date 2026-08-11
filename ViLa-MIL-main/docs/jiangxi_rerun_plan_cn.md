# 江西外部队列重跑方案（单层 TIFF 对齐训练尺度）

本文档对应当前仓库里的真实数据状态：

- 训练集 `yiyuan` 是多层 `svs`
- 江西外部集 `jiangxi` 是单层 `tif/OME-TIFF`
- 因此江西不能直接照搬 `level=2 / level=1` 的训练裁图定义

## 1. 这次为什么要重跑

抽样核对后，训练集与旧江西流程的真实定义如下：

- `yiyuan` 低倍支路：
  - `patch_level=2`
  - `patch_size=256`
  - 对应 `level0` 视野大约 `256 x 16 = 4096`
- `yiyuan` 高倍支路：
  - `patch_level=1`
  - `patch_size=256`
  - 对应 `level0` 视野大约 `256 x 4 = 1024`

而旧江西流程实际是：

- 低倍支路：
  - `patch_level=0`
  - `patch_size=2048`
  - 再缩成 `256`
- 高倍支路：
  - `patch_level=0`
  - `patch_size=512`
  - 再缩成 `256`

这会让江西看到的物理视野比训练时缩小约一半，属于主要分布偏移来源。

## 2. 当前更合理的对齐方案

因为江西 WSI 当前只有单层：

- `level_count = 1`
- `patch_level` 只能稳定使用 `0`

所以推荐用“level0 等效视野对齐”的方式近似训练尺度：

- 江西低倍支路：
  - `patch_size=4096`
  - `step_size=4096`
  - `patch_level=0`
  - 裁 patch 时缩成 `256`
- 江西高倍支路：
  - `patch_size=1024`
  - `step_size=1024`
  - `patch_level=0`
  - 裁 patch 时缩成 `256`

这个方案的目标不是复制 SVS 的金字塔结构，而是尽量复制训练时两条支路在 `level0` 上看到的视野范围。

## 3. preset 应该怎么处理

`presets/tcga.csv` 和 `presets/heyuan.csv` 只能作为起点，不应该直接当成江西最终参数。

当前仓库已新增：

- `presets/jiangxi.csv`

它的作用是：

- 先把江西与别的外部队列解耦
- 方便后续单独调参
- 对单层 TIFF 默认走 `seg_level=-1 / vis_level=-1`，让代码自动落到单层 fallback

当前初版参数偏保守，接近 `heyuan.csv`，适合先跑 pilot。

## 4. 建议目录

建议把这次重跑结果单独放到新目录，避免污染旧江西产物：

- `data/jiangxi/reseg_v2/patches_coords_5x_eq4096`
- `data/jiangxi/reseg_v2/patches_coords_20x_eq1024`
- `data/jiangxi/reseg_v2/patches_5x_eq4096_to256`
- `data/jiangxi/reseg_v2/patches_20x_eq1024_to256`
- `data/jiangxi/reseg_v2/features_biomedclip_5x_eq4096_to256`
- `data/jiangxi/reseg_v2/features_biomedclip_20x_eq1024_to256`

pilot 可以单独放：

- `data/jiangxi/reseg_v2_pilot/`

## 5. Pilot 命令

以下命令默认在仓库根目录执行：

```bash
cd /private/ljh-data/shared/ViLMIL/ViLa-MIL-main
```

### 5.1 低倍等效视野坐标

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python create_patches_fp_jiangxi.py \
  --source data/jiangxi/wsi \
  --csv dataset_csv/all_data_jiangxi.csv \
  --save-dir data/jiangxi/reseg_v2_pilot/patches_coords_5x_eq4096 \
  --preset jiangxi.csv \
  --patch-size 4096 \
  --step-size 4096 \
  --patch-level 0 \
  --seg \
  --patch \
  --limit 5
```

### 5.2 高倍等效视野坐标

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python create_patches_fp_jiangxi.py \
  --source data/jiangxi/wsi \
  --csv dataset_csv/all_data_jiangxi.csv \
  --save-dir data/jiangxi/reseg_v2_pilot/patches_coords_20x_eq1024 \
  --preset jiangxi.csv \
  --patch-size 1024 \
  --step-size 1024 \
  --patch-level 0 \
  --seg \
  --patch \
  --limit 5
```

### 5.3 裁低倍 patch

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python patch_generation_5x_jiangxi.py \
  --source data/jiangxi/wsi \
  --csv dataset_csv/all_data_jiangxi.csv \
  --coords-root data/jiangxi/reseg_v2_pilot/patches_coords_5x_eq4096 \
  --coords-size 4096 \
  --patch-level 0 \
  --out-size 256 \
  --output-root data/jiangxi/reseg_v2_pilot/patches_5x_eq4096_to256 \
  --workers 4 \
  --limit 5
```

### 5.4 裁高倍 patch

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python patch_generation_20x_jiangxi.py \
  --source data/jiangxi/wsi \
  --csv dataset_csv/all_data_jiangxi.csv \
  --coords-root data/jiangxi/reseg_v2_pilot/patches_coords_20x_eq1024 \
  --coords-size 1024 \
  --patch-level 0 \
  --out-size 256 \
  --output-root data/jiangxi/reseg_v2_pilot/patches_20x_eq1024_to256 \
  --workers 4 \
  --limit 5
```

### 5.5 提取 BiomedCLIP 特征

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/jiangxi/reseg_v2_pilot/patches_5x_eq4096_to256 \
  --library_path data/jiangxi/reseg_v2_pilot/features_biomedclip_5x_eq4096_to256 \
  --batch_size 32 \
  --dataset adenocarcinoma
```

```bash
/home/ljh/anaconda3/envs/vila_mil/bin/python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/jiangxi/reseg_v2_pilot/patches_20x_eq1024_to256 \
  --library_path data/jiangxi/reseg_v2_pilot/features_biomedclip_20x_eq1024_to256 \
  --batch_size 32 \
  --dataset adenocarcinoma
```

### 5.6 先做 pilot 评估

pilot 评估建议只用 pilot CSV 和 `fold 0`，先验证链路和方向是否正常。

## 6. 全量重跑建议

只有在 pilot 满足以下条件后，再跑全量：

- mask 看起来合理，没有明显漏组织或大片背景
- 低倍和高倍的 patch 数量处于可接受范围
- 特征提取能稳定完成
- `fold 0` pilot 评估没有明显异常

如果 pilot 表现仍差，再继续微调：

- `presets/jiangxi.csv`
- 低倍步长是否放宽到 `2048`
- 高倍步长是否放宽到 `512`
- 是否需要先把江西 TIFF 做成带金字塔的代理 WSI
