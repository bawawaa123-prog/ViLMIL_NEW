# 江西外部队列重跑任务交接提示词

请先基于以下上下文继续工作，不要重复从零分析。

---

## 1. 项目与主线

- 仓库根目录：
  - `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main`
- 当前主线模型：
  - `ViLa_MIL_BiomedCLIP`
- 当前任务不是训练，而是：
  - 重新处理并评估江西外部队列 `jiangxi`

训练集与外部集位置：

- 本地私有训练集：
  - `data/yiyuan`
- 河源外部集：
  - `data/heyuan`
- 江西外部集：
  - `data/jiangxi`

训练权重目录：

- `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`

评估 CSV：

- 训练/内部：
  - `dataset_csv/all_data.csv`
- 江西：
  - `dataset_csv/all_data_jiangxi.csv`
- 河源：
  - `dataset_csv/all_data_heyuan.csv`

---

## 2. 已确认的关键事实

### 2.1 `drop_out` 不是当前江西问题主因

- `eval.py` 和训练入口里虽然保留了 `--drop_out`
- 但对 `ViLa_MIL_BiomedCLIP` 主线，这个参数没有真正传入模型结构
- 所以江西差结果的核心原因不是 `drop_out`

### 2.2 训练集与江西 WSI 结构不同

训练集 `yiyuan`：

- `svs`
- 多层金字塔
- 抽样确认：
  - 高倍支路真实定义：`patch_level=1, patch_size=256`
  - 低倍支路真实定义：`patch_level=2, patch_size=256`

江西外部集 `jiangxi`：

- `tif / OME-TIFF`
- 当前是单层 WSI
- 抽样确认：
  - `level_count=1`
  - `patch_level` 只能稳定使用 `0`

### 2.3 旧江西流程的核心问题

旧江西 coords 定义是：

- 高倍旧方案：
  - `patch_level=0`
  - `patch_size=512`
  - 再缩成 `256`
- 低倍旧方案：
  - `patch_level=0`
  - `patch_size=2048`
  - 再缩成 `256`

这与训练集真实物理视野不对齐。

### 2.4 当前确定采用的新江西尺度定义

为了逼近训练时两条分支在 `level0` 上的视野，当前正式方案为：

- 高倍支路：
  - `patch_size=1024`
  - `step_size=1024`
  - `patch_level=0`
  - 裁 patch 时缩成 `256`
- 低倍支路：
  - `patch_size=4096`
  - `step_size=4096`
  - `patch_level=0`
  - 裁 patch 时缩成 `256`

这套方案已经被江西 pilot 实跑验证可行。

---

## 3. 当前正式江西 preset

`presets/jiangxi.csv` 不是最初那个和 `heyuan.csv` 一样的占位版了。

已经通过 5 张江西 pilot A/B，选定为：

```csv
seg_level,sthresh,mthresh,close,use_otsu,a_t,a_h,max_n_holes,vis_level,line_thickness,white_thresh,black_thresh,use_padding,contour_fn,keep_ids,exclude_ids
-1,6,7,6,TRUE,8,2,12,-1,100,5,50,TRUE,four_pt,none,none
```

候选对比结论：

- `v1`：
  - 实际上接近原 `heyuan` 占位参数
- `v2`：
  - 被选为最终版
- `v3`：
  - 偏保守，掩膜收缩更明显，且处理稳定性不如 `v2`

当前正式使用：

- `presets/jiangxi.csv`

---

## 4. 已修改/新增的重要文件

### 已修改

- `create_patches_fp_jiangxi.py`
  - 默认路径改为 `data/jiangxi/wsi`
  - 默认 preset 改为 `jiangxi.csv`
  - 支持 `4096`
  - 新增 `--limit`

- `patch_generation_20x_jiangxi.py`
  - 默认路径改为 `data/jiangxi/wsi`
  - 用于 `1024 -> 256`

- `patch_generation_5x_jiangxi.py`
  - 默认路径改为 `data/jiangxi/wsi`
  - 用于 `4096 -> 256`

### 已新增文档

- `docs/jiangxi_rerun_plan_cn.md`
  - 江西重跑思路说明

- `docs/jiangxi_full_pipeline_execution_cn.md`
  - 江西从数据处理到评估的完整执行手册

### 已新增 preset

- `presets/jiangxi.csv`
- `presets/jiangxi_v1.csv`
- `presets/jiangxi_v2.csv`
- `presets/jiangxi_v3.csv`

---

## 5. 已完成的实跑验证

### 5.1 单张江西验证

针对 `20251222-06`：

- 新 `jiangxi.csv` + `1024` coords 生成成功
- mask 与 coords H5 已生成
- 高倍 patch 裁剪已实际生成大量 PNG

### 5.2 五张江西 pilot

已完成 5 张样本：

- `20251222-06`
- `20251222-08`
- `20251222-10`
- `20251222-19`
- `20251222-20`

在新尺度下已成功生成：

- 高倍：
  - `reseg_v2_pilot/patches_coords_20x_eq1024/patches_1024/*.h5`
- 低倍：
  - `reseg_v2_pilot/patches_coords_5x_eq4096/patches_4096/*.h5`

并已基于这 5 张完成 `jiangxi_v1/v2/v3` 的 preset A/B。

### 5.3 新旧 coords 数量对比

旧高倍 `512`：

- 平均 coords 数约 `9677.8`

新高倍 `1024`：

- 平均 coords 数约 `2335.2`

旧低倍 `2048`：

- 平均 coords 数约 `633.8`

新低倍 `4096`：

- 平均 coords 数约 `162.4`

这说明新尺度映射与几何预期一致。

---

## 6. 关于 graph 的结论

- 江西流程里的 `graph_*` 默认不会生成，除非运行 `create_patches_fp_jiangxi.py` 时显式加 `--stitch`
- 我们当前江西重跑命令基本没加 `--stitch`
- 所以 `graph_*` 没有内容，首先是因为没有真正生成
- 即便生成，当前代码画的也是 patch 拼接预览，不是高亮 coverage 图，视觉上容易显得空白
- 当前任务已决定先不处理这一部分

---

## 7. 当前可直接使用的执行手册

后续如果要真正执行江西全量链路，请直接参考：

- `docs/jiangxi_full_pipeline_execution_cn.md`

那份文档已经给出：

- 环境
- 每一步命令
- 每一步输入输出
- 最终评估命令

按那份文档执行即可。

---

## 8. 当前建议的新会话起点

新会话建议直接从以下动作开始，而不是重新分析历史问题：

1. 打开并遵循：
   - `docs/jiangxi_full_pipeline_execution_cn.md`
2. 默认使用：
   - `presets/jiangxi.csv`
3. 默认采用新尺度：
   - 高倍 `1024 -> 256`
   - 低倍 `4096 -> 256`
4. 若要正式推进：
   - 从全量 coords 生成开始
   - 再裁 patch
   - 再提 BiomedCLIP 特征
   - 最后跑江西外部评估

---

## 9. 如需一句话提示词

可直接把下面这段作为新会话开头：

> 请继续处理 `ViLa-MIL-main` 中的江西外部队列重跑任务。不要从零开始分析。当前主线是 `ViLa_MIL_BiomedCLIP`；训练集 `data/yiyuan` 为多层 `svs`，江西外部集 `data/jiangxi` 为单层 `tif/OME-TIFF`。江西旧流程的核心问题是尺度错配，现已确认采用新的等效尺度：高倍 `patch_size=1024, step_size=1024, patch_level=0, resize到256`，低倍 `patch_size=4096, step_size=4096, patch_level=0, resize到256`。当前正式 preset 为 `presets/jiangxi.csv`，内容是 `-1,6,7,6,TRUE,8,2,12,-1,100,5,50,TRUE,four_pt,none,none`。相关执行手册已写在 `docs/jiangxi_full_pipeline_execution_cn.md`，请直接基于该手册继续推进江西全量流程或后续评估，不要再回退到 `tcga.csv` 或旧的 `512/2048 -> 256` 江西方案。

