# 会话压缩总结（ViLa-MIL 江西外部验证）

## 1. 目标与整体结论
- 目标：基于已训练模型 `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`，完成江西外部队列（`all_data_jiangxi.csv`）双尺度外部验证。
- 已确认：江西数据可用（415/415 文件匹配），OpenSlide 在 `vila_mil` 环境下可正常读取 `.tif`。
- 外部验证已跑通，但效果较差：`AUC≈0.805`，`ACC≈0.269`，`F1≈0.217`，主要表现为多折过度预测为同一类（阈值/分布偏移问题）。

---

## 2. 关键事实（按时间线）

### 2.1 训练结果分析
- 目录：`trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`
- 5 折内部结果：
  - mean AUC: `0.9672`
  - mean ACC: `0.9225`
  - mean F1: `0.9160`
- 严格 CV 检查：test fold 互斥，`TEST_SET_UNION_SIZE=968`，重叠为 0。

### 2.2 江西数据可用性检查
- CSV：`dataset_csv/all_data_jiangxi.csv`（415 条，Adenocarcinoma 399，NonAdenocarcinoma 16）
- WSI 根目录：`/private/ljh-data/shared/data/VILMIL_DATA/data/jiangxi_external/wsi`
- 文件类型：全部 `.tif`（BigTIFF）
- 在 `vila_mil` 环境实测：
  - OpenSlide 导入成功（1.4.2）
  - `OPENSLIDE_OK=415, OPENSLIDE_BAD=0`
  - 全部为 `generic-tiff`，`level_count=1`（单层）

### 2.3 双尺度策略讨论结论
- 由于单层 TIFF，不能用真实多 level 的 5x/20x。
- 原先 `1024→256` 与 `256→256` 更接近“相对双尺度”，不是真实 5x/20x。
- 后续改为更接近目标尺度：
  - 低尺度：`2048 -> 256`
  - 高尺度：`512 -> 256`

### 2.4 为什么 `graph_1024` 没内容
- `--stitch` 在单层超大 TIFF 上触发 `DecompressionBombError`（已复现）。
- 因此图目录空是正常现象，不影响坐标与后续 patch/特征流程。

### 2.5 外部验证差结果排查结论
- 结果目录：`eval_results/EVAL_jiangxi_external_biomedclip_2048_512_to256`
- 关键现象：
  - AUC 仍有一定区分度（~0.805），但 ACC/F1 低。
  - 多个 fold 预测极度偏向 class=1（导致大量假阳性）。
- 推断原因：
  1. 强域偏移（中心/扫描/染色差异）
  2. 外部尺度构造与训练分布不一致导致 logit 偏移
  3. 单层 TIFF 的“伪双尺度”跨域鲁棒性较弱
  4. 未做外部阈值校准

---

## 3. 本次已修改的代码文件

### 3.1 新增脚本
1. `tools/inspect_jiangxi_coords.py`
   - 功能：同时检查 `coords_512` 与 `coords_2048` 的 h5 坐标；
   - 输出：每个 slide 的 `shape / n_coords / patch_size / patch_level / sample coords`；
   - 支持参数：
     - `--coords-512-root`
     - `--coords-2048-root`
     - `--head`
     - `--limit`
     - `--save-csv`
     - `--slide-id`（后续新增，可指定单文件）

### 3.2 适配 512/2048 的参数修改
2. `create_patches_fp_jiangxi.py`
   - `--coord-size` / `--patch-size` 的 choices 从 `[256,1024]` 扩展到 `[256,512,1024,2048]`。

3. `patch_generation_5x_jiangxi.py`
   - `--coords-size` 扩展到 `[256,512,1024,2048]`
   - 新增 `--out-size`（默认 256）
   - 低尺度由“固定输出 256”改为可配置输出尺寸（默认保持 256）。

4. `patch_generation_20x_jiangxi.py`
   - `--coords-size` 扩展到 `[256,512,1024,2048]`
   - 新增 `--out-size`（默认 `coords-size`）
   - 支持高尺度 `512 -> 256`（读取 512 后 resize 到 256）。

---

## 4. 已给出的核心命令流（最终版本）

### 4.1 坐标生成（已执行）
- `coord-size=2048` 与 `coord-size=512` 两套。

### 4.2 patch 裁剪（最终推荐）
- 低尺度：`2048 -> 256`（`patch_generation_5x_jiangxi.py`）
- 高尺度：`512 -> 256`（`patch_generation_20x_jiangxi.py --out-size 256`）

### 4.3 特征提取（已执行）
- 低尺度：`features_biomedclip_5x`
- 高尺度：`features_biomedclip_20x`

### 4.4 外部评估（已执行）
- `eval.py` + `models_exp_code=adenocarcinoma_biomedclip_dual_strict5_s1`
- 输出：`EVAL_jiangxi_external_biomedclip_2048_512_to256`

---

## 5. 你当前可直接继续做的事
1. 基于外部集做阈值校准（而非直接用 0.5/argmax）。
2. 分 fold 分析最稳定折（当前 fold2 相对最好）。
3. 若时间允许，做 stain normalization 或轻量外部适配（可显著改善 ACC/F1）。

---

## 6. 备注
- `ls -l | wc -l` 会多算一行 `total`，比真实文件数多 1（已解释）。
- 当前两套坐标均已完成（415/415），两套特征也已完成（415/415）。
