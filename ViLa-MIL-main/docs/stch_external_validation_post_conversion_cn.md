# 汕头 `.image` 转换后外部验证执行清单

本文档用于汕头外部验证的“转换后专用”落地流程。

它解决的问题是：

1. 汕头原始数据当前为 `data/stch/wsi/*.image`
2. 现有仓库的 WSI 预处理链路依赖 OpenSlide，可直接处理的格式是 `.svs/.tif/.tiff/.ome.tif/.ome.tiff`
3. 因此，必须先把 `.image` 转成 OpenSlide 可读的 WSI，再进入坐标生成、patch 裁剪、BiomedCLIP 特征提取和外部评估

这份清单默认你已经完成了“.image -> 可被 OpenSlide 打开的 WSI”转换工作；如果还没完成转换，请先停在本文档的“转换验收标准”部分做检查。

---

## 1. 当前已确认的汕头输入信息

截至 2026-06-09，本仓库内已确认：

- 汕头 CSV：
  - `dataset_csv/all_data_shantou.csv`
- 汕头原始目录：
  - `data/stch/wsi`
- 原始文件格式：
  - `.image`
- 文本提示：
  - `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- 五折权重：
  - `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`

额外核对结果：

- `all_data_shantou.csv` 当前共有 `198` 条记录
- `slide_id` 与原始 `*.image` 文件名去后缀后当前是可对齐的
- 原始目录下存在额外文件，不影响外部验证
- 标签分布极不平衡：
  - `Adenocarcinoma = 196`
  - `NonAdenocarcinoma = 2`

说明：

- 这个类别分布不影响“能不能做外部验证”
- 但会显著影响外部验证指标的稳定性，尤其是负类召回、F1、AUC 的解释需要谨慎

---

## 2. 本文档适用前提

只有同时满足下面 4 条，才建议继续往下执行：

1. 你已经把 `.image` 转成了 OpenSlide 可读取的 WSI 文件
2. 转换后文件的文件名主干与 CSV 中 `slide_id` 一致
3. 转换后的文件建议单独放到新目录，而不是覆盖原始 `.image`
4. 你准备复用现有五折权重做“纯外部验证”，而不是在汕头队列上重新训练

建议使用的新目录：

- `data/stch/wsi_converted`

不建议：

- 直接覆盖 `data/stch/wsi`
- 一边保留 `.image` 一边在同目录混放大量不同格式文件

---

## 3. 转换验收标准

在正式跑外部验证前，先确认转换结果满足下面标准。

### 3.1 文件命名标准

转换后的文件名主干必须与 `slide_id` 完全一致，例如：

- `1.svs`
- `2.tif`
- `3.ome.tiff`

如果 CSV 中是：

- `slide_id = 1`

那么不应该变成：

- `STCH_1.svs`
- `1_converted.tif`
- `001.svs`

除非你同步修改了 CSV。

### 3.2 可读取标准

转换后的文件必须能被 OpenSlide 打开。

推荐检查 1 个样本：

```bash
cd /private/ljh-data/shared/ViLMIL/ViLa-MIL-main

conda run -n vila_mil python -c "import openslide; s=openslide.open_slide('data/stch/wsi_converted/1.svs'); print('level_count=', s.level_count); print('dimensions=', s.dimensions); print('objective=', s.properties.get('aperio.AppMag') or s.properties.get('openslide.objective-power'))"
```

如果你的转换输出不是 `.svs`，把命令里的文件名改成实际存在的扩展名即可。

### 3.3 推荐格式标准

优先级建议如下：

1. 金字塔 `SVS`
2. 金字塔 `TIFF / OME-TIFF`
3. 单层大 `TIFF`

说明：

- 单层大 TIFF 在当前仓库里不是绝对不能跑
- 但坐标生成和裁 patch 通常会更慢
- 如果可以控制转换参数，优先导出为带金字塔层级的 WSI

### 3.4 数量对齐标准

转换后目录中，至少应该覆盖 CSV 里的全部 `198` 个 `slide_id`。

先做目录比对：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_shantou.csv \
  --dirs_root data/stch/wsi_converted \
  --output_path eval_results/stch/stch_csv_vs_converted_wsi.csv
```

理想结果：

- `missing_in_dirs = 0`

`extra_in_dirs > 0` 可以接受，只代表目录里有额外文件。

---

## 4. 这次流程的总体策略

汕头转换后，直接沿用当前仓库的双尺度外部验证流程：

- `5x` 支路：
  - `patch_level=2`
  - patch size = `256`
- `20x` 支路：
  - `patch_level=1`
  - patch size = `256`

与广东医流程一致：

1. 先生成组织分割和 patch 坐标
2. 再裁出 PNG patch
3. 再提取 BiomedCLIP 特征
4. 最后复用现有五折权重外部评估

这次流程不需要：

- 在汕头队列上重新生成五折训练划分
- 在 `eval.py` 中使用 `test` split 限制样本

外部验证阶段应使用：

- `eval.py --split all`

---

## 5. 本次建议使用的核心目录

仓库根目录：

- `/private/ljh-data/shared/ViLMIL/ViLa-MIL-main`

关键输入：

- 转换后 WSI 根目录：
  - `data/stch/wsi_converted`
- 汕头 CSV：
  - `dataset_csv/all_data_shantou.csv`
- 预设：
  - `presets/gdmuah.csv`
- 文本提示：
  - `text_prompt/adenocarcinoma_dual_scale_prompt.csv`
- 五折权重：
  - `trained_models/adenocarcinoma_biomedclip_dual_strict5_s1`

建议输出目录：

- `data/stch/patches_coords_5x`
- `data/stch/patches_coords_20x`
- `data/stch/patches_5x`
- `data/stch/patches_20x`
- `data/stch/features_biomedclip_5x`
- `data/stch/features_biomedclip_20x`
- `eval_results/stch`

说明：

- 当前仓库里还没有汕头专用 preset
- 第一轮建议先复用 `gdmuah.csv`
- 如果分割质量明显不稳，再单独为汕头补一个 `presets/stch.csv`

---

## 6. 运行前约定

以下命令默认在仓库根目录执行：

```bash
cd /private/ljh-data/shared/ViLMIL/ViLa-MIL-main
```

---

## 7. Step 0：创建输出目录

```bash
mkdir -p data/stch/wsi_converted
mkdir -p data/stch/patches_coords_5x
mkdir -p data/stch/patches_coords_20x
mkdir -p data/stch/patches_5x
mkdir -p data/stch/patches_20x
mkdir -p data/stch/features_biomedclip_5x
mkdir -p data/stch/features_biomedclip_20x
mkdir -p eval_results/stch
```

---

## 8. Step 0.5：先做 10 例烟雾测试子集

不建议一上来就全量跑 198 例。

先生成一个前 10 例的烟雾测试 CSV：

```bash
conda run -n vila_mil python -c "import pandas as pd; df=pd.read_csv('dataset_csv/all_data_shantou.csv'); df.head(10).to_csv('dataset_csv/all_data_shantou_smoke10.csv', index=False)"
```

后面 Step 1 到 Step 9 可以先把 `all_data_shantou.csv` 暂时替换成：

- `dataset_csv/all_data_shantou_smoke10.csv`

等 10 例全流程打通，再换回全量 CSV。

---

## 9. Step 1：检查 CSV 与转换后 WSI 的一致性

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_shantou.csv \
  --dirs_root data/stch/wsi_converted \
  --output_path eval_results/stch/stch_csv_vs_wsi.csv
```

这一步重点确认：

1. `missing_in_dirs = 0`
2. 没有因为重命名导致的 `slide_id` 对不上
3. 转换目录中的文件扩展名不影响 stem 匹配

如果 `missing_in_dirs > 0`，不要继续往下跑。

---

## 10. Step 1.5：抽查 3 个转换后 WSI 是否真的能被 OpenSlide 正常打开

建议至少手动验证 3 个样本：

```bash
conda run -n vila_mil python -c "import openslide; import os; samples=['1.svs','2.svs','3.svs']; root='data/stch/wsi_converted'; \
for fn in samples: \
    path=os.path.join(root, fn); \
    s=openslide.open_slide(path); \
    print(fn, 'level_count=', s.level_count, 'dimensions=', s.dimensions, 'objective=', s.properties.get('aperio.AppMag') or s.properties.get('openslide.objective-power'))"
```

如果你的扩展名不是 `.svs`，请替换为真实文件名。

验收建议：

- 能正常打开
- `level_count >= 2` 更理想
- `dimensions` 合理，不是异常的小图

---

## 11. Step 2：生成 5x 坐标和分割结果

这里直接复用 `create_patches_fp_heyuan.py`。

```bash
python create_patches_fp_heyuan.py \
  --source data/stch/wsi_converted \
  --csv dataset_csv/all_data_shantou.csv \
  --save-dir data/stch/patches_coords_5x \
  --preset gdmuah.csv \
  --patch-size 256 \
  --step-size 256 \
  --patch-level 2 \
  --seg \
  --patch \
  --stitch
```

输出重点：

- `data/stch/patches_coords_5x/patches_256/*.h5`
- `data/stch/patches_coords_5x/masks/*.jpg`
- `data/stch/patches_coords_5x/graph_256/*.jpg`

跑完后建议抽查：

1. `masks/*.jpg`
2. `graph_256/*.jpg`

重点看：

1. 组织区域是否被合理圈出
2. 背景是否被大面积误保留
3. patch 是否基本覆盖组织区域

---

## 12. Step 3：生成 20x 坐标和分割结果

```bash
python create_patches_fp_heyuan.py \
  --source data/stch/wsi_converted \
  --csv dataset_csv/all_data_shantou.csv \
  --save-dir data/stch/patches_coords_20x \
  --preset gdmuah.csv \
  --patch-size 256 \
  --step-size 256 \
  --patch-level 1 \
  --seg \
  --patch \
  --stitch
```

输出重点：

- `data/stch/patches_coords_20x/patches_256/*.h5`
- `data/stch/patches_coords_20x/masks/*.jpg`
- `data/stch/patches_coords_20x/graph_256/*.jpg`

---

## 13. Step 4：检查 5x / 20x 坐标是否覆盖全部样本

5x 检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_shantou.csv \
  --dirs_root data/stch/patches_coords_5x/patches_256 \
  --output_path eval_results/stch/stch_csv_vs_coords5x.csv
```

20x 检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_shantou.csv \
  --dirs_root data/stch/patches_coords_20x/patches_256 \
  --output_path eval_results/stch/stch_csv_vs_coords20x.csv
```

如果有缺失，优先排查：

1. 某些转换后 WSI 虽然存在，但 OpenSlide 读取异常
2. 该 slide 分割失败，没有有效组织轮廓
3. 文件名主干虽然看起来接近，但和 `slide_id` 仍存在细微不一致

---

## 14. Step 5：裁剪 5x PNG patch

```bash
python patch_generation_5x.py \
  --source data/stch/wsi_converted \
  --csv dataset_csv/all_data_shantou.csv \
  --coords-root data/stch/patches_coords_5x \
  --patch-size 256 \
  --patch-level 2 \
  --output-root data/stch/patches_5x \
  --workers 8 \
  --skip-existing
```

输出通常为：

- `data/stch/patches_5x/<slide_id>/*.png`

---

## 15. Step 6：裁剪 20x PNG patch

```bash
python patch_generation_20x.py \
  --source data/stch/wsi_converted \
  --csv dataset_csv/all_data_shantou.csv \
  --coords-root data/stch/patches_coords_20x \
  --patch-size 256 \
  --patch-level 1 \
  --output-root data/stch/patches_20x \
  --workers 8 \
  --skip-existing
```

输出通常为：

- `data/stch/patches_20x/<slide_id>/*.png`

---

## 16. Step 7：检查 patch 裁剪结果是否完整

5x patch 目录检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_shantou.csv \
  --dirs_root data/stch/patches_5x \
  --output_path eval_results/stch/stch_csv_vs_patches5x.csv
```

20x patch 目录检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_shantou.csv \
  --dirs_root data/stch/patches_20x \
  --output_path eval_results/stch/stch_csv_vs_patches20x.csv
```

同时建议随机抽查几个子目录，确认：

1. patch 数量不是 0
2. patch 不是几乎全白背景
3. 5x 和 20x 的视野差异符合预期

---

## 17. Step 8：提取 5x BiomedCLIP 特征

```bash
python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/stch/patches_5x \
  --library_path data/stch/features_biomedclip_5x \
  --batch_size 32 \
  --dataset adenocarcinoma
```

输出：

- `data/stch/features_biomedclip_5x/*.h5`

---

## 18. Step 9：提取 20x BiomedCLIP 特征

```bash
python feature_extraction/patch_extraction_biomedclip.py \
  --patches_path data/stch/patches_20x \
  --library_path data/stch/features_biomedclip_20x \
  --batch_size 32 \
  --dataset adenocarcinoma
```

输出：

- `data/stch/features_biomedclip_20x/*.h5`

---

## 19. Step 10：检查特征文件是否完整

5x 特征检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_shantou.csv \
  --dirs_root data/stch/features_biomedclip_5x \
  --output_path eval_results/stch/stch_csv_vs_feat5x.csv
```

20x 特征检查：

```bash
python tools/compare_csv_with_dirs_2_5x.py \
  --csv_path dataset_csv/all_data_shantou.csv \
  --dirs_root data/stch/features_biomedclip_20x \
  --output_path eval_results/stch/stch_csv_vs_feat20x.csv
```

只有当两路特征都完整后，才建议进入外部评估。

---

## 20. Step 11：复用现有五折权重做汕头外部评估

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
  --save_exp_code adenocarcinoma_biomedclip_dual_strict5_s1_stch_external \
  --model_type ViLa_MIL_BiomedCLIP \
  --mode transformer \
  --splits_dir trained_models/adenocarcinoma_biomedclip_dual_strict5_s1 \
  --data_root_dir data/stch \
  --data_folder_s features_biomedclip_5x \
  --data_folder_l features_biomedclip_20x \
  --csv_path dataset_csv/all_data_shantou.csv \
  --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv \
  --prototype_number 16
```

输出目录通常为：

- `eval_results/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_stch_external`

说明：

- 这里必须使用 `--split all`
- 不要用训练 fold 自带的 `test` split 去筛汕头外部样本

---

## 21. Step 12：建议保留的关键结果

至少建议保留下面这些目录和文件：

- `eval_results/stch/*.csv`
- `eval_results/stch/*.txt`
- `data/stch/patches_coords_5x/process_list_autogen.csv`
- `data/stch/patches_coords_20x/process_list_autogen.csv`
- `eval_results/EVAL_adenocarcinoma_biomedclip_dual_strict5_s1_stch_external`

这些文件能帮助你后续定位：

1. 哪些 slide 在坐标生成时失败
2. 哪些 slide 在 patch 裁剪后为空
3. 哪些 slide 在特征或评估阶段缺失
4. 每个 fold 的误判样本

---

## 22. 建议的执行顺序

建议按下面顺序推进，而不是一次性全量开跑：

1. 完成 `.image` 转换，并放入 `data/stch/wsi_converted`
2. 跑 Step 1 和 Step 1.5，先确认命名和 OpenSlide 读取都没问题
3. 用 `all_data_shantou_smoke10.csv` 做 10 例烟雾测试
4. 烟雾测试完整打通后，再替换成全量 `all_data_shantou.csv`
5. 全量跑完整条链路后，再执行 `eval.py`

---

## 23. 常见失败点与处理建议

### 23.1 `missing_in_dirs > 0`

常见原因：

1. 转换后文件名被改了
2. 转换输出目录与命令中的 `--source` 不一致
3. 文件扩展名没问题，但 stem 与 `slide_id` 不一致

### 23.2 OpenSlide 打不开转换后的文件

说明当前转换结果不合格。

优先排查：

1. 是否真的导出了标准 WSI，而不是普通静态图
2. 是否是厂商私有格式改了后缀
3. 是否导出过程损坏了多分辨率层级

### 23.3 有 WSI，但坐标 `h5` 没生成

常见原因：

1. 组织分割失败
2. WSI 虽能打开，但 level 或属性异常
3. 某些样本尺寸过大、读取过慢或异常中断

这时先去看：

- `patches_coords_5x/masks/`
- `patches_coords_20x/masks/`
- `process_list_autogen.csv`

### 23.4 patch 子目录存在，但里面几乎全是背景

说明当前 preset 对汕头转换后的图像不够合适。

这时建议：

1. 先抽查 10 例 mask 和 graph
2. 再考虑单独做 `presets/stch.csv`
3. 不建议在没抽查分割质量前直接全量提特征

### 23.5 特征缺失

常见原因：

1. 某个 slide 的 patch 目录为空
2. patch 文件损坏
3. BiomedCLIP 运行中断

优先回查：

- `data/stch/patches_5x`
- `data/stch/patches_20x`

---

## 24. 一句话版结论

汕头外部验证是可以落地的，但前提不是“直接处理 `.image`”，而是：

1. 先把 `.image` 稳定转换为 OpenSlide 可读取的 WSI
2. 再按本文档走标准双尺度外部验证链路

如果你后面确认了转换工具和实际输出扩展名，我建议下一步再补一份更具体的：

- “汕头 `.image` 转换验收脚本”
- 或者 “`presets/stch.csv` 的第一版调参清单”
