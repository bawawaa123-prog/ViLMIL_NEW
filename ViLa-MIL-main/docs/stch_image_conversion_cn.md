# 汕头 `.image` 文件处理与转换说明

这份说明只解决一件事：

- 如何把 `data/stch/wsi/*.image` 处理成你现有程序可以直接使用的标准 WSI

结论先说：

1. 这批 `.image` 不是普通图片格式
2. 它们当前不能被 OpenSlide 直接打开
3. 你的现有流程依赖 OpenSlide，所以不能直接吃 `.image`
4. 现阶段最短可落地路径不是“改代码强行读 `.image`”，而是“先把 `.image` 转成标准金字塔 WSI”

---

## 1. 我们已经确认了什么

在当前环境里已经确认：

1. `openslide.open_slide('.../1.image')` 会报 `Unsupported or missing image file`
2. `.image` 文件内部存在 `KFB` 标记，强烈提示它是 KFB 系列私有病理切片格式或兼容容器
3. OpenSlide 官方当前公开列出的支持格式里不包含 KFB
4. OpenSlide 的 issue 列表中存在 “Support KFB format” 请求，说明它不是现成支持格式

这意味着：

- 你现在的 Python / OpenSlide 代码不应该直接改去读 `.image`
- 即使强行重命名为 `.svs`，也不会让 OpenSlide magically 支持它

---

## 2. 你现有程序真正需要什么

你仓库里的现有流程需要的是下面这类文件之一：

1. `.svs`
2. `.tif`
3. `.tiff`
4. `.ome.tif`
5. `.ome.tiff`

并且这些文件必须满足：

1. 能被 OpenSlide 正常打开
2. 最好是金字塔、多分辨率 WSI
3. 文件名主干和 `all_data_shantou.csv` 里的 `slide_id` 一致

推荐目标：

- `data/stch/wsi_converted/1.svs`
- `data/stch/wsi_converted/2.svs`

或者：

- `data/stch/wsi_converted/1.ome.tiff`
- `data/stch/wsi_converted/2.ome.tiff`

---

## 3. 正确的处理路线

### 路线 A：最推荐

使用 KFB 专用软件或 SDK，把 `.image` 导出成：

1. 金字塔 `SVS`
2. 金字塔 `TIFF / OME-TIFF`

然后再交给当前仓库的 OpenSlide 流程。

### 路线 B：可接受

如果厂商工具不能直接导出 `SVS`，那就导出：

1. 金字塔 BigTIFF
2. 金字塔 OME-TIFF

只要 OpenSlide 能打开，也可以进入当前流程。

### 不推荐路线

1. 导出普通单张 `PNG / JPG`
2. 导出截图
3. 导出局部 patch，而不是整张 WSI
4. 直接覆盖原始 `.image`

这些都不能无缝接到你现在的外部验证流程里。

---

## 4. 我已经给你补好的两个工具

### 4.1 先把 `.image` 映射成 `.kfb`

有些厂商工具只认 `.kfb` 扩展名，不认 `.image`。

我补了：

- [prepare_stch_kfb_aliases.py](/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/tools/prepare_stch_kfb_aliases.py)

它的作用不是最终转换，而是先安全地产生一套 `.kfb` 别名文件，方便外部 KFB 工具识别。

默认做法是：

- 用硬链接生成 `data/stch/wsi_kfb_alias/*.kfb`

这样不会复制一遍 200 多个大文件。

使用方法：

```bash
cd /private/ljh-data/shared/ViLMIL

conda run -n vila_mil python ViLa-MIL-main/tools/prepare_stch_kfb_aliases.py
```

如果你想先做 10 例烟雾测试：

```bash
conda run -n vila_mil python ViLa-MIL-main/tools/prepare_stch_kfb_aliases.py --limit 10
```

输出目录默认是：

- `ViLa-MIL-main/data/stch/wsi_kfb_alias`

生成结果示例：

- `wsi_kfb_alias/1.kfb`
- `wsi_kfb_alias/2.kfb`

### 4.2 转换后验收 OpenSlide 可读性

我补了：

- [verify_stch_converted_wsi.py](/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/tools/verify_stch_converted_wsi.py)

这个脚本会检查：

1. `slide_id` 是否都有对应转换后 WSI
2. 转换后的文件是否能被 OpenSlide 打开
3. 每个文件的层级数、尺寸、vendor、objective 信息

使用方法：

```bash
cd /private/ljh-data/shared/ViLMIL

conda run -n vila_mil python ViLa-MIL-main/tools/verify_stch_converted_wsi.py \
  --wsi-dir ViLa-MIL-main/data/stch/wsi_converted
```

输出：

- `ViLa-MIL-main/eval_results/stch/stch_converted_wsi_validation.csv`
- `ViLa-MIL-main/eval_results/stch/stch_converted_wsi_validation_summary.txt`

---

## 5. 你现在应该怎么做

### Step 1：先生成 `.kfb` 别名

```bash
cd /private/ljh-data/shared/ViLMIL

conda run -n vila_mil python ViLa-MIL-main/tools/prepare_stch_kfb_aliases.py
```

### Step 2：用 KFB 专用软件打开 `wsi_kfb_alias/*.kfb`

推荐优先尝试厂商工具：

1. K-Viewer
2. KFSlideOS
3. 厂商提供的批量导出工具
4. 厂商 SDK / converter

你需要导出的不是截图，而是整张 WSI。

导出要求：

1. 保持全分辨率
2. 保持多层金字塔
3. 文件名主干不变
4. 输出到 `ViLa-MIL-main/data/stch/wsi_converted`

理想结果示例：

- `wsi_converted/1.svs`
- `wsi_converted/2.svs`

### Step 3：验证转换结果

```bash
cd /private/ljh-data/shared/ViLMIL

conda run -n vila_mil python ViLa-MIL-main/tools/verify_stch_converted_wsi.py \
  --wsi-dir ViLa-MIL-main/data/stch/wsi_converted
```

只要验证通过，你现有的外部验证文档就能直接接上。

---

## 6. 如果厂商工具支持批量导出，导出时这样选

如果软件里有导出选项，优先选择：

1. `SVS`
2. `Pyramidal TIFF`
3. `OME-TIFF`

导出参数建议：

1. 保留 pyramid / multiresolution
2. 保留 tile 结构
3. 不要 downsample 到普通平面图
4. 不要只导出 thumbnail
5. 不要改文件主干名

如果能选压缩，优先：

1. JPEG tile 压缩的 pyramidal TIFF / SVS
2. 无损或高质量 TIFF

---

## 7. 如果厂商工具不支持整张导出

那就需要向设备供应商要下面任一项：

1. KFB 批量转换器
2. KFB SDK
3. 命令行导出工具
4. 能导出标准 WSI 的最新版 Viewer

你当前环境里没有现成的 KFB 解码库，所以这一步没法只靠我在仓库里改几行 Python 来绕过去。

---

## 8. 转换完成后怎么接回现有流程

转换完成后，直接看：

- [stch_external_validation_post_conversion_cn.md](/private/ljh-data/shared/ViLMIL/ViLa-MIL-main/docs/stch_external_validation_post_conversion_cn.md)

从那里开始继续：

1. 坐标生成
2. patch 裁剪
3. BiomedCLIP 特征提取
4. 外部评估

---

## 9. 一句话结论

要让你现有程序正常使用这批 `.image`，正确做法是：

1. 先用 `prepare_stch_kfb_aliases.py` 把它们映射成 `.kfb`
2. 再用 KFB 专用软件把它们导出成 OpenSlide 可读的 `SVS / pyramidal TIFF`
3. 最后用 `verify_stch_converted_wsi.py` 验证转换结果

如果你能提供你手头实际可用的 KFB 软件名字，或者厂商给你的 converter 命令行，我下一步可以继续帮你把“批量转换命令”也接成仓库里的脚本。
