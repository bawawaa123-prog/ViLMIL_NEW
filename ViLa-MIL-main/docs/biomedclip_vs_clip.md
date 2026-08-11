# BiomedCLIP 与原始 CLIP 流程详细对比

> 说明：应你的请求，将 ViLa-MIL 原始 CLIP 流程与 BiomedCLIP 集成方案的差异、结构、维度流向放在同一文档，方便查阅。

## 1. 总体差异概览

| 维度 | 原始流程 (CLIP RN50) | BiomedCLIP 流程 |
| --- | --- | --- |
| 图像编码器 | `clip.load("RN50")`，ResNet-50 主干 | `open_clip.create_model_from_pretrained("BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")`，ViT-B/16 主干 |
| 文本编码器 | OpenAI CLIP Transformer，tokenizer 使用 `clip.simple_tokenizer` | PubMedBERT (HuggingFace 架构)，tokenizer 使用 `open_clip` 提供的 `get_tokenizer` |
| Patch 特征维度 | 1024 | 512 |
| Prompt 生成 | 手工定位 `token_embedding`，上下文拼接多种模式 | 需通过 `text.transformer.embeddings.word_embeddings` 获取嵌入，BERT 结构 |
| 训练入口 | `main.py --model_type ViLa_MIL` | `main.py --model_type ViLa_MIL_BiomedCLIP` |
| 依赖 | `clip` | `open_clip_torch`、`transformers` |

## 2. 特征提取流程对比

### 2.1 调用脚本

- **原始 CLIP**：`feature_extraction/patch_extraction.py`
  - 核心函数：`patch_extraction_utils.create_embeddings`
  - 常见命令：
    ```bash
    python feature_extraction/patch_extraction.py \
        --patches_path patches_coords_5x/patches_256 \
        --library_path features_clip_5x \
        --model_name clip_RN50 \
        --batch_size 32
    ```
- **BiomedCLIP**：`feature_extraction/patch_extraction_biomedclip.py`
  - 核心函数：`patch_extraction_utils_biomedclip.create_embeddings_biomedclip`
  - 常见命令：
    ```bash
    python feature_extraction/patch_extraction_biomedclip.py \
        --patches_path patches_coords_5x/patches_256 \
        --library_path features_biomedclip_5x \
        --batch_size 32 \
        --dataset adenocarcinoma
    ```

### 2.2 流程差异

1. **模型加载**
   - CLIP：`clip.load("RN50", device=device)` → 返回模型与预处理器。
   - BiomedCLIP：`create_model_from_pretrained(model_path)` → 返回 `CustomTextCLIP` （包含 ViT 和 PubMedBERT）。
2. **图像预处理**
   - CLIP：`eval_transforms_clip`（`ToTensor` → `Resize(224,224)` → Normalize）。
   - BiomedCLIP：`get_biomedclip_transforms`（保持 224×224 + ImageNet 均值方差）。
3. **特征导出**
   - CLIP：经 ResNet50 截断后的 1024 维向量；存入 `.h5`。
   - BiomedCLIP：经 ViT-B/16 + CLS token 的 512 维向量；同样 `.h5`。
4. **Token 处理**
   - CLIP：部分流程会调用 `clip.tokenize`，旧版 `tokenize()` 容易触发 CUDA 越界。
   - BiomedCLIP：必须使用 `tokenizer(prompts)`（`open_clip` 提供）。

> **注意**：训练时 `--library_path` / `--data_folder_*` 必须匹配对应维度，否则将报 `size mismatch`。

## 3. 训练阶段差异

### 3.1 模型初始化

- `utils/core_utils.py` 内部根据 `args.model_type` 选择模型：
  ```python
  if args.model_type == 'ViLa_MIL_BiomedCLIP':
      from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
      model_dict = {
          'input_size': 512,
          'hidden_size': 256,
          'text_prompt': args.text_prompt,
          'prototype_number': args.prototype_number,
      }
      model = ViLa_MIL_BiomedCLIP(config=Config(**model_dict), num_classes=args.n_classes)
  elif args.model_type == 'ViLa_MIL':
      from models.model_ViLa_MIL import ViLa_MIL_Model
      model_dict = {
          'input_size': 1024,
          'hidden_size': 256,
          'text_prompt': args.text_prompt,
          'prototype_number': args.prototype_number,
      }
      model = ViLa_MIL_Model(config=Config(**model_dict), num_classes=args.n_classes)
  ```

### 3.2 文本编码部分

| 模块 | CLIP 版本 (`models/model_ViLa_MIL.py`) | BiomedCLIP 版本 (`models/model_ViLa_MIL_BiomedCLIP.py`) |
| --- | --- | --- |
| PromptLearner | 直接访问 `clip_model.token_embedding` 获取前缀/后缀 | 需要通过 `text.transformer.embeddings.word_embeddings` 获取 BERT Embedding |
| TextEncoder | `transformer` + `ln_final` + `text_projection` | `BiomedCLIPTextEncoder` 调 `encode_text`，内部冻结 (`torch.no_grad()`) |
| Tokenizer | `_Tokenizer` / `clip.tokenize` | `tokenizer = get_tokenizer(model_path)` |
| Prompt 长度 | 默认上下文 16；支持前/中/后插入 | 同为 16，但固定 `[SOS] + ctx + class name + [EOS]` |

### 3.3 训练循环

- `train_loop` / `validate` / `summary` 等函数对两种模型完全通用。
- 损失：`nn.CrossEntropyLoss()`；优化器：`get_optim`（与 `args` 配置一致）。
- 唯一的代码差异是模型内部的张量维度及前向逻辑。

## 4. 模型结构与维度流向

### 4.1 关键层尺寸对比

| 名称 | CLIP 版本 | BiomedCLIP 版本 |
| --- | --- | --- |
| 输入特征 `x_s/x_l` | `[1, N, 1024]` | `[1, N, 512]` |
| 可学习原型 `learnable_image_center` | `[prototype_number, 1, 1024]` | `[prototype_number, 1, 512]` |
| Attention 映射 `A` | `[1, N]`（与原型数有关） | 同上 |
| image 特征 `image_features_*` | `[1, 1024]` | `[1, 512]` |
| 文本特征 `text_features_*` | `[num_classes, 1024]` | `[num_classes, 512]` |
| 最终 logits | `[1, num_classes]` | `[1, num_classes]` |

### 4.2 前向过程（BiomedCLIP）

1. **Prompt 生成**：`BiomedCLIPPromptLearner` 拼出 `[n_cls, seq_len, 512]` 的 prompt 嵌入。
2. **文本编码**：`tokenizer(prompts)` → `encode_text` → 输出 `[2 * num_classes, 512]`（通常按低/高分辨率拆分）。
3. **图像分支**：
   - 输入 `x_s` / `x_l`（预提取的 512 维 patch features）。
   - 通过 `MultiheadAttention` 与 `learnable_image_center` 交互，聚合得到 `image_features_low` / `image_features_high`。
4. **跨模态对齐**：
   - `cross_attention_2` 将文本原型与图像上下文拼接对齐（尺寸仍保持 512）。
5. **分类**：`logits_low = image_features_low @ text_features_low.T`，同理高分辨率；两者求和进入 softmax。

### 4.3 维度注意事项

- 任何线性层的输入维度都与 `self.L` 绑定（512 vs 1024）。
- `LayerNorm`、`MultiheadAttention` 的 `embed_dim` 必须匹配当前特征维。
- 若误用旧特征，最早在 `self.cross_attention_1` 或 `nn.Linear(self.L, self.D)` 会报维度不匹配。

## 5. 配置与命令示例

### 5.1 训练命令

```bash
# 原始 CLIP 版本
python main.py \
    --model_type ViLa_MIL \
    --data_folder_s features_clip_5x \
    --data_folder_l features_clip_20x \
    --text_prompt prompts/adenocarcinoma_dual_scale_prompt.csv \
    ...

# BiomedCLIP 版本
python main.py \
    --model_type ViLa_MIL_BiomedCLIP \
    --data_folder_s features_biomedclip_5x \
    --data_folder_l features_biomedclip_20x \
    --text_prompt prompts/adenocarcinoma_dual_scale_prompt.csv \
    --prototype_number 16 \
    ...
```

### 5.2 评估命令（同理传入 `--model_type`）

```bash
python eval.py --model_type ViLa_MIL_BiomedCLIP --model_path trained_models/... --results_dir eval_results/...
```

## 6. 实践建议

1. **特征目录区分清楚**：建议以 `features_clip_*`、`features_biomedclip_*` 命名，避免覆盖。
2. **Prompt 文本**：两种版本都可使用 `text_prompt/*.csv`；BiomedCLIP 对医学术语更加敏感，可探索多样化 prompt。
3. **任务切换**：如果要比较两种方法，保持数据划分 (`splits/`)、训练参数一致，仅替换特征与 `--model_type`。
4. **依赖管理**：BiomedCLIP 需安装 `open_clip_torch>=2.23` 与 `transformers`；原版本只依赖 `clip`。
5. **GPU 内存**：BiomedCLIP 的 ViT-B/16 特征更紧凑（512 维），通常更省显存，但 `encode_text` 在推理时仍需注意 batch 大小。

## 7. 关键文件索引

| 功能 | CLIP 版本 | BiomedCLIP 版本 |
| --- | --- | --- |
| 特征提取脚本 | `feature_extraction/patch_extraction.py` | `feature_extraction/patch_extraction_biomedclip.py` |
| 特征提取工具 | `feature_extraction/patch_extraction_utils.py` | `feature_extraction/patch_extraction_utils_biomedclip.py` |
| MIL 模型实现 | `models/model_ViLa_MIL.py` | `models/model_ViLa_MIL_BiomedCLIP.py` |
| 训练入口 | `main.py` + `utils/core_utils.py` | 同文件，通过 `--model_type` 切换 |
| Prompt CSV | `text_prompt/*.csv` | 同上 |

---

如需在文档中补充实验记录，可继续在此文件追加章节。