# 双尺度提示 (Dual-scale Prompts) 使用说明（详细）

本文档解释项目中是否仍在使用双尺度提示文本（Dual-scale Prompts），以及它们如何加载、处理并在训练中被使用（包括 CLIP 与 BiomedCLIP 两个流程）。

---

## 一、结论（简短）

是的 —— 系统仍然支持并使用“双尺度提示”机制（低分辨率 + 高分辨率描述）。

- 在 `main.py` 中，CSV（例如 `text_prompt/adenocarcinoma_dual_scale_prompt.csv`）被解析为 `args.text_prompt = [low_prompts..., high_prompts...]` 的扁平列表（长度应为 2 * num_classes）。
- 在模型前向中（`models/model_ViLa_MIL.py` 与 `models/model_ViLa_MIL_BiomedCLIP.py`），`text_features` 被切分为 `text_features_low = text_features[:num_classes]` 与 `text_features_high = text_features[num_classes:]`，分别用于低/高分辨率通道的跨模态对齐与分类。

---

## 二、代码关键点（加载、拼装、切分）

1) `main.py` 中 CSV 解析并生成 `args.text_prompt`：

- 核心逻辑在 `main.py` 中：

    - 读取 CSV (`args.text_prompt_path`) — CSV 里每一行代表一个 class（例如 `class_name,low_resolution_description,high_resolution_description`）。
    - low_prompts = 第一列（或 `low_resolution_description` 字段），high_prompts = 第二列（或 `high_resolution_description` 字段）。
    - 最终组合：`args.text_prompt = low_prompts + high_prompts`（因此扁平顺序为 [low-Class0, low-Class1, ..., high-Class0, high-Class1, ...]）。

- 若 CSV 只有单列（仅 low 描述），则 `high_prompts = []`；`main.py` 会打印警告："The number of text prompts does not match 2 x n_classes."。但是脚本仍然会继续运行（可能导致 high 部分为空，需要谨慎）。

关键片段（`main.py`）：
```
# Compose in expected order: [low... , high...]
args.text_prompt = list(map(str, low_prompts)) + list(map(str, high_prompts))
```

同时有一个完整的双尺度例子文件：
- `text_prompt/adenocarcinoma_dual_scale_prompt.csv`，包含 `class_name`,`low_resolution_description`,`high_resolution_description` 三列。

2) 模型中如何使用扁平 prompt 列表（双尺度）

- 在 CLIP 的 `ViLa_MIL_Model`：
  - `self.prompt_learner` 读取 `config.text_prompt`（即 `args.text_prompt`）并 tokenizes（`clip.tokenize`）统一处理。
  - `text_features = self.text_encoder(prompts, tokenized_prompts)`。
  - 切分：
    ```python
    text_features_low = text_features[:self.num_classes]
    text_features_high = text_features[self.num_classes:]
    ```
  - `text_features_low` 和 `text_features_high` 分别用于各自分辨率通道的 cross-attention 对齐。

- 在 BiomedCLIP 的 `ViLa_MIL_BiomedCLIP`：
  - `self.prompt_learner = BiomedCLIPPromptLearner(config.text_prompt, biomedclip_model, tokenizer)` 接收同样的扁平 prompt 列表（低分辨 + 高分辨）
  - 然后 `text_features = self.text_encoder(tokenized_prompts)`（调用 `model.encode_text`）
  - 同样切分为 `text_features_low` 与 `text_features_high`（上面 slicing 语句）进行低/高分辨率通道对齐。

> 注意：所有模型前向都依赖 `args.text_prompt` 长度为 `2 * n_classes`（默认顺序是低分辨 + 高分辨），否则会发出警告或出现索引长度不匹配的问题。

---

## 三、示例（n_classes=2）

- 假设 2 个类：`Adenocarcinoma`、`NonAdenocarcinoma`；CSV 中每行有 low/high 两列。
- `main.py` 解构后：
  - `args.text_prompt` = [ low(Adeno), low(NonAdeno), high(Adeno), high(NonAdeno) ] （长度 4）

- 在模型中：
  - `tokenized_prompts` 对上述 4 个 prompt 逐个 tokenized
  - `text_features` 将是 [4, 512] 或 [4, 1024]（基于模型）
  - `text_features_low = text_features[:2]` -> `[low(Adeno), low(NonAdeno)]`
  - `text_features_high = text_features[2:]` -> `[high(Adeno), high(NonAdeno)]`
  - 分别将 `text_features_low` 用于 5x 分支、`text_features_high` 用于 20x 分支。

---

## 四、常见配置/定制场景

- 如果你不想使用 dual-scale（只使用低分辨提示）:
  - 方案1（推荐）: 在 CSV 只写 low 描述（一列），但模型会警告“没匹配到 high prompts”。解决方法：将低分辨描述复制到高分辨描述列，或创建只有 low 的 prompt 列表然后通过程序把高分 prompt set 为 `low_prompts`（手动构造 `args.text_prompt = low_prompts + low_prompts`）。
  - 方案2: 在 `main.py` 中绕开 CSV 解析，直接把 `--text_prompt` 设置为 `"[low_prompt_class0, low_prompt_class1, ...]"` 并确保长度为 `2*n_classes`（可以令后半段与前半段相同）。

- 如果你想更改双尺度排序（例如希望 prompt 按 class interleaved: low(c0), high(c0), low(c1), high(c1)），则需要同步修改模型中 `text_features` 的切分与索引（或在 `main.py` 生成新的排列），但默认现在是 [all low] + [all high]，并在模型中按 `[:n]` 和 `[n:]` 划分。

---

## 五、脚本举证（指向关键文件）

- CSV 示例：
  - `text_prompt/adenocarcinoma_dual_scale_prompt.csv`（含 `class_name`, `low_resolution_description`, `high_resolution_description`）
- prompt 列表构建：
  - `main.py`：解析 CSV 将 `args.text_prompt = low_prompts + high_prompts`
- CLIP 版本：
  - 文件：`models/model_ViLa_MIL.py`
  - 关键片段：`prompts = self.prompt_learner(); tokenized_prompts = self.prompt_learner.tokenized_prompts; text_features = self.text_encoder(prompts, tokenized_prompts); text_features_low = text_features[:self.num_classes]; text_features_high = text_features[self.num_classes:]`
- BiomedCLIP 版本：
  - 文件：`models/model_ViLa_MIL_BiomedCLIP.py`
  - 关键片段：构造 `BiomedCLIPPromptLearner(config.text_prompt, ...)`; 在 `forward` 中 `tokenized_prompts = self.prompt_learner.tokenized_prompts.to(device); text_features = self.text_encoder(tokenized_prompts); text_features_low = text_features[:self.num_classes]; text_features_high = text_features[self.num_classes:]`
- Script: `scripts/extract_modified_text_features.py` 演示如何解析双尺度 CSV 并（示例中的实现）使用 low_resolution 描述作为 prompt，请查阅片段中如何 `args.text_prompt` 被替换以供后续分析步骤调用。

---

## 六、建议（实施与调试）

- 若你想强制“只用 low”（降低复杂性/调试目的），请在 `main.py` 生成 `args.text_prompt = low_prompts + low_prompts`（复制 low 列），或在命令行中直接传入 `--text_prompt` 为一个已构造的 2*n list。
- 若欲对比实验：
  - 保持训练集/划分/split 等所有外部条件一致，切换 `--text_prompt_path` 文件设计（双尺度 vs 低尺度重复），比较 AUC/F1 等指标。
- 编码兼容性：
  - 训练前请核对 `args.text_prompt` 的长度是否等于 `2 * n_classes`。
  - `main.py` 会打印 `Text prompts loaded: {len(args.text_prompt)} items (expected {expected} = 2 x n_classes)`，这是用于快速 sanity check。

---

## 七、结论（详细）

- 项目仍然使用双尺度提示机制：`args.text_prompt` 明确由 `low_prompts + high_prompts` 组成；模型（CLIP / BiomedCLIP）在 forward 将 `text_features` 切分为 `low` 与 `high` 两段并对应图像的低/高分辨率分支。
- 在 BiomedCLIP 的集成中，行为没有改变（同样切分并分别用于低/高通道）。BiomedCLIP 只是替换了文本编码器和图像编码器（维度从 1024→512），但模态对齐的双尺度 prompt 机制仍然被保留。

如果你愿意，我可以：
- 在 `README` 或 `docs/biomedclip_vs_clip.md` 中添加这份双尺度专门解释章节（可选）；
- 编写一个小脚本用于快速验证：给定一个 CSV 或 `--text_prompt`，输出 `args.text_prompt` 列表、长度、并验证 `2*n` 匹配；或，在训练前自动对不完整的 prompt 列表做延展（复制 low→high）。

---

文件自动生成：`docs/dual_scale_prompt_usage.md`（此文件）