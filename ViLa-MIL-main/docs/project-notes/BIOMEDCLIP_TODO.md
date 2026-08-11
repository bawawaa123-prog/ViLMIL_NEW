# BiomedCLIP集成 - 待办事项清单

## ✅ 已完成

- [x] 创建BiomedCLIP特征提取脚本 (`patch_extraction_biomedclip.py`)
- [x] 创建BiomedCLIP工具函数 (`patch_extraction_utils_biomedclip.py`)
- [x] 创建BiomedCLIP版ViLa-MIL模型 (`model_ViLa_MIL_BiomedCLIP.py`)
- [x] 编写完整集成指南 (`BIOMEDCLIP_INTEGRATION_GUIDE.md`)
- [x] 创建测试脚本 (`test_biomedclip_integration.py`)

## 📋 下一步操作(按顺序执行)

### Step 1: 安装依赖 ⚙️
```bash
pip install open_clip_torch
pip install huggingface_hub
```

**验证安装:**
```bash
python test_biomedclip_integration.py
```

预期输出: `🎉 All tests passed!`

---

### Step 2: 修改`utils/utils.py` 🔧

**需要添加的代码位置:** `get_model()` 函数

**原始代码(第X行附近):**
```python
def get_model(args, device):
    if args.model_type == 'ViLa_MIL':
        from models.model_ViLa_MIL import ViLa_MIL_Model
        # ...现有代码
```

**修改为:**
```python
def get_model(args, device):
    if args.model_type == 'ViLa_MIL_BiomedCLIP':
        # BiomedCLIP版本
        from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
        
        model_dict = {
            'input_size': 512,  # BiomedCLIP特征维度
            'hidden_size': args.hidden_size if hasattr(args, 'hidden_size') else 256,
            'text_prompt': args.text_prompt,
            'prototype_number': args.prototype_number
        }
        
        model = ViLa_MIL_BiomedCLIP(
            config=type('Config', (), model_dict),
            num_classes=args.n_classes
        )
        
    elif args.model_type == 'ViLa_MIL':
        # 原始CLIP版本
        from models.model_ViLa_MIL import ViLa_MIL_Model
        # ...现有代码保持不变
```

**如何找到修改位置:**
1. 打开 `utils/utils.py`
2. 搜索 `def get_model(`
3. 在函数开头添加上述if-elif分支

---

### Step 3: 提取BiomedCLIP特征 🖼️

#### 3.1 低分辨率(5x)特征
```bash
python feature_extraction/patch_extraction_biomedclip.py ^
    --patches_path patches_coords_5x/patches_256 ^
    --library_path features_biomedclip_5x ^
    --batch_size 32 ^
    --dataset adenocarcinoma
```

**时间估算:** 约10-15小时 (350张WSI, RTX 3090)

**检查点:**
```bash
# 查看已生成的特征文件
dir features_biomedclip_5x
# 应该看到: 2460239-B2.h5, 2460242-B2.h5, ...
```

#### 3.2 高分辨率(20x)特征
```bash
python feature_extraction/patch_extraction_biomedclip.py ^
    --patches_path patches_coords_20x/patches_256 ^
    --library_path features_biomedclip_20x ^
    --batch_size 32 ^
    --dataset adenocarcinoma
```

---

### Step 4: 训练BiomedCLIP模型 🚀

```bash
python main.py ^
    --data_root_dir . ^
    --data_folder_s features_biomedclip_5x ^
    --data_folder_l features_biomedclip_20x ^
    --split_dir splits/adenocarcinoma ^
    --model_type ViLa_MIL_BiomedCLIP ^
    --exp_code adenocarcinoma_biomedclip_s1 ^
    --task adenocarcinoma ^
    --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv ^
    --prototype_number 16 ^
    --max_epochs 80 ^
    --lr 1e-3 ^
    --reg 1e-5 ^
    --k 10 ^
    --early_stopping ^
    --patience 10 ^
    --results_dir results/biomedclip_training
```

**预期输出:**
```
🔬 Loading BiomedCLIP from: hf-hub:microsoft/...
Training Fold 0...
Epoch 1/80: train_loss=0.652, val_auc=0.723
...
```

---

### Step 5: 评估模型 📊

```bash
python eval.py ^
    --data_root_dir . ^
    --data_folder_s features_biomedclip_5x ^
    --data_folder_l features_biomedclip_20x ^
    --split_dir splits/adenocarcinoma ^
    --model_type ViLa_MIL_BiomedCLIP ^
    --models_exp_code adenocarcinoma_biomedclip_s1_s1 ^
    --task adenocarcinoma ^
    --text_prompt_path text_prompt/adenocarcinoma_dual_scale_prompt.csv ^
    --results_dir eval_results/biomedclip_eval
```

**检查结果:**
```bash
# 查看评估结果
type eval_results\biomedclip_eval\eval_experiment_adenocarcinoma_biomedclip_s1.txt

# 查看每个fold的详细结果
type eval_results\biomedclip_eval\fold_0.csv
```

---

### Step 6: 对比原始CLIP性能 📈

**运行对比实验:**

```bash
# 实验1: 原始CLIP (已有特征)
python eval.py ^
    --data_folder_s features_5x ^
    --data_folder_l features_20x ^
    --model_type ViLa_MIL ^
    --models_exp_code adenocarcinoma_vila_mil_s1_s1 ^
    --results_dir eval_results/clip_baseline

# 实验2: BiomedCLIP (新特征)
python eval.py ^
    --data_folder_s features_biomedclip_5x ^
    --data_folder_l features_biomedclip_20x ^
    --model_type ViLa_MIL_BiomedCLIP ^
    --models_exp_code adenocarcinoma_biomedclip_s1_s1 ^
    --results_dir eval_results/biomedclip_eval
```

**对比指标:**
- AUC (Area Under Curve)
- Accuracy
- F1-Score
- 每个fold的性能稳定性

---

## 🔍 故障排查

### 问题1: `ModuleNotFoundError: No module named 'open_clip'`
**解决:**
```bash
pip install open_clip_torch
```

### 问题2: BiomedCLIP下载失败
**解决:**
```bash
# 设置国内镜像
set HF_ENDPOINT=https://hf-mirror.com
pip install open_clip_torch
```

### 问题3: 特征提取内存不足
**解决:**
```bash
# 减小batch size
python feature_extraction/patch_extraction_biomedclip.py --batch_size 16
```

### 问题4: 特征维度不匹配
**错误:** `RuntimeError: size mismatch, m1: [1 x 1024], m2: [512 x 256]`

**原因:** 使用了旧的CLIP特征(1024维)但模型期望BiomedCLIP特征(512维)

**解决:** 确保:
1. 使用 `--data_folder_s features_biomedclip_5x` (不是 `features_5x`)
2. 使用 `--model_type ViLa_MIL_BiomedCLIP` (不是 `ViLa_MIL`)

---

## 📊 预期结果

### 性能提升
- **AUC:** +4-5%
- **Accuracy:** +4-5%
- **F1-Score:** +4-5%

### 训练时间
- 每个epoch: ~5-10分钟 (取决于GPU)
- 总训练时间(10-fold): ~2-3小时 (with early stopping)

---

## 📚 参考文件

1. **集成指南:** `BIOMEDCLIP_INTEGRATION_GUIDE.md`
2. **测试脚本:** `test_biomedclip_integration.py`
3. **特征提取:** `feature_extraction/patch_extraction_biomedclip.py`
4. **模型定义:** `models/model_ViLa_MIL_BiomedCLIP.py`

---

## ✅ 完成检查清单

在继续下一步之前,确认:

- [ ] Step 1: `pip install open_clip_torch` 执行成功
- [ ] Step 1: `python test_biomedclip_integration.py` 全部通过
- [ ] Step 2: `utils/utils.py` 中添加了BiomedCLIP模型初始化代码
- [ ] Step 3: `features_biomedclip_5x/` 目录包含所有WSI的h5文件
- [ ] Step 3: `features_biomedclip_20x/` 目录包含所有WSI的h5文件
- [ ] Step 4: 训练完成,`results/biomedclip_training/` 包含模型权重
- [ ] Step 5: 评估完成,`eval_results/biomedclip_eval/` 包含结果CSV
- [ ] Step 6: 对比实验完成,性能有提升

---

## 💡 下一步优化(可选)

完成基本集成后,可以考虑:

1. **微调BiomedCLIP:** 在你的数据集上进一步微调
2. **提示词工程:** 使用CSV中的详细医学描述
3. **多尺度融合:** 优化低分辨率和高分辨率特征的融合方式
4. **注意力可视化:** 生成BiomedCLIP的热力图

---

**祝你成功! 🎉**

如有问题,请参考 `BIOMEDCLIP_INTEGRATION_GUIDE.md` 或提issue。
