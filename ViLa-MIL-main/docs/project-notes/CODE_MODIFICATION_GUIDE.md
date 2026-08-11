# ViLa-MIL BiomedCLIP集成 - 代码修改说明

## 📝 需要修改的文件: `utils/core_utils.py`

### 修改位置: `train()` 函数中的模型初始化部分

**文件路径:** `d:\FenLei\ViLa-MIL-main\utils\core_utils.py`

**大约行号:** 134-145行

---

### 原始代码(第134-145行):

```python
if args.model_type == 'ViLa_MIL':
    print('\nViLa_MIL模型初始化')
    from models.model_ViLa_MIL import ViLa_MIL_Model
    
    model_dict = {
        'input_size': 1024,  # CLIP RN50特征维度
        'hidden_size': 256,
        'text_prompt': args.text_prompt,
        'prototype_number': args.prototype_number
    }
    
    model = ViLa_MIL_Model(**model_dict)
```

---

### 修改后代码:

```python
if args.model_type == 'ViLa_MIL_BiomedCLIP':
    # BiomedCLIP版本
    print('\n🔬 BiomedCLIP-based ViLa_MIL模型初始化')
    from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
    
    model_dict = {
        'input_size': 512,  # BiomedCLIP ViT-B/16特征维度
        'hidden_size': 256,
        'text_prompt': args.text_prompt,
        'prototype_number': args.prototype_number
    }
    
    class Config:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    model = ViLa_MIL_BiomedCLIP(
        config=Config(**model_dict),
        num_classes=args.n_classes
    )
    
elif args.model_type == 'ViLa_MIL':
    # 原始CLIP版本(保持不变)
    print('\nViLa_MIL模型初始化')
    from models.model_ViLa_MIL import ViLa_MIL_Model
    
    model_dict = {
        'input_size': 1024,  # CLIP RN50特征维度
        'hidden_size': 256,
        'text_prompt': args.text_prompt,
        'prototype_number': args.prototype_number
    }
    
    model = ViLa_MIL_Model(**model_dict)
```

---

## 🔧 修改步骤

### 步骤1: 找到修改位置
1. 打开文件: `d:\FenLei\ViLa-MIL-main\utils\core_utils.py`
2. 按 `Ctrl+F` 搜索: `if args.model_type == 'ViLa_MIL':`
3. 应该定位到第134行附近

### 步骤2: 添加BiomedCLIP分支
在 `if args.model_type == 'ViLa_MIL':` **之前**添加新的if分支:

```python
# 在第134行之前插入
if args.model_type == 'ViLa_MIL_BiomedCLIP':
    print('\n🔬 BiomedCLIP-based ViLa_MIL模型初始化')
    from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
    
    model_dict = {
        'input_size': 512,  # BiomedCLIP特征维度
        'hidden_size': 256,
        'text_prompt': args.text_prompt,
        'prototype_number': args.prototype_number
    }
    
    class Config:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    model = ViLa_MIL_BiomedCLIP(
        config=Config(**model_dict),
        num_classes=args.n_classes
    )
```

### 步骤3: 修改原有分支
将原来的 `if args.model_type == 'ViLa_MIL':` 改为 `elif args.model_type == 'ViLa_MIL':`

---

## ✅ 验证修改

修改完成后,运行以下命令验证:

```bash
python -c "import sys; sys.path.insert(0, 'utils'); from core_utils import train; print('✅ 修改成功!')"
```

如果没有报错,说明修改正确。

---

## 📊 完整修改后的代码段

```python
# utils/core_utils.py 第134行附近

# BiomedCLIP版本
if args.model_type == 'ViLa_MIL_BiomedCLIP':
    print('\n🔬 BiomedCLIP-based ViLa_MIL模型初始化')
    from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
    
    model_dict = {
        'input_size': 512,  # BiomedCLIP ViT-B/16特征维度
        'hidden_size': 256,
        'text_prompt': args.text_prompt,
        'prototype_number': args.prototype_number
    }
    
    class Config:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    model = ViLa_MIL_BiomedCLIP(
        config=Config(**model_dict),
        num_classes=args.n_classes
    )

# 原始CLIP版本
elif args.model_type == 'ViLa_MIL':
    print('\nViLa_MIL模型初始化')
    from models.model_ViLa_MIL import ViLa_MIL_Model
    
    model_dict = {
        'input_size': 1024,  # CLIP RN50特征维度
        'hidden_size': 256,
        'text_prompt': args.text_prompt,
        'prototype_number': args.prototype_number
    }
    
    model = ViLa_MIL_Model(**model_dict)

else:
    raise NotImplementedError(f"Model type '{args.model_type}' not recognized")
```

---

## 🎯 关键差异说明

| 参数 | 原始ViLa_MIL | BiomedCLIP版本 |
|------|-------------|---------------|
| `input_size` | 1024 | **512** |
| 模型类 | `ViLa_MIL_Model` | **`ViLa_MIL_BiomedCLIP`** |
| 导入路径 | `models.model_ViLa_MIL` | **`models.model_ViLa_MIL_BiomedCLIP`** |
| 初始化方式 | `Model(**dict)` | **`Model(config=Config(), num_classes=...)`** |

---

## 🚨 常见错误

### 错误1: 忘记修改`elif`
**症状:** 两个版本都会被执行

**解决:** 确保第一个是 `if`,第二个是 `elif`

### 错误2: 缩进错误
**症状:** `IndentationError`

**解决:** 确保所有代码使用4个空格缩进,与原代码对齐

### 错误3: 特征维度不匹配
**症状:** `RuntimeError: size mismatch, m1: [1 x 1024], m2: [512 x 256]`

**原因:** 使用了旧的CLIP特征(1024维)但模型是BiomedCLIP(512维)

**解决:** 确保使用正确的特征文件夹:
```bash
--data_folder_s features_biomedclip_5x   # ✅ 正确
--data_folder_s features_5x              # ❌ 错误(这是CLIP特征)
```

---

## 📖 修改完成后的下一步

1. ✅ 验证修改: `python test_biomedclip_integration.py`
2. ✅ 提取特征: `python feature_extraction/patch_extraction_biomedclip.py ...`
3. ✅ 训练模型: `python main.py --model_type ViLa_MIL_BiomedCLIP ...`
4. ✅ 评估模型: `python eval.py --model_type ViLa_MIL_BiomedCLIP ...`

详细步骤见: `BIOMEDCLIP_TODO.md`
