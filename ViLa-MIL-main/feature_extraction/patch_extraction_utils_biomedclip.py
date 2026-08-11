"""
BiomedCLIP特征提取工具函数
支持使用BiomedCLIP的图像编码器提取医学图像patch特征
"""

import importlib.util
import os
from pathlib import Path

DEFAULT_BIOMEDCLIP_REPO = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
DEFAULT_BIOMEDCLIP_MODEL = f'hf-hub:{DEFAULT_BIOMEDCLIP_REPO}'
DEFAULT_TEXT_REPO = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'


def _static_candidate_cache_dirs(explicit_cache_dir=None):
    """返回可能的 HuggingFace 缓存目录候选列表。"""
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        explicit_cache_dir,
        os.environ.get('HUGGINGFACE_HUB_CACHE'),
        repo_root / 'hf_cache',
        repo_root.parent / 'hf_cache',
        repo_root / 'model_cache',
    ]

    seen = set()
    resolved = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        candidate_str = str(candidate_path)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        resolved.append(candidate_path)
    return resolved


def _resolve_snapshot_dir(cache_dir, repo_id):
    """从 HuggingFace 缓存目录中解析某个仓库的快照目录。"""
    snapshots_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / 'snapshots'
    if not snapshots_dir.is_dir():
        return None

    snapshot_dirs = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshot_dirs:
        return None
    return snapshot_dirs[-1]


def _bootstrap_hf_environment():
    """在导入 open_clip / transformers 之前优先配置本地缓存环境。"""
    for candidate_cache_dir in _static_candidate_cache_dirs():
        if not candidate_cache_dir.exists():
            continue

        clip_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_BIOMEDCLIP_REPO)
        text_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_TEXT_REPO)
        if not clip_snapshot:
            continue

        os.environ.setdefault('HF_HOME', str(candidate_cache_dir))
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(candidate_cache_dir))
        if text_snapshot:
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        return str(candidate_cache_dir)

    return None


_BOOTSTRAP_CACHE_DIR = _bootstrap_hf_environment()

import h5py
import numpy as np
import torch
import torch.multiprocessing
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# BiomedCLIP依赖
from open_clip import create_model_from_pretrained, get_tokenizer

torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _candidate_cache_dirs(explicit_cache_dir=None):
    return _static_candidate_cache_dirs(explicit_cache_dir)


def _prepare_biomedclip_loading(model_path, cache_dir=None):
    """
    解析 BiomedCLIP 加载方式。

    优先使用本地缓存的 snapshot 目录，避免在网络不稳定时访问 HF Hub。
    """
    for candidate_cache_dir in _candidate_cache_dirs(cache_dir):
        if not candidate_cache_dir.exists():
            continue

        clip_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_BIOMEDCLIP_REPO)
        text_snapshot = _resolve_snapshot_dir(candidate_cache_dir, DEFAULT_TEXT_REPO)
        if not clip_snapshot:
            continue

        os.environ.setdefault('HF_HOME', str(candidate_cache_dir))
        os.environ.setdefault('HUGGINGFACE_HUB_CACHE', str(candidate_cache_dir))

        offline_enabled = text_snapshot is not None
        if offline_enabled:
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

        resolved_model_path = model_path
        if model_path == DEFAULT_BIOMEDCLIP_MODEL:
            resolved_model_path = f'local-dir:{clip_snapshot}'

        return resolved_model_path, str(candidate_cache_dir), offline_enabled

    return model_path, cache_dir, False


def _load_biomedclip_model(model_path, cache_dir=None, load_tokenizer=False):
    """加载 BiomedCLIP，并在失败时给出更明确的诊断信息。"""
    resolved_model_path, resolved_cache_dir, offline_enabled = _prepare_biomedclip_loading(
        model_path=model_path,
        cache_dir=cache_dir,
    )

    if resolved_model_path.startswith('local-dir:'):
        print(f"📁 Using local BiomedCLIP snapshot: {resolved_model_path[len('local-dir:'):]}")
    elif resolved_cache_dir:
        print(f"📦 Using HuggingFace cache: {resolved_cache_dir}")
    if offline_enabled:
        print('📴 Offline cache mode enabled')

    try:
        model, preprocess = create_model_from_pretrained(
            resolved_model_path,
            cache_dir=resolved_cache_dir,
        )
        tokenizer = get_tokenizer(resolved_model_path) if load_tokenizer else None
        return model, preprocess, tokenizer
    except Exception as exc:
        hints = []
        error_text = str(exc)

        if importlib.util.find_spec('transformers') is None:
            hints.append("缺少依赖 `transformers`，请执行 `conda run -n vila_mil pip install --upgrade 'transformers==4.46.3'`。")
        elif 'float8_e8m0fnu' in error_text:
            hints.append("当前 `transformers` 版本与 `torch` 不兼容，请执行 `conda run -n vila_mil pip install --upgrade 'transformers==4.46.3'`。")

        if 'Cannot assign requested address' in error_text or 'client has been closed' in error_text:
            hints.append('当前机器访问 HuggingFace 失败；如果模型已缓存，请确认缓存目录中包含 BiomedCLIP 和 BiomedBERT 两个仓库。')

        if resolved_cache_dir:
            hints.append(f'已尝试缓存目录: {resolved_cache_dir}')

        if hints:
            raise RuntimeError(f"{exc}\n" + "\n".join(hints)) from exc
        raise


def get_biomedclip_transforms():
    """
    BiomedCLIP图像预处理
    输入: 任意尺寸的RGB图像
    输出: 224x224 归一化张量
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # BiomedCLIP使用224x224
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),  # ImageNet统计量(BiomedCLIP通用)
            std=(0.229, 0.224, 0.225)
        )
    ])


class PatchesDataset(Dataset):
    """Patch数据集加载器"""
    def __init__(self, file_path, transform=None):
        file_names = os.listdir(file_path)
        self.imgs = [os.path.join(file_path, fn) for fn in file_names]
        self.coords = file_names
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        coord = self.coords[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, coord

    def __len__(self):
        return len(self.imgs)


def save_embeddings_biomedclip(model, fname, dataloader, overwrite=False):
    """
    使用BiomedCLIP提取并保存patch特征

    参数:
        model: BiomedCLIP模型
        fname: 输出文件名(不含.h5后缀)
        dataloader: Patch数据加载器
        overwrite: 是否覆盖已有文件
    """
    if os.path.isfile(f'{fname}.h5') and not overwrite:
        print(f'⏩ Skipping {fname}.h5 (already exists)')
        return None

    embeddings, coords, file_names = [], [], []

    # 提取图像特征
    for batch, coord in dataloader:
        with torch.no_grad():
            batch = batch.to(device)
            # BiomedCLIP图像编码器
            # 输出: [batch_size, 512] (ViT-B/16特征维度)
            image_features = model.encode_image(batch)
            embeddings.append(image_features.detach().cpu().numpy())
            file_names.append(coord)

    # 解析坐标(格式: slidename_x_y.png)
    for file_name in file_names:
        for coord in file_name:
            coord_parts = coord.rstrip('.png').split('_')
            if len(coord_parts) >= 2:
                try:
                    x = int(coord_parts[-2])  # 倒数第二个是x
                    y = int(coord_parts[-1])  # 最后一个是y
                    coords.append([x, y])
                except ValueError:
                    print(f'⚠️ Warning: 无法解析坐标 {coord}, 跳过')
                    continue

    # 保存到HDF5
    embeddings = np.vstack(embeddings)
    coords = np.vstack(coords)

    print(f'💾 Saving {fname}.h5 | Features: {embeddings.shape}, Coords: {coords.shape}')

    with h5py.File(f'{fname}.h5', 'w') as f:
        f.create_dataset('features', data=embeddings, compression='gzip')
        f.create_dataset('coords', data=coords, compression='gzip')


def create_embeddings_biomedclip(
    embeddings_dir,
    model_path,
    dataset,
    batch_size,
    patch_datasets='path/to/patches',
    cache_dir=None,
):
    """
    批量提取BiomedCLIP特征

    参数:
        embeddings_dir: 输出目录
        model_path: BiomedCLIP模型路径(HF hub或本地)
        dataset: 数据集名称
        batch_size: 批量大小
        patch_datasets: Patch根目录
        cache_dir: HuggingFace缓存目录
    """
    print(f"\n🔬 Extracting BiomedCLIP Features for '{dataset}'")
    print(f'📦 Model: {model_path}')
    print(f'🖼️ Device: {device}')

    # 加载BiomedCLIP模型
    model, _, _ = _load_biomedclip_model(
        model_path=model_path,
        cache_dir=cache_dir,
        load_tokenizer=False,
    )
    print('✅ BiomedCLIP model loaded successfully')

    model = model.to(device)
    model.eval()

    # 使用自定义transforms(而非preprocess,以保持与原代码一致)
    eval_transforms = get_biomedclip_transforms()

    # 获取特征维度
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = model.encode_image(dummy_input)
        feature_dim = dummy_output.shape[1]
    print(f'📐 Feature Dimension: {feature_dim}')

    # 遍历所有WSI
    wsi_list = os.listdir(patch_datasets)
    print(f'🔍 Found {len(wsi_list)} WSI folders')

    for wsi_name in tqdm(wsi_list, desc='Processing WSIs'):
        wsi_patch_dir = os.path.join(patch_datasets, wsi_name)

        # 跳过非目录文件
        if not os.path.isdir(wsi_patch_dir):
            continue

        # 创建数据加载器
        dataset_obj = PatchesDataset(wsi_patch_dir, transform=eval_transforms)

        # Windows下num_workers=0
        num_workers = 0 if os.name == 'nt' else 4
        dataloader = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        # 保存特征
        fname = os.path.join(embeddings_dir, wsi_name)
        save_embeddings_biomedclip(model, fname, dataloader)

    print(f'\n✨ All features saved to: {embeddings_dir}')


def extract_text_features_biomedclip(
    text_prompts,
    model_path=DEFAULT_BIOMEDCLIP_MODEL,
    output_path='text_features_biomedclip.pt',
    cache_dir=None,
):
    """
    使用BiomedCLIP提取文本特征(可选功能)

    参数:
        text_prompts: 文本提示列表 e.g., ["adenocarcinoma histopathology", ...]
        model_path: BiomedCLIP模型路径
        output_path: 输出文件路径
        cache_dir: HuggingFace缓存目录

    返回:
        text_features: [num_prompts, 512] torch.Tensor
    """
    print('\n📝 Extracting Text Features with BiomedCLIP')

    model, _, tokenizer = _load_biomedclip_model(
        model_path=model_path,
        cache_dir=cache_dir,
        load_tokenizer=True,
    )

    model = model.to(device)
    model.eval()

    # 文本编码(使用tokenizer而非tokenize函数)
    texts = tokenizer(text_prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(texts)

    # 归一化(可选,用于余弦相似度)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 保存
    torch.save(text_features, output_path)
    print(f'💾 Text features saved to: {output_path}')
    print(f'📐 Shape: {text_features.shape}')

    return text_features


# 示例用法
if __name__ == "__main__":
    # 测试图像特征提取
    test_prompts = [
        'adenocarcinoma histopathology',
        'normal tissue histopathology'
    ]

    text_features = extract_text_features_biomedclip(
        test_prompts,
        output_path='test_text_features.pt'
    )

    print('\n✅ Text feature extraction test passed!')
    print(f'Feature shape: {text_features.shape}')
