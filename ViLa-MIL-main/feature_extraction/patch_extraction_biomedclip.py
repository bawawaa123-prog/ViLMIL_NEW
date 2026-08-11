#!/usr/bin/env python
"""
BiomedCLIP特征提取脚本
用法:
    python feature_extraction/patch_extraction_biomedclip.py \
        --patches_path patches_coords_5x/patches_256 \
        --library_path features_biomedclip_5x \
        --batch_size 32 \
        --dataset adenocarcinoma
"""

import argparse
import os

import torch
import torch.multiprocessing
from PIL import ImageFile

from patch_extraction_utils_biomedclip import create_embeddings_biomedclip

torch.multiprocessing.set_sharing_strategy('file_system')
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='BiomedCLIP Feature Extraction')
parser.add_argument('--patches_path', type=str, required=True,
                    help='Path to patches directory (e.g., patches_coords_5x/patches_256)')
parser.add_argument('--library_path', type=str, required=True,
                    help='Output directory for h5 features')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for feature extraction')
parser.add_argument('--dataset', type=str, default='adenocarcinoma',
                    help='Dataset name')
parser.add_argument('--model_path', type=str,
                    default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                    help='BiomedCLIP model path or HuggingFace hub ID')
parser.add_argument('--cache_dir', type=str, default=None,
                    help='Optional HuggingFace cache directory; defaults to auto-discovery')

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    args = parser.parse_args()

    os.makedirs(args.library_path, exist_ok=True)

    print('=' * 60)
    print('BiomedCLIP Feature Extraction Configuration')
    print('=' * 60)
    print(f'Patches Path: {args.patches_path}')
    print(f'Output Path: {args.library_path}')
    print(f'Batch Size: {args.batch_size}')
    print(f'Dataset: {args.dataset}')
    print(f'Model: {args.model_path}')
    print(f'Cache Dir: {args.cache_dir or "auto"}')
    print('=' * 60)

    create_embeddings_biomedclip(
        patch_datasets=args.patches_path,
        embeddings_dir=args.library_path,
        model_path=args.model_path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    print('\n✅ Feature extraction completed!')
