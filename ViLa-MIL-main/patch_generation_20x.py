#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
from functools import partial

import h5py
import pandas as pd

from wsi_core.WholeSlideImage import WholeSlideImage

SUPPORTED_WSI_SUFFIXES = ('.svs', '.ome.tif', '.ome.tiff', '.tif', '.tiff')
WSI_SEARCH_SUBDIRS = ('', 'benign', 'non_benign')


def normalize_slide_id(slide_name):
    slide_name = str(slide_name).strip()
    lower_name = slide_name.lower()
    for suffix in ('.ome.tiff', '.ome.tif', '.svs', '.tiff', '.tif'):
        if lower_name.endswith(suffix):
            return slide_name[: len(slide_name) - len(suffix)]
    return os.path.splitext(slide_name)[0]


def _build_filename_candidates(slide_id):
    slide_id = str(slide_id).strip()
    lower_id = slide_id.lower()
    if any(lower_id.endswith(suffix) for suffix in SUPPORTED_WSI_SUFFIXES):
        return [slide_id]
    return [slide_id + suffix for suffix in SUPPORTED_WSI_SUFFIXES]


def resolve_wsi_path(source_root, slide_id):
    filename_candidates = _build_filename_candidates(slide_id)
    for subdir in WSI_SEARCH_SUBDIRS:
        search_root = source_root if subdir == '' else os.path.join(source_root, subdir)
        for filename in filename_candidates:
            path = os.path.join(search_root, filename)
            if os.path.isfile(path):
                return path
    return None


def parse_existing_coords(slide_save_dir):
    existing_coords = set()
    if not os.path.isdir(slide_save_dir):
        return existing_coords

    for fn in os.listdir(slide_save_dir):
        if not fn.endswith('.png'):
            continue
        parts = fn.rsplit('_', 2)
        if len(parts) < 3:
            continue
        try:
            x = int(parts[-2])
            y = int(os.path.splitext(parts[-1])[0])
        except Exception:
            continue
        existing_coords.add((x, y))
    return existing_coords


def process_slide(slide_name, source_root, coords_dir, output_dir, patch_level, patch_size, skip_existing=False):
    """处理单个 WSI 文件，按 coords 在指定 level 直接裁剪 20x patches。"""
    base_slide_id = normalize_slide_id(slide_name)

    slide_path = resolve_wsi_path(source_root, slide_name)
    if slide_path is None:
        print(f"WSI文件不存在: {slide_name} in {source_root}")
        return

    patch_file = os.path.join(coords_dir, base_slide_id + '.h5')
    if not os.path.exists(patch_file):
        print(f"坐标文件不存在: {patch_file}")
        return

    slide_save_dir = os.path.join(output_dir, base_slide_id)
    os.makedirs(slide_save_dir, exist_ok=True)

    with h5py.File(patch_file, 'r') as f:
        coords = f['coords'][:]

    print(f"处理 {base_slide_id}: {len(coords)} patches")

    existing_coords = parse_existing_coords(slide_save_dir) if skip_existing else set()
    coords_total = len(coords)
    if skip_existing:
        if len(existing_coords) > coords_total:
            print(f"裁剪异常: {base_slide_id} 已存在 {len(existing_coords)} 个 patches，但 coords 文件只有 {coords_total} 个。")
            print("请检查输出目录是否包含异常文件名或重复的 patch。脚本将继续裁剪缺失的 patches。")
        elif len(existing_coords) == coords_total:
            print(f"已跳过 {base_slide_id}（已存在 {len(existing_coords)} 个 patches，裁剪已正常完成）")
            return
        else:
            print(f"继续裁剪 {base_slide_id}，已存在 {len(existing_coords)} 个 patches，剩余 {coords_total - len(existing_coords)} 个将被生成。")

    wsi = WholeSlideImage(slide_path)

    for i, coord in enumerate(coords):
        x, y = int(coord[0]), int(coord[1])
        if skip_existing and (x, y) in existing_coords:
            continue

        patch = wsi.wsi.read_region((x, y), int(patch_level), (int(patch_size), int(patch_size))).convert('RGB')
        patch_name = f"{base_slide_id}_{x}_{y}.png"
        patch_path = os.path.join(slide_save_dir, patch_name)
        patch.save(patch_path)

        if (i + 1) % 1000 == 0:
            print(f"{base_slide_id}: 已处理 {i + 1}/{len(coords)}")


def main():
    parser = argparse.ArgumentParser(description='从 create_patches_fp.py 输出的 coords 裁剪 20x patches')
    parser.add_argument('--source', required=True, help='WSI 根目录')
    parser.add_argument('--csv', required=True, help='包含 slide_id 列的 CSV')
    parser.add_argument('--coords-root', required=True, help='create_patches_fp.py 的 save_dir')
    parser.add_argument('--patch-size', type=int, default=256, help='coords 目录对应的 patch size，例如 256')
    parser.add_argument('--patch-level', type=int, default=1, help='read_region 使用的 WSI level')
    parser.add_argument('--output-root', default=None, help='输出目录（默认: coords_root/patch_images_<patch_size>）')
    parser.add_argument('--workers', type=int, default=1, help='并行 worker 数')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已经完全裁剪过的 slide（检测输出目录）')
    parser.add_argument('--limit', type=int, default=None, help='仅处理前 N 个 slide')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if 'slide_id' not in df.columns:
        raise ValueError(f"CSV missing required column: slide_id ({args.csv})")

    coords_dir = os.path.join(args.coords_root, f'patches_{args.patch_size}')
    output_dir = args.output_root or os.path.join(args.coords_root, f'patch_images_{args.patch_size}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"开始裁剪 20x patches (patch_size={args.patch_size}, level={args.patch_level})...")
    print(f"coords 目录: {coords_dir}")
    print(f"输出目录: {output_dir}")

    slide_ids = df['slide_id'].astype(str).tolist()
    if args.limit is not None:
        slide_ids = slide_ids[: args.limit]

    worker_fn = partial(
        process_slide,
        source_root=args.source,
        coords_dir=coords_dir,
        output_dir=output_dir,
        patch_level=args.patch_level,
        patch_size=args.patch_size,
        skip_existing=args.skip_existing,
    )

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            pool.map(worker_fn, slide_ids)
    else:
        for slide_name in slide_ids:
            worker_fn(slide_name)

    print("20x patches裁剪完成!")


if __name__ == "__main__":
    main()
