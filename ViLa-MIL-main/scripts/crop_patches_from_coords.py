#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从坐标 h5 (coords) 批量裁剪 PNG patch。
- 输入：
  1) --slide_root: 存放 .svs 的根目录，例如 D:\\FenLei\\data
  2) --coords_root: 坐标目录的根，例如 Z:\\ljh\\data\\results\\new\\coords_5x
     该目录下应有子目录 'patches_256'，里面是 {slide_id}.h5，含 'coords' 数据集。
  3) --out_root: 输出 PNG 的根目录，例如 Z:\\ljh\\data\\results\\new\\patches_5x
  4) --level: OpenSlide 读取层级，1=约20x，2=约5x（与坐标对应）
- 输出：
  out_root/slide_id/*.png

用法（cmd）：
  python scripts\crop_patches_from_coords.py ^
    --slide_root "D:\\FenLei\\data" ^
    --coords_root "Z:\\ljh\\data\\results\\new\\coords_5x" ^
    --out_root "Z:\\ljh\\data\\results\\new\\patches_5x" ^
    --level 2

依赖：openslide-python, h5py, pandas, numpy, pillow, tqdm
"""

import os
import argparse
import h5py
import openslide
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image


def process_one_slide(slide_id: str, slide_root: str, coords_h5: str, out_root: str, level: int, patch_size: int = 256, suffix: str = '.svs'):
    slide_path = os.path.join(slide_root, slide_id + suffix)
    if not os.path.isfile(slide_path):
        return (slide_id, 0, f'slide-not-found:{slide_path}')

    out_dir = os.path.join(out_root, slide_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        return (slide_id, 0, f'openslide-error:{e}')

    try:
        with h5py.File(coords_h5, 'r') as f:
            if 'coords' not in f:
                return (slide_id, 0, 'coords-missing')
            coords = f['coords'][:]
    except Exception as e:
        return (slide_id, 0, f'h5-error:{e}')

    count = 0
    for xy in coords:
        x, y = int(xy[0]), int(xy[1])
        try:
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert('RGB')
            fn = f"{slide_id}_{x}_{y}.png"
            patch.save(os.path.join(out_dir, fn))
            count += 1
        except Exception as e:
            # 单个坐标失败，跳过
            continue

    slide.close()
    return (slide_id, count, 'ok')


def main():
    parser = argparse.ArgumentParser(description='Crop PNG patches from coords h5')
    parser.add_argument('--slide_root', type=str, required=True, help='根目录，存放 .svs')
    parser.add_argument('--coords_root', type=str, required=True, help='根目录，内含 patches_256/{slide_id}.h5')
    parser.add_argument('--out_root', type=str, required=True, help='输出 PNG 根目录')
    parser.add_argument('--level', type=int, required=True, help='OpenSlide 层级：1=约20x，2=约5x')
    parser.add_argument('--suffix', type=str, default='.svs', help='WSI 文件后缀，默认 .svs')
    parser.add_argument('--workers', type=int, default=4, help='并行裁剪线程数')
    args = parser.parse_args()

    coords_dir = os.path.join(args.coords_root, 'patches_256')
    if not os.path.isdir(coords_dir):
        print(f"坐标目录不存在或缺少 patches_256: {coords_dir}")
        return 1

    h5_files = [f for f in os.listdir(coords_dir) if f.lower().endswith('.h5')]
    if not h5_files:
        print(f"在 {coords_dir} 下未发现 .h5 坐标文件")
        return 1

    os.makedirs(args.out_root, exist_ok=True)

    print(f"将处理 {len(h5_files)} 个 slide：coords_dir={coords_dir} level={args.level}")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for fn in h5_files:
            slide_id = os.path.splitext(fn)[0]
            coords_h5 = os.path.join(coords_dir, fn)
            futures.append(ex.submit(process_one_slide, slide_id, args.slide_root, coords_h5, args.out_root, args.level, 256, args.suffix))
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Cropping slides', ncols=80):
            results.append(fut.result())

    ok = sum(1 for sid, n, s in results if s == 'ok')
    total_patches = sum(n for _, n, _ in results)
    failed = [r for r in results if r[2] != 'ok']

    print(f"完成。成功 {ok}/{len(results)} slides，总计裁剪 {total_patches} 个 patch。")
    if failed:
        print("以下 slide 失败或跳过：")
        for sid, n, s in failed[:20]:
            print(f" - {sid}: {s}")
        if len(failed) > 20:
            print(f" ... 以及 {len(failed)-20} 个未列出")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
