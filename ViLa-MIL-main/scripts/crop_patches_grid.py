#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按网格从 WSI 裁剪 PNG patches（支持 5x/20x），无需坐标 h5。
- 输入：
  --slide_root ：.svs 根目录
  --csv        ：包含 slide_id 列的 CSV（如 dataset_csv/dataset.csv）
  --out_root   ：输出根目录（将创建子目录 slide_id 并写入 PNG）
  --level      ：OpenSlide 读取层级（1≈20x，2≈5x）
  --patch_size ：patch 边长（默认 256）
  --step_size  ：步长（默认 256；与 patch_size 相同即无重叠）
  --skip_white ：跳过近白背景（默认开启）
  --white_thr  ：[0-255]，近白阈值（默认 240）
- 输出：
  out_root/slide_id/*.png，文件名为 slide_x_y.png，x/y 为 level-0 坐标

用法（cmd）：
  python scripts\crop_patches_grid.py ^
    --slide_root "D:\\FenLei\\data" ^
    --csv "D:\\FenLei\\ViLa-MIL-main\\dataset_csv\\dataset.csv" ^
    --out_root "Z:\\ljh\\data\\results\\new\\patches_5x" ^
    --level 2 ^
    --patch_size 256 --step_size 256 --skip_white --white_thr 240

依赖：openslide-python, pandas, numpy, pillow, tqdm
注意：网格裁剪可能量很大，建议先小范围试跑，或开启 --skip_white 以减少背景。
"""

import os
import argparse
import pandas as pd
import openslide
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np


def is_white_patch(img: Image.Image, thr: int = 240, ratio_thr: float = 0.8) -> bool:
    # 快速近白判定：将图像缩小取均值，并计算高亮像素占比
    arr = np.asarray(img.resize((32, 32))).astype(np.uint8)
    if arr.ndim == 3:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr
    bright = (gray >= thr).mean()
    return bright >= ratio_thr


def process_one_slide(slide_id: str, slide_root: str, out_root: str, level: int,
                      patch_size: int, step_size: int, suffix: str, skip_white: bool, white_thr: int):
    slide_path = os.path.join(slide_root, slide_id + suffix)
    if not os.path.isfile(slide_path):
        return (slide_id, 0, f'slide-not-found:{slide_path}')

    out_dir = os.path.join(out_root, slide_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        return (slide_id, 0, f'openslide-error:{e}')

    # level-0 尺寸和下采样因子
    w0, h0 = slide.dimensions
    ds = slide.level_downsamples[level]
    if isinstance(ds, float):
        ds = int(round(ds))
    ds = max(1, int(ds))

    step0 = step_size * ds
    patch0 = patch_size * ds

    count = 0
    ys = range(0, max(0, h0 - patch0 + 1), step0)
    xs = range(0, max(0, w0 - patch0 + 1), step0)

    for y0 in ys:
        for x0 in xs:
            try:
                img = slide.read_region((x0, y0), level, (patch_size, patch_size)).convert('RGB')
                if skip_white and is_white_patch(img, thr=white_thr):
                    continue
                fn = f"{slide_id}_{x0}_{y0}.png"
                img.save(os.path.join(out_dir, fn))
                count += 1
            except Exception:
                continue

    slide.close()
    return (slide_id, count, 'ok')


def main():
    parser = argparse.ArgumentParser(description='Crop PNG patches from WSI by grid')
    parser.add_argument('--slide_root', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--level', type=int, required=True)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--step_size', type=int, default=256)
    parser.add_argument('--suffix', type=str, default='.svs')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--skip_white', action='store_true', default=False)
    parser.add_argument('--white_thr', type=int, default=240)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if 'slide_id' not in df.columns:
        print(f"CSV 缺少 slide_id 列: {args.csv}")
        return 1

    slide_ids = df['slide_id'].dropna().astype(str).tolist()
    if not slide_ids:
        print('CSV 中未找到有效 slide_id')
        return 1

    os.makedirs(args.out_root, exist_ok=True)
    print(f"待处理 {len(slide_ids)} 个 slides -> {args.out_root}; level={args.level}, patch={args.patch_size}, step={args.step_size}")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for sid in slide_ids:
            futures.append(ex.submit(process_one_slide, sid, args.slide_root, args.out_root, args.level,
                                     args.patch_size, args.step_size, args.suffix, args.skip_white, args.white_thr))
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
