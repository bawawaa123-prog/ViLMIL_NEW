#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import os
from functools import partial

import h5py
import pandas as pd
from PIL import Image

from wsi_core.WholeSlideImage import WholeSlideImage

SUPPORTED_WSI_SUFFIXES = ('.svs', '.ome.tif', '.ome.tiff', '.tif', '.tiff')
WSI_SEARCH_SUBDIRS = ('', 'benign', 'non_benign')
SUPPORTED_COORD_SIZES = (256, 512, 1024, 2048, 4096)


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


def parse_existing_coords(slide_save_dir, expected_out_size):
    existing_coords = set()
    if not os.path.isdir(slide_save_dir):
        return existing_coords

    for fn in os.listdir(slide_save_dir):
        if not fn.endswith('.png'):
            continue
        parts = fn.rsplit('_', 3)
        if len(parts) < 4:
            continue
        try:
            out_size = int(parts[-3])
            x = int(parts[-2])
            y = int(os.path.splitext(parts[-1])[0])
        except Exception:
            continue
        if out_size == expected_out_size:
            existing_coords.add((x, y))
    return existing_coords


def process_slide_high(
    slide_name,
    source_root,
    coords_dir,
    output_dir,
    patch_level,
    coords_size,
    out_size=None,
    skip_existing=False,
):
    """High-scale branch: cut by coords_size and optionally resize to out_size."""
    base_slide_id = normalize_slide_id(slide_name)

    slide_path = resolve_wsi_path(source_root, slide_name)
    if slide_path is None:
        print(f"WSI file not found: {slide_name} in {source_root}")
        return

    patch_file = os.path.join(coords_dir, base_slide_id + '.h5')
    if not os.path.exists(patch_file):
        print(f"Coords file not found: {patch_file}")
        return

    slide_save_dir = os.path.join(output_dir, base_slide_id)
    os.makedirs(slide_save_dir, exist_ok=True)

    with h5py.File(patch_file, 'r') as f:
        coords = f['coords'][:]

    output_size = int(coords_size if out_size is None else out_size)
    existing_coords = parse_existing_coords(slide_save_dir, output_size) if skip_existing else set()

    if skip_existing and len(existing_coords) >= len(coords):
        print(f"Skip {base_slide_id}: existing={len(existing_coords)} >= coords={len(coords)}")
        return

    wsi = WholeSlideImage(slide_path)

    print(
        f"Process {base_slide_id}: coords={len(coords)} level={patch_level} "
        f"read={coords_size} out={output_size}"
    )

    saved_count = 0
    failed_count = 0

    for i, coord in enumerate(coords):
        src_x, src_y = int(coord[0]), int(coord[1])

        if skip_existing and (src_x, src_y) in existing_coords:
            continue

        try:
            patch = wsi.wsi.read_region((src_x, src_y), int(patch_level), (int(coords_size), int(coords_size))).convert('RGB')
            if int(coords_size) != output_size:
                patch = patch.resize((output_size, output_size), Image.BILINEAR)
        except Exception as e:
            failed_count += 1
            if failed_count <= 20:
                print(f"{base_slide_id}: skip patch ({src_x}, {src_y}), reason: {e}")
            elif failed_count == 21:
                print(f"{base_slide_id}: too many failures (>20), suppressing detailed logs")
            continue

        # Keep source coord in filename so low/high patches are easy to align.
        patch_name = f"{base_slide_id}_{output_size}_{src_x}_{src_y}.png"
        patch_path = os.path.join(slide_save_dir, patch_name)
        patch.save(patch_path)
        saved_count += 1

        if (i + 1) % 1000 == 0:
            print(f"{base_slide_id}: processed {i + 1}/{len(coords)}")

    print(
        f"{base_slide_id}: done, saved={saved_count}, failed={failed_count}, total_coords={len(coords)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Jiangxi high-scale patch generation: read level-0 windows and resize to 256'
    )
    parser.add_argument('--source', default='data/jiangxi/wsi', help='WSI root')
    parser.add_argument('--csv', required=True, help='CSV containing slide_id column')
    parser.add_argument('--coords-root', required=True, help='create_patches output root')
    parser.add_argument('--coords-size', type=int, choices=SUPPORTED_COORD_SIZES, required=True,
                        help='coords dir suffix: patches_<size>')

    parser.add_argument('--patch-level', type=int, default=0, help='WSI level for read_region')
    parser.add_argument('--out-size', type=int, default=None, help='optional output patch size (default: coords-size)')

    parser.add_argument('--output-root', default=None, help='output dir')
    parser.add_argument('--workers', type=int, default=1, help='parallel workers')
    parser.add_argument('--skip-existing', action='store_true', help='skip patches already saved')
    parser.add_argument('--limit', type=int, default=None, help='process first N slides only')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if 'slide_id' not in df.columns:
        raise ValueError(f"CSV missing required column: slide_id ({args.csv})")

    slide_ids = df['slide_id'].astype(str).tolist()
    if args.limit is not None:
        slide_ids = slide_ids[: args.limit]

    coords_dir = os.path.join(args.coords_root, f'patches_{args.coords_size}')

    if args.output_root is None:
        output_dir = os.path.join(args.coords_root, f'patch_images_high_direct_{args.coords_size}')
    else:
        output_dir = args.output_root
    os.makedirs(output_dir, exist_ok=True)

    print('Start high-scale patch generation')
    print(f"coords_dir: {coords_dir}")
    print(f"output_dir: {output_dir}")

    worker_fn = partial(
        process_slide_high,
        source_root=args.source,
        coords_dir=coords_dir,
        output_dir=output_dir,
        patch_level=args.patch_level,
        coords_size=args.coords_size,
        out_size=args.out_size,
        skip_existing=args.skip_existing,
    )

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            pool.map(worker_fn, slide_ids)
    else:
        for sid in slide_ids:
            worker_fn(sid)

    print('High-scale patch generation finished')


if __name__ == '__main__':
    main()
