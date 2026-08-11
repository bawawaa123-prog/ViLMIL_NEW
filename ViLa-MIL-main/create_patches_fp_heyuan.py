#!/usr/bin/env python3

import argparse
import os
import tempfile

import pandas as pd

from create_patches_fp import seg_and_patch

SUPPORTED_COORD_SIZES = (256, 512, 1024, 2048)


def build_uuid_name_file(input_csv, output_csv):
    """Build temporary no-header csv: case_id, slide_id, label."""
    df = pd.read_csv(input_csv)
    if 'slide_id' not in df.columns:
        raise ValueError(f"CSV missing required column: slide_id ({input_csv})")

    if 'case_id' in df.columns:
        case_ids = df['case_id'].astype(str)
    else:
        case_ids = df['slide_id'].astype(str)

    if 'label' in df.columns:
        labels = df['label'].astype(str)
    else:
        labels = pd.Series(['Unknown'] * len(df))

    out_df = pd.DataFrame({
        0: case_ids,
        1: df['slide_id'].astype(str),
        2: labels,
    })
    out_df.to_csv(output_csv, header=False, index=False)


def load_params_from_preset(preset_name):
    seg_params = {
        'seg_level': -1,
        'sthresh': 8,
        'mthresh': 7,
        'close': 4,
        'use_otsu': False,
        'keep_ids': 'none',
        'exclude_ids': 'none',
    }
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset_name:
        preset_path = os.path.join('presets', preset_name)
        preset_df = pd.read_csv(preset_path)

        for key in seg_params:
            seg_params[key] = preset_df.loc[0, key]
        for key in filter_params:
            filter_params[key] = preset_df.loc[0, key]
        for key in vis_params:
            vis_params[key] = preset_df.loc[0, key]
        for key in patch_params:
            patch_params[key] = preset_df.loc[0, key]

    return seg_params, filter_params, vis_params, patch_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='Heyuan external queue: create tissue masks + coords h5 from SVS'
    )
    parser.add_argument('--source', default='data/heyuan/wsi', help='WSI root directory')
    parser.add_argument('--csv', required=True, help='CSV containing slide_id column')
    parser.add_argument('--save-dir', required=True, help='Output root directory')

    parser.add_argument('--coord-size', type=int, choices=SUPPORTED_COORD_SIZES, default=None,
                        help='Coordinate size for h5 generation (256/512/1024/2048)')
    parser.add_argument('--patch-size', type=int, choices=SUPPORTED_COORD_SIZES, default=256,
                        help='Backward-compatible alias of --coord-size')
    parser.add_argument('--step-size', type=int, default=None,
                        help='Coordinate step size (default: same as coord-size)')
    parser.add_argument('--patch-level', type=int, default=2, help='WSI level used for coords')

    parser.add_argument('--preset', default='heyuan.csv', help='Preset csv under presets/')
    parser.add_argument('--process-list', default=None, help='Optional process list csv path')

    parser.add_argument('--seg', dest='seg', action='store_true', help='Run segmentation (default: on)')
    parser.add_argument('--no-seg', dest='seg', action='store_false', help='Disable segmentation')
    parser.set_defaults(seg=True)

    parser.add_argument('--patch', dest='patch', action='store_true', help='Run coords generation (default: on)')
    parser.add_argument('--no-patch', dest='patch', action='store_false', help='Disable coords generation')
    parser.set_defaults(patch=True)

    parser.add_argument('--stitch', action='store_true', help='Generate graph heatmap images')
    parser.add_argument('--disable-auto-skip', action='store_true', help='Do not skip existing h5')

    return parser.parse_args()


def main():
    args = parse_args()

    coord_size = args.coord_size if args.coord_size is not None else args.patch_size
    step_size = args.step_size if args.step_size is not None else coord_size

    patch_save_dir = os.path.join(args.save_dir, f'patches_{coord_size}')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    only_mask_save_dir = os.path.join(args.save_dir, 'only_masks')
    stitch_save_dir = os.path.join(args.save_dir, f'graph_{coord_size}')

    directories = {
        'source': args.source,
        'save_dir': args.save_dir,
        'patch_save_dir': patch_save_dir,
        'mask_save_dir': mask_save_dir,
        'only_mask_save_dir': only_mask_save_dir,
        'stitch_save_dir': stitch_save_dir,
    }

    for key, value in directories.items():
        print(f"{key}: {value}")
        if key != 'source':
            os.makedirs(value, exist_ok=True)

    print(f"coord_size: {coord_size}")
    print(f"step_size: {step_size}")

    seg_params, filter_params, vis_params, patch_params = load_params_from_preset(args.preset)

    if args.process_list is not None:
        process_list = os.path.join(args.save_dir, args.process_list)
    else:
        process_list = None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            tmp_path = tmp.name

        build_uuid_name_file(args.csv, tmp_path)

        seg_times, patch_times = seg_and_patch(
            **directories,
            slide_name_file=None,
            patch_size=coord_size,
            step_size=step_size,
            seg_params=seg_params,
            filter_params=filter_params,
            vis_params=vis_params,
            patch_params=patch_params,
            patch_level=args.patch_level,
            use_default_params=False,
            seg=args.seg,
            save_mask=True,
            stitch=args.stitch,
            patch=args.patch,
            auto_skip=(not args.disable_auto_skip),
            process_list=process_list,
            uuid_name_file=tmp_path,
        )
        print(f"Done. avg_seg_time={seg_times:.4f}s avg_patch_time={patch_times:.4f}s")
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            os.remove(tmp_path)


if __name__ == '__main__':
    main()
