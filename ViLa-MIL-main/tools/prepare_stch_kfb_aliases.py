#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create .kfb aliases for Shantou .image files without duplicating data by default.'
    )
    parser.add_argument(
        '--csv-path',
        default='ViLa-MIL-main/dataset_csv/all_data_shantou.csv',
        help='CSV containing slide_id column',
    )
    parser.add_argument(
        '--source-dir',
        default='ViLa-MIL-main/data/stch/wsi',
        help='Directory containing original .image files',
    )
    parser.add_argument(
        '--output-dir',
        default='ViLa-MIL-main/data/stch/wsi_kfb_alias',
        help='Directory to place .kfb aliases',
    )
    parser.add_argument(
        '--mode',
        choices=('hardlink', 'symlink', 'copy'),
        default='hardlink',
        help='Alias creation mode; hardlink is preferred when source and output are on the same filesystem',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing destination files',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Only process the first N slide_ids for smoke testing',
    )
    return parser.parse_args()


def ensure_removed(path: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()


def create_alias(source: Path, destination: Path, mode: str):
    if mode == 'hardlink':
        os.link(source, destination)
    elif mode == 'symlink':
        os.symlink(source.resolve(), destination)
    else:
        shutil.copy2(source, destination)


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if 'slide_id' not in df.columns:
        raise ValueError(f'CSV missing required column: slide_id ({csv_path})')

    slide_ids = df['slide_id'].astype(str).tolist()
    if args.limit is not None:
        slide_ids = slide_ids[: args.limit]

    created = 0
    skipped = 0
    missing = []

    for slide_id in slide_ids:
        source = source_dir / f'{slide_id}.image'
        destination = output_dir / f'{slide_id}.kfb'

        if not source.is_file():
            missing.append(slide_id)
            continue

        if destination.exists() or destination.is_symlink():
            if args.overwrite:
                ensure_removed(destination)
            else:
                skipped += 1
                continue

        try:
            create_alias(source, destination, args.mode)
            created += 1
        except OSError as exc:
            # Hardlinks can fail across filesystems; suggest a deterministic fallback.
            if args.mode == 'hardlink':
                print(f'[Warn] hardlink failed for {slide_id}: {exc}. Retry with --mode symlink or --mode copy.')
            else:
                print(f'[Warn] alias creation failed for {slide_id}: {exc}')

    print('=' * 80)
    print('Shantou KFB Alias Preparation')
    print('=' * 80)
    print(f'CSV: {csv_path}')
    print(f'Source dir: {source_dir}')
    print(f'Output dir: {output_dir}')
    print(f'Mode: {args.mode}')
    print(f'Processed slide_ids: {len(slide_ids)}')
    print(f'Created: {created}')
    print(f'Skipped existing: {skipped}')
    print(f'Missing source files: {len(missing)}')

    if missing:
        print('Missing slide_ids (first 20):')
        for slide_id in missing[:20]:
            print(f'  - {slide_id}')

    print()
    print('Next step:')
    print('  Use the generated .kfb aliases with your vendor-specific KFB reader/exporter,')
    print('  then export full-resolution pyramidal .svs or .tif files into data/stch/wsi_converted.')


if __name__ == '__main__':
    main()
