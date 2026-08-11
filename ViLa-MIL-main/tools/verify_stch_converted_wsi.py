#!/usr/bin/env python3

import argparse
from pathlib import Path

import openslide
import pandas as pd

SUPPORTED_SUFFIXES = ('.svs', '.tif', '.tiff', '.ome.tif', '.ome.tiff')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Verify whether converted Shantou WSI files are aligned with CSV and readable by OpenSlide.'
    )
    parser.add_argument(
        '--csv-path',
        default='ViLa-MIL-main/dataset_csv/all_data_shantou.csv',
        help='CSV containing slide_id column',
    )
    parser.add_argument(
        '--wsi-dir',
        default='ViLa-MIL-main/data/stch/wsi_converted',
        help='Directory containing converted WSI files',
    )
    parser.add_argument(
        '--output-csv',
        default='ViLa-MIL-main/eval_results/stch/stch_converted_wsi_validation.csv',
        help='Validation report CSV path',
    )
    parser.add_argument(
        '--summary-txt',
        default='ViLa-MIL-main/eval_results/stch/stch_converted_wsi_validation_summary.txt',
        help='Summary text path',
    )
    return parser.parse_args()


def resolve_slide_path(wsi_dir: Path, slide_id: str):
    for suffix in SUPPORTED_SUFFIXES:
        path = wsi_dir / f'{slide_id}{suffix}'
        if path.is_file():
            return path
    return None


def safe_get(props, key):
    return props.get(key, '') if props else ''


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    wsi_dir = Path(args.wsi_dir)
    output_csv = Path(args.output_csv)
    summary_txt = Path(args.summary_txt)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if 'slide_id' not in df.columns:
        raise ValueError(f'CSV missing required column: slide_id ({csv_path})')

    slide_ids = df['slide_id'].astype(str).tolist()
    rows = []

    for slide_id in slide_ids:
        row = {
            'slide_id': slide_id,
            'status': 'missing',
            'path': '',
            'suffix': '',
            'level_count': '',
            'width': '',
            'height': '',
            'vendor': '',
            'objective': '',
            'error': '',
        }

        path = resolve_slide_path(wsi_dir, slide_id)
        if path is None:
            rows.append(row)
            continue

        row['path'] = str(path)
        row['suffix'] = ''.join(path.suffixes)

        try:
            slide = openslide.open_slide(str(path))
            row['status'] = 'ok'
            row['level_count'] = slide.level_count
            row['width'] = slide.dimensions[0]
            row['height'] = slide.dimensions[1]
            row['vendor'] = safe_get(slide.properties, 'openslide.vendor')
            row['objective'] = (
                safe_get(slide.properties, 'aperio.AppMag')
                or safe_get(slide.properties, 'openslide.objective-power')
            )
            slide.close()
        except Exception as exc:
            row['status'] = 'unreadable'
            row['error'] = str(exc)

        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)

    counts = result_df['status'].value_counts().to_dict()
    missing = counts.get('missing', 0)
    unreadable = counts.get('unreadable', 0)
    ok = counts.get('ok', 0)

    summary_lines = [
        '=' * 80,
        'Shantou Converted WSI Validation Summary',
        '=' * 80,
        f'CSV: {csv_path}',
        f'WSI dir: {wsi_dir}',
        f'Total slide_ids: {len(slide_ids)}',
        f'OK: {ok}',
        f'Missing: {missing}',
        f'Unreadable: {unreadable}',
        '',
    ]

    if missing:
        summary_lines.append('Missing slide_ids (first 20):')
        for slide_id in result_df.loc[result_df['status'] == 'missing', 'slide_id'].head(20):
            summary_lines.append(f'  - {slide_id}')
        summary_lines.append('')

    if unreadable:
        summary_lines.append('Unreadable slide_ids (first 20):')
        unreadable_rows = result_df[result_df['status'] == 'unreadable'].head(20)
        for _, item in unreadable_rows.iterrows():
            summary_lines.append(f"  - {item['slide_id']}: {item['error']}")
        summary_lines.append('')

    summary_txt.write_text('\n'.join(summary_lines), encoding='utf-8')

    print('\n'.join(summary_lines))
    print(f'Detailed report saved to: {output_csv}')
    print(f'Summary saved to: {summary_txt}')


if __name__ == '__main__':
    main()
