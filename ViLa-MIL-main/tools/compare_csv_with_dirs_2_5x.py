#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较 CSV (slide_id) 与指定目录下的目录名

用法: python compare_csv_with_dirs_2_5x.py

说明:
- 从 CSV 中读取 `slide_id` 列
- 从给定目录读取一级子目录名称（不递归）
- 比较三类: common, missing_in_dirs (CSV但目录中没有), extra_in_dirs (目录中有但CSV没有)
- 将结果输出到控制台，并保存为 CSV 与汇总 TXT

默认路径 (可以修改):
- CSV: D:\FenLei\ViLa-MIL-main\dataset_csv\all_data.csv
- 目录: Z:\ljh\MsaMIL_Net_Data\results\patches_2.5x
- 输出 CSV: D:\FenLei\ViLa-MIL-main\dataset_csv\comparison_results_2_5x.csv
"""

import os
from pathlib import Path
import pandas as pd
import argparse


def compare_csv_with_dirs(csv_path, dirs_root, output_path=None):
    """
    对比 CSV 中的 slide_id 与目录下的一级子目录名
    :param csv_path: CSV 文件路径，包含 slide_id 列
    :param dirs_root: 需要比对的目录的根路径（子目录名将与 slide_id 对比）
    :param output_path: 可选：保存比对详细结果的 CSV 文件路径
    :return: 字典，包含 common, missing_in_dirs, extra_in_dirs, 以及计数
    """
    print('='*80)
    print('比对 CSV 与目录名 (2.5x patches)')
    print('='*80)
    print(f'CSV: {csv_path}')
    print(f'目录根: {dirs_root}')

    # 1. 读取 CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'slide_id' not in df.columns:
        raise ValueError('CSV 文件中没有 slide_id 列')

    csv_ids = set(df['slide_id'].astype(str).tolist())
    print(f'CSV 中的 slide_id 数量: {len(csv_ids)}')

    # 2. 读取目录下的子目录名称
    dirs_root_path = Path(dirs_root)
    if not dirs_root_path.exists():
        raise FileNotFoundError(f'目录不存在: {dirs_root}')

    # 仅获取一级子目录名称（目录或文件夹名）
    child_entries = [p for p in dirs_root_path.iterdir() if p.is_dir() or p.is_file()]
    # We want directory names; some datasets store files directly under directory name (if they are files with slide ID as name too), but per user request, treat directory names
    # If there are files with .h5 in this directory, we will use stem names of files too
    dir_names = set()
    for p in child_entries:
        if p.is_dir():
            dir_names.add(p.name)
        else:
            # If it's a file (e.g., .h5), take the stem as slide id; include file extension removal
            dir_names.add(p.stem)

    print(f'找到的目录/文件条目数量: {len(dir_names)}')

    # 3. 比对
    common = sorted(csv_ids & dir_names)
    missing_in_dirs = sorted(csv_ids - dir_names)
    extra_in_dirs = sorted(dir_names - csv_ids)

    # 4. 输出结果
    print('\n' + '='*80)
    print('比对结果:')
    print('='*80)
    print(f'✅ 完全匹配数量: {len(common)}')
    print(f'❌ CSV 中有但目录中缺少数量: {len(missing_in_dirs)}')
    print(f'⚠️  目录中有但 CSV 中缺少数量: {len(extra_in_dirs)}')

    if missing_in_dirs:
        print('\nCSV 中有但目录中缺少 (前 20):')
        for s in missing_in_dirs[:20]:
            print('  -', s)
    if extra_in_dirs:
        print('\n目录中有但 CSV 中缺少 (前 20):')
        for s in extra_in_dirs[:20]:
            print('  +', s)

    results = {
        'common': common,
        'missing_in_dirs': missing_in_dirs,
        'extra_in_dirs': extra_in_dirs,
        'csv_total': len(csv_ids),
        'dirs_total': len(dir_names),
        'common_total': len(common)
    }

    # 5. 保存结果
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        max_len = max(len(common), len(missing_in_dirs), len(extra_in_dirs))
        data = {
            'common': common + ['']*(max_len - len(common)),
            'missing_in_dirs': missing_in_dirs + ['']*(max_len - len(missing_in_dirs)),
            'extra_in_dirs': extra_in_dirs + ['']*(max_len - len(extra_in_dirs))
        }
        result_df = pd.DataFrame(data)
        result_df.to_csv(output_path, index=False)

        # save summary
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('='*80 + '\n')
            f.write('CSV 与目录名比对 (2.5x patches) 结果\n')
            f.write('='*80 + '\n')
            f.write(f'CSV: {csv_path}\n')
            f.write(f'目录根: {dirs_root}\n\n')
            f.write(f'CSV 中的 slide_id 总数: {len(csv_ids)}\n')
            f.write(f'目录中条目总数: {len(dir_names)}\n')
            f.write(f'完全匹配数量: {len(common)}\n')
            f.write(f'CSV 中有但目录中缺少数量: {len(missing_in_dirs)}\n')
            f.write(f'目录中有但 CSV 中缺少数量: {len(extra_in_dirs)}\n')
            if missing_in_dirs:
                f.write('\nCSV 中有但目录中缺少列表:\n')
                for s in missing_in_dirs:
                    f.write('  - ' + s + '\n')
            if extra_in_dirs:
                f.write('\n目录中有但 CSV 中缺少列表:\n')
                for s in extra_in_dirs:
                    f.write('  + ' + s + '\n')

        print('\n✅ 比对详细结果已保存到:', output_path)
        print('✅ 汇总已保存到:', summary_path)

    print('\n' + '='*80 + '\n')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare CSV slide_ids with directory names under a given path (2.5x patches)')
    parser.add_argument('--csv_path', type=str, default=r'D:\FenLei\ViLa-MIL-main\dataset_csv\all_data.csv', help='CSV file path')
    parser.add_argument('--dirs_root', type=str, default=r'Z:\ljh\MsaMIL_Net_Data\results\patches_vila_5x', help='Root directory containing subdirectories to compare (2.5x)')
    parser.add_argument('--output_path', type=str, default=r'D:\FenLei\ViLa-MIL-main\dataset_csv\comparison_results_5x.csv', help='CSV file to save results')
    args = parser.parse_args()

    try:
        compare_csv_with_dirs(args.csv_path, args.dirs_root, args.output_path)
    except Exception as e:
        print('Error:', e)
