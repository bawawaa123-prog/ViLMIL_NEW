#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比对 CSV 文件中的 slide_id 与 h5 文件目录
检查哪些 slide_id 缺少对应的 h5 文件，以及哪些 h5 文件没有对应的 slide_id
"""

import os
import pandas as pd
from pathlib import Path


def compare_slides_with_h5_files(csv_path, h5_dir, output_path=None):
    """
    比对 CSV 文件中的 slide_id 与 h5 文件目录
    
    参数:
        csv_path: CSV 文件路径
        h5_dir: h5 文件所在目录
        output_path: 结果保存路径（可选）
    
    返回:
        dict: 包含比对结果的字典
    """
    print(f"{'='*80}")
    print(f"开始比对 CSV 与 H5 文件")
    print(f"{'='*80}")
    print(f"CSV 文件: {csv_path}")
    print(f"H5 目录:  {h5_dir}")
    print()
    
    # 1. 读取 CSV 文件中的 slide_id
    df = pd.read_csv(csv_path)
    csv_slide_ids = set(df['slide_id'].astype(str).tolist())
    print(f"CSV 文件中的 slide_id 数量: {len(csv_slide_ids)}")
    
    # 2. 读取 h5 目录中的文件
    h5_dir_path = Path(h5_dir)
    if not h5_dir_path.exists():
        print(f"错误: 目录不存在 - {h5_dir}")
        return None
    
    h5_files = list(h5_dir_path.glob("*.h5"))
    h5_slide_ids = set([f.stem for f in h5_files])  # 去掉 .h5 后缀
    print(f"H5 目录中的文件数量:   {len(h5_slide_ids)}")
    print()
    
    # 3. 比对差异
    # CSV 中有，但 H5 目录中缺少的
    missing_in_h5 = sorted(csv_slide_ids - h5_slide_ids)
    
    # H5 目录中有，但 CSV 中没有的
    extra_in_h5 = sorted(h5_slide_ids - csv_slide_ids)
    
    # 两者都有的（完全匹配）
    common_slides = sorted(csv_slide_ids & h5_slide_ids)
    
    # 4. 打印结果
    print(f"{'='*80}")
    print(f"比对结果汇总")
    print(f"{'='*80}")
    print(f"✅ 完全匹配的 slide_id 数量: {len(common_slides)}")
    print(f"❌ CSV 中有但 H5 缺少的数量:  {len(missing_in_h5)}")
    print(f"⚠️  H5 中有但 CSV 缺少的数量:  {len(extra_in_h5)}")
    print()
    
    # 5. 详细列出缺少的和多余的
    if missing_in_h5:
        print(f"{'='*80}")
        print(f"❌ CSV 中有但 H5 目录缺少的 slide_id ({len(missing_in_h5)} 个):")
        print(f"{'='*80}")
        for slide_id in missing_in_h5:
            print(f"  - {slide_id}")
        print()
    
    if extra_in_h5:
        print(f"{'='*80}")
        print(f"⚠️  H5 目录中有但 CSV 缺少的 slide_id ({len(extra_in_h5)} 个):")
        print(f"{'='*80}")
        for slide_id in extra_in_h5:
            print(f"  + {slide_id}")
        print()
    
    if not missing_in_h5 and not extra_in_h5:
        print(f"✅ 完美匹配! CSV 与 H5 目录完全一致，没有缺失或多余。")
        print()
    
    # 6. 保存结果到文件
    results = {
        'common_slides': common_slides,
        'missing_in_h5': missing_in_h5,
        'extra_in_h5': extra_in_h5,
        'csv_total': len(csv_slide_ids),
        'h5_total': len(h5_slide_ids),
        'common_total': len(common_slides)
    }
    
    if output_path:
        # 保存详细结果到 CSV
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 创建结果 DataFrame
        max_len = max(len(common_slides), len(missing_in_h5), len(extra_in_h5))
        
        result_df = pd.DataFrame({
            'common_slides': common_slides + [''] * (max_len - len(common_slides)),
            'missing_in_h5': missing_in_h5 + [''] * (max_len - len(missing_in_h5)),
            'extra_in_h5': extra_in_h5 + [''] * (max_len - len(extra_in_h5))
        })
        
        result_df.to_csv(output_path, index=False)
        print(f"✅ 详细结果已保存到: {output_path}")
        
        # 保存汇总统计
        summary_path = output_path.replace('.csv', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"CSV 与 H5 文件比对结果汇总\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"CSV 文件: {csv_path}\n")
            f.write(f"H5 目录:  {h5_dir}\n\n")
            f.write(f"CSV 中的 slide_id 总数:        {len(csv_slide_ids)}\n")
            f.write(f"H5 目录中的文件总数:           {len(h5_slide_ids)}\n")
            f.write(f"完全匹配的 slide_id 数量:      {len(common_slides)}\n")
            f.write(f"CSV 中有但 H5 缺少的数量:      {len(missing_in_h5)}\n")
            f.write(f"H5 中有但 CSV 缺少的数量:      {len(extra_in_h5)}\n\n")
            
            if missing_in_h5:
                f.write(f"{'='*80}\n")
                f.write(f"CSV 中有但 H5 目录缺少的 slide_id:\n")
                f.write(f"{'='*80}\n")
                for slide_id in missing_in_h5:
                    f.write(f"  - {slide_id}\n")
                f.write("\n")
            
            if extra_in_h5:
                f.write(f"{'='*80}\n")
                f.write(f"H5 目录中有但 CSV 缺少的 slide_id:\n")
                f.write(f"{'='*80}\n")
                for slide_id in extra_in_h5:
                    f.write(f"  + {slide_id}\n")
                f.write("\n")
        
        print(f"✅ 汇总统计已保存到: {summary_path}")
    
    print(f"\n{'='*80}")
    print(f"比对完成!")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    # 配置路径
    csv_path = r"D:\FenLei\ViLa-MIL-main\dataset_csv\all_data.csv"
    h5_dir = r"D:\FenLei\ViLa-MIL-main\patches_coords_5x\patches_256"
    output_path = r"D:\FenLei\ViLa-MIL-main\dataset_csv\comparison_results.csv"
    
    # 执行比对
    results = compare_slides_with_h5_files(csv_path, h5_dir, output_path)
    
    # 可选: 返回结果供后续处理
    if results:
        print(f"\n可通过以下方式访问结果:")
        print(f"  - results['common_slides']:  完全匹配的 slide_id 列表")
        print(f"  - results['missing_in_h5']:  CSV 中有但 H5 缺少的列表")
        print(f"  - results['extra_in_h5']:    H5 中有但 CSV 缺少的列表")
