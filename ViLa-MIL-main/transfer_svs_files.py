#!/usr/bin/env python3
"""
SVS文件传输脚本
从CSV文件读取filename列，在源目录中递归查找对应的.svs文件，并复制到目标目录
"""

import os
import shutil
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm

def find_svs_file(filename, source_dir):
    """
    在源目录中递归查找对应的SVS文件
    
    Args:
        filename: 要查找的文件名（不含扩展名）
        source_dir: 源目录路径
    
    Returns:
        匹配的SVS文件路径，如果未找到返回None
    """
    # 使用glob进行递归搜索，支持多种可能的文件名格式
    patterns = [
        f"**/{filename}.svs",
        f"**/{filename}.SVS",
        f"**/{filename}*.svs",
        f"**/{filename}*.SVS"
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(source_dir, pattern), recursive=True)
        if matches:
            # 优先返回完全匹配的文件
            exact_matches = [m for m in matches if os.path.basename(m).lower() == f"{filename}.svs"]
            if exact_matches:
                return exact_matches[0]
            # 如果没有完全匹配，返回第一个匹配项
            return matches[0]
    
    return None

def transfer_svs_files(csv_file, source_dir, target_dir):
    """
    主函数：读取CSV文件并传输对应的SVS文件
    
    Args:
        csv_file: CSV文件路径
        source_dir: 源目录路径
        target_dir: 目标目录路径
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file)
        print(f"成功读取CSV文件，共有 {len(df)} 条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 检查是否存在filename列
    if 'filename' not in df.columns:
        print("错误：CSV文件中没有找到'filename'列")
        return
    
    # 获取所有filename
    filenames = df['filename'].tolist()
    print(f"需要查找和传输 {len(filenames)} 个文件")
    
    # 统计变量
    found_count = 0
    transferred_count = 0
    not_found_files = []
    transfer_errors = []
    
    # 逐个查找和传输文件
    for filename in tqdm(filenames, desc="处理文件"):
        # 查找SVS文件
        svs_file_path = find_svs_file(filename, source_dir)
        
        if svs_file_path:
            found_count += 1
            print(f"找到文件: {filename} -> {svs_file_path}")
            
            # 构建目标文件路径
            target_file_path = os.path.join(target_dir, os.path.basename(svs_file_path))
            
            try:
                # 检查目标文件是否已存在
                if os.path.exists(target_file_path):
                    print(f"  目标文件已存在，跳过: {os.path.basename(svs_file_path)}")
                    transferred_count += 1
                else:
                    # 复制文件
                    shutil.copy2(svs_file_path, target_file_path)
                    print(f"  成功传输: {os.path.basename(svs_file_path)}")
                    transferred_count += 1
                    
            except Exception as e:
                print(f"  传输失败: {filename} - {e}")
                transfer_errors.append((filename, str(e)))
        else:
            print(f"未找到文件: {filename}")
            not_found_files.append(filename)
    
    # 打印统计结果
    print("\n" + "="*60)
    print("传输完成统计:")
    print(f"总文件数: {len(filenames)}")
    print(f"找到文件数: {found_count}")
    print(f"成功传输数: {transferred_count}")
    print(f"未找到文件数: {len(not_found_files)}")
    print(f"传输错误数: {len(transfer_errors)}")
    
    # 输出未找到的文件列表
    if not_found_files:
        print(f"\n未找到的文件 ({len(not_found_files)}个):")
        for filename in not_found_files:  # 显示所有未找到的文件
            print(f"  - {filename}")
        
        # 将未找到的文件名保存到文件
        not_found_file_path = os.path.join(os.path.dirname(csv_file), "not_found_files.txt")
        try:
            with open(not_found_file_path, 'w', encoding='utf-8') as f:
                for filename in not_found_files:
                    f.write(f"{filename}\n")
            print(f"\n未找到的文件名已保存到: {not_found_file_path}")
        except Exception as e:
            print(f"保存未找到文件列表失败: {e}")
    
    # 输出传输错误列表
    if transfer_errors:
        print(f"\n传输错误 ({len(transfer_errors)}个):")
        for filename, error in transfer_errors[:10]:  # 只显示前10个
            print(f"  - {filename}: {error}")
        if len(transfer_errors) > 10:
            print(f"  ... 还有 {len(transfer_errors) - 10} 个错误未列出")

def main():
    """主程序入口"""
    # 配置路径
    csv_file = r"d:\FenLei\ViLa-MIL-main\datasss.csv"
    source_dir = r"Z:\shared\medical\MedicalData\Adenocarcinoma"
    target_dir = r"D:\FenLei\ViLa-MIL-main\data\non_benign"
    
    print("SVS文件传输脚本")
    print("="*60)
    print(f"CSV文件: {csv_file}")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print("="*60)
    
    # 检查路径是否存在
    if not os.path.exists(csv_file):
        print(f"错误：CSV文件不存在: {csv_file}")
        return
    
    if not os.path.exists(source_dir):
        print(f"错误：源目录不存在: {source_dir}")
        return
    
    # 确认是否继续
    response = input("\n是否继续执行传输操作？(y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("操作已取消")
        return
    
    # 执行传输
    transfer_svs_files(csv_file, source_dir, target_dir)

if __name__ == "__main__":
    main()
