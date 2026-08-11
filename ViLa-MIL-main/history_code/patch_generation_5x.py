#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from wsi_core.WholeSlideImage import WholeSlideImage
import h5py
from PIL import Image
import multiprocessing as mp
from functools import partial

def process_slide_5x(slide_name, slide_folder, root_folder, save_folder):
    """处理单个WSI文件，裁剪5x patches"""
    
    slide_path = os.path.join(slide_folder, slide_name + '.svs')
    if not os.path.exists(slide_path):
        print(f"WSI文件不存在: {slide_path}")
        return
    
    # 加载坐标文件
    patch_file = os.path.join(root_folder, 'patches_256', slide_name + '.h5')
    if not os.path.exists(patch_file):
        print(f"坐标文件不存在: {patch_file}")
        return
    
    # 创建保存目录
    slide_save_dir = os.path.join(save_folder, slide_name)
    os.makedirs(slide_save_dir, exist_ok=True)
    
    # 加载WSI
    wsi = WholeSlideImage(slide_path)
    
    # 读取坐标
    with h5py.File(patch_file, 'r') as f:
        coords = f['coords'][:]
    
    print(f"处理 {slide_name}: {len(coords)} patches")
    
    # 裁剪patches (patch_level=2对应5x)
    for i, coord in enumerate(coords):
        x, y = coord
        patch = wsi.wsi.read_region((x, y), 2, (256, 256)).convert('RGB')
        patch_name = f"{slide_name}_{x}_{y}.png"
        patch_path = os.path.join(slide_save_dir, patch_name)
        patch.save(patch_path)
        
        if (i + 1) % 1000 == 0:
            print(f"{slide_name}: 已处理 {i+1}/{len(coords)}")

def main():
    # 配置参数 - 5x低倍率
    slide_folder = 'D:/FenLei/data'
    all_data_path = 'D:/FenLei/ViLa-MIL-main/dataset_csv/dataset.csv'
    root_folder = 'D:/FenLei/ViLa-MIL-main/patches_coords_xin_5x'
    save_folder = 'Z:/ljh/data/results/new/patches_5x'
    
    # 读取数据
    df = pd.read_csv(all_data_path)
    
    # 创建保存目录
    os.makedirs(save_folder, exist_ok=True)
    
    print(f"开始裁剪5x patches...")
    print(f"输入坐标目录: {root_folder}")
    print(f"输出patch目录: {save_folder}")
    
    # 处理每个slide
    for _, row in df.iterrows():
        slide_name = row['slide_id']
        
        # 检查benign和non_benign目录
        slide_path_benign = os.path.join(slide_folder, 'benign', slide_name + '.svs')
        slide_path_non_benign = os.path.join(slide_folder, 'non_benign', slide_name + '.svs')
        
        if os.path.exists(slide_path_benign):
            slide_subfolder = os.path.join(slide_folder, 'benign')
        elif os.path.exists(slide_path_non_benign):
            slide_subfolder = os.path.join(slide_folder, 'non_benign')
        else:
            print(f"未找到WSI文件: {slide_name}")
            continue
            
        process_slide_5x(slide_name, slide_subfolder, root_folder, save_folder)
    
    print("5x patches裁剪完成!")

if __name__ == "__main__":
    main()
