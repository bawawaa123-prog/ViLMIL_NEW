#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成正确的注意力热力图
使用修复后的 forward_with_attention 方法

用法（cmd）：
  python regenerate_heatmaps.py ^
    --task task_adenocarcinoma_xin ^
    --data_root_dir "Z:\\ljh\\data\\results" ^
    --data_folder_s "features_5x" ^
    --data_folder_l "features_20x" ^
    --wsi_root "Z:\\ljh\\data" ^
    --checkpoint "results\\training\\adenocarcinoma_vila_mil_s1\\s_4_checkpoint.pt" ^
    --csv_path "dataset_csv\\all_data.csv" ^
    --results_dir "Z:\\ljh\\data\\eval_heatmap_results_correct" ^
    --text_prompt_path "text_prompt\\adenocarcinoma_dual_scale_prompt.csv" ^
    --heatmap_downsample 64 ^
    --heatmap_alpha 0.5
"""

from __future__ import print_function
import argparse
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import time

from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_ViLa_MIL import ViLa_MIL_Model
from utils.heatmap_utils import create_attention_heatmap, create_attention_heatmap_scatter, save_attention_to_h5
import ml_collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='重新生成正确的热力图')
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--data_root_dir', type=str, required=True)
parser.add_argument('--data_folder_s', type=str, required=True)
parser.add_argument('--data_folder_l', type=str, required=True)
parser.add_argument('--wsi_root', type=str, default='Z:/ljh/data', help='WSI文件根目录')
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--results_dir', type=str, default='./heatmaps_correct')
parser.add_argument('--text_prompt_path', type=str, required=True)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--heatmap_downsample', type=int, default=64)
parser.add_argument('--heatmap_alpha', type=float, default=0.5)
parser.add_argument('--heatmap_style', type=str, default='scatter', choices=['scatter', 'region'], 
                    help='热力图风格: scatter=散点图(推荐), region=区域填充')
parser.add_argument('--point_size', type=int, default=20, help='散点图模式下点的大小')
parser.add_argument('--max_slides', type=int, default=None, help='最多处理多少个slides(测试用)')

args = parser.parse_args()

# 任务配置
if args.task == 'task_adenocarcinoma_xin':
    args.n_classes = 2
    label_dict = {'Adenocarcinoma': 0, 'NonAdenocarcinoma': 1}
elif args.task == 'task_tcga_lung_subtyping':
    args.n_classes = 2
    label_dict = {'LUAD': 0, 'LUSC': 1}
elif args.task == 'task_tcga_rcc_subtyping':
    args.n_classes = 3
    label_dict = {'CCRCC': 0, 'PRCC': 1, 'CRCC': 2}
else:
    raise NotImplementedError(f'Unknown task: {args.task}')

print(f'\n{"="*60}')
print(f'重新生成热力图')
print(f'{"="*60}')
print(f'任务: {args.task}')
print(f'类别数: {args.n_classes}')
print(f'WSI根目录: {args.wsi_root}')
print(f'输出目录: {args.results_dir}')
print(f'热力图风格: {args.heatmap_style}')
if args.heatmap_style == 'scatter':
    print(f'散点大小: {args.point_size}')
print(f'{"="*60}\n')

# 创建输出目录
os.makedirs(args.results_dir, exist_ok=True)
heatmap_dir = os.path.join(args.results_dir, 'heatmaps')
attention_h5_dir = os.path.join(args.results_dir, 'attention_h5')
os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(attention_h5_dir, exist_ok=True)

# 加载数据集
print('加载数据集...')
data_dir_s = os.path.join(args.data_root_dir, args.data_folder_s)
data_dir_l = os.path.join(args.data_root_dir, args.data_folder_l)

dataset = Generic_MIL_Dataset(
    data_dir_s=data_dir_s,
    data_dir_l=data_dir_l,
    mode='transformer',
    csv_path=args.csv_path,
    shuffle=False,
    print_info=True,
    label_dict=label_dict,
    patient_strat=False,
    ignore=[]
)

if args.max_slides:
    print(f'⚠️ 仅处理前 {args.max_slides} 个 slides（测试模式）')

print(f'数据集大小: {len(dataset)} 个 slides')

# 创建模型配置
print('创建模型配置...')

# 读取文本提示 - 每个类别有两个描述(低倍+高倍)
text_prompt_df = pd.read_csv(args.text_prompt_path)
text_prompt = []
for _, row in text_prompt_df.iterrows():
    text_prompt.append(row['low_resolution_description'])   # 低倍描述
    text_prompt.append(row['high_resolution_description'])  # 高倍描述

print(f'文本提示数量: {len(text_prompt)} (每个类别2个描述)')

config = ml_collections.ConfigDict()
config.input_size = 1024
config.hidden_size = 192  # 必须与训练时的配置一致
config.text_prompt = text_prompt
config.prototype_number = 16  # 默认16个prototypes

# 加载模型
print('加载模型...')
model = ViLa_MIL_Model(config=config, num_classes=args.n_classes)
model.to(device)

# 加载训练好的权重
print(f'加载权重: {args.checkpoint}')
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint, strict=True)
model.eval()
print('模型加载完成!')

# 开始生成热力图
print(f'\n开始生成注意力热力图...')
start_time = time.time()

success_count = 0
failed_slides = []

# 使用 DataLoader 避免索引越界问题
from torch.utils.data import DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

with torch.no_grad():
    loader = tqdm(data_loader, desc='生成热力图', ncols=100, total=len(dataset))
    
    for idx, batch in enumerate(loader):
        if args.max_slides and idx >= args.max_slides:
            print(f'\n达到最大处理数量 {args.max_slides}，停止')
            break
        
        # 数据集返回: features_s, coords_s, features_l, coords_l, label, slide_id
        features_s, coords_s, features_l, coords_l, label, slide_id = batch
        
        # slide_id 是一个 tuple，取第一个元素
        if isinstance(slide_id, (tuple, list)):
            slide_id = slide_id[0]
        
        # 构建 WSI 文件路径
        wsi_path = os.path.join(args.wsi_root, slide_id + '.svs')
        
        if not os.path.isfile(wsi_path):
            failed_slides.append((slide_id, 'WSI文件不存在'))
            continue
        
        # 移动数据到设备
        features_s = features_s.to(device)
        label = torch.tensor(label).to(device)
        coords_s = coords_s.to(device)
        features_l = features_l.to(device)
        coords_l = coords_l.to(device)
        
        # 前向传播获取注意力权重
        try:
            logits, Y_prob, Y_hat, attention_s, attention_l = model.forward_with_attention(
                features_s, coords_s, features_l, coords_l, label
            )
        except Exception as e:
            failed_slides.append((slide_id, f'模型前向传播失败: {str(e)}'))
            continue
        
        # 转换为numpy
        attention_s_np = attention_s.cpu().numpy()
        attention_l_np = attention_l.cpu().numpy()
        coords_s_np = coords_s.cpu().numpy()
        coords_l_np = coords_l.cpu().numpy()
        
        # 去除batch维度
        if coords_s_np.ndim == 3:
            coords_s_np = coords_s_np[0]
        if coords_l_np.ndim == 3:
            coords_l_np = coords_l_np[0]
        
        # 检查维度匹配
        if len(attention_s_np) != len(coords_s_np):
            failed_slides.append((slide_id, f'5x维度不匹配: attn={len(attention_s_np)}, coords={len(coords_s_np)}'))
            continue
        if len(attention_l_np) != len(coords_l_np):
            failed_slides.append((slide_id, f'20x维度不匹配: attn={len(attention_l_np)}, coords={len(coords_l_np)}'))
            continue
        
        # 生成5x热力图
        heatmap_path_5x = os.path.join(heatmap_dir, f'{slide_id}_5x_heatmap.png')
        if args.heatmap_style == 'scatter':
            success_5x = create_attention_heatmap_scatter(
                attention_scores=attention_s_np,
                coords=coords_s_np,
                wsi_path=wsi_path,
                output_path=heatmap_path_5x,
                downsample=args.heatmap_downsample,
                point_size=args.point_size,
                alpha=args.heatmap_alpha
            )
        else:
            success_5x = create_attention_heatmap(
                attention_scores=attention_s_np,
                coords=coords_s_np,
                wsi_path=wsi_path,
                output_path=heatmap_path_5x,
                downsample=args.heatmap_downsample,
                alpha=args.heatmap_alpha,
                level=2  # 5x 对应 level=2
            )
        
        # 生成20x热力图
        heatmap_path_20x = os.path.join(heatmap_dir, f'{slide_id}_20x_heatmap.png')
        if args.heatmap_style == 'scatter':
            success_20x = create_attention_heatmap_scatter(
                attention_scores=attention_l_np,
                coords=coords_l_np,
                wsi_path=wsi_path,
                output_path=heatmap_path_20x,
                downsample=args.heatmap_downsample,
                point_size=args.point_size,
                alpha=args.heatmap_alpha
            )
        else:
            success_20x = create_attention_heatmap(
                attention_scores=attention_l_np,
                coords=coords_l_np,
                wsi_path=wsi_path,
                output_path=heatmap_path_20x,
                downsample=args.heatmap_downsample,
                alpha=args.heatmap_alpha,
                level=1  # 20x 对应 level=1
            )
        
        # 保存注意力数据到h5
        h5_path_5x = os.path.join(attention_h5_dir, f'{slide_id}_5x_attention.h5')
        save_attention_to_h5(attention_s_np, coords_s_np, h5_path_5x)
        
        h5_path_20x = os.path.join(attention_h5_dir, f'{slide_id}_20x_attention.h5')
        save_attention_to_h5(attention_l_np, coords_l_np, h5_path_20x)
        
        if success_5x or success_20x:
            success_count += 1

elapsed_time = time.time() - start_time

print(f'\n{"="*60}')
print(f'热力图生成完成!')
print(f'{"="*60}')
print(f'成功生成: {success_count}/{len(dataset)} slides')
print(f'用时: {elapsed_time:.2f} 秒')
print(f'热力图保存路径: {heatmap_dir}')
print(f'注意力数据保存路径: {attention_h5_dir}')

if failed_slides:
    print(f'\n失败的slides ({len(failed_slides)})：')
    for slide_id, reason in failed_slides[:20]:
        print(f'  - {slide_id}: {reason}')
    if len(failed_slides) > 20:
        print(f'  ... 还有 {len(failed_slides) - 20} 个')
    
    # 保存失败列表
    failed_df = pd.DataFrame(failed_slides, columns=['slide_id', 'reason'])
    failed_csv = os.path.join(args.results_dir, 'failed_slides.csv')
    failed_df.to_csv(failed_csv, index=False)
    print(f'\n失败列表已保存: {failed_csv}')

print(f'{"="*60}')
