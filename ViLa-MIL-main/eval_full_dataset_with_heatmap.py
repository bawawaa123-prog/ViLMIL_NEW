#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全数据集评估 + 注意力热力图生成
加载训练好的权重，对整个数据集进行评估并生成每个 WSI 的注意力热力图

用法（cmd）：
  python eval_full_dataset_with_heatmap.py ^
    --task task_adenocarcinoma_xin ^
    --mode transformer ^
    --model_type ViLa_MIL ^
    --data_root_dir "Z:\\ljh\\data\\results\\new" ^
    --data_folder_s "features_5x_h5" ^
    --data_folder_l "features_20x_h5" ^
    --wsi_root "D:\\FenLei\\data" ^
    --checkpoint "trained_models\\adenocarcinoma_vila_mil_new_s1\\s_0_checkpoint.pt" ^
    --results_dir "./eval_heatmap_results" ^
    --text_prompt_path "text_prompt\\adenocarcinoma_dual_scale_prompt.csv" ^
    --generate_heatmap
"""

from __future__ import print_function
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc as calc_auc
import time

from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_ViLa_MIL import ViLa_MIL_Model
from utils.heatmap_utils import create_attention_heatmap, save_attention_to_h5
import ml_collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='全数据集评估 + 热力图生成')
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--mode', type=str, default='transformer')
parser.add_argument('--model_type', type=str, default='ViLa_MIL')
parser.add_argument('--data_root_dir', type=str, required=True)
parser.add_argument('--data_folder_s', type=str, required=True)
parser.add_argument('--data_folder_l', type=str, required=True)
parser.add_argument('--wsi_root', type=str, required=True, help='WSI 文件根目录')
parser.add_argument('--checkpoint', type=str, required=True, help='训练好的权重文件路径')
parser.add_argument('--results_dir', type=str, default='./eval_heatmap_results')
parser.add_argument('--text_prompt_path', type=str, required=True)
parser.add_argument('--csv_path', type=str, default='dataset_csv/all_data.csv', help='数据集 CSV')
parser.add_argument('--generate_heatmap', action='store_true', default=False)
parser.add_argument('--heatmap_downsample', type=int, default=64)
parser.add_argument('--heatmap_alpha', type=float, default=0.5)
parser.add_argument('--prototype_number', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=1, help='batch size (推荐保持1)')

args = parser.parse_args()

# 加载文本提示
if args.text_prompt_path:
    try:
        df_tp = pd.read_csv(args.text_prompt_path)
        cols = [c.strip().lower() for c in df_tp.columns]
        low_prompts, high_prompts = [], []
        if 'low_resolution_description' in cols and 'high_resolution_description' in cols:
            low_idx = cols.index('low_resolution_description')
            high_idx = cols.index('high_resolution_description')
            low_prompts = df_tp.iloc[:, low_idx].astype(str).fillna("").tolist()
            high_prompts = df_tp.iloc[:, high_idx].astype(str).fillna("").tolist()
        args.text_prompt = list(map(str, low_prompts)) + list(map(str, high_prompts))
    except Exception:
        arr = pd.read_csv(args.text_prompt_path, header=None).values
        args.text_prompt = [str(x) for x in arr.reshape(-1).tolist()]

os.makedirs(args.results_dir, exist_ok=True)

# 加载数据集
print('\n加载数据集...')
if args.task == 'task_adenocarcinoma_xin':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_path,
        mode=args.mode,
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict={'Adenocarcinoma': 0, 'NonAdenocarcinoma': 1},
        patient_strat=False,
        ignore=[]
    )
else:
    raise NotImplementedError(f"Task {args.task} not implemented")

print(f'数据集大小: {len(dataset)} 个 slides')

# 加载模型
print('\n加载模型...')
config = ml_collections.ConfigDict()
config.input_size = 1024
config.hidden_size = 192  # 必须与训练时的配置一致
config.text_prompt = args.text_prompt
config.prototype_number = args.prototype_number

model = ViLa_MIL_Model(config, num_classes=args.n_classes)

if os.path.isfile(args.checkpoint):
    print(f'加载权重: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    print('模型加载完成!')
else:
    raise FileNotFoundError(f'权重文件不存在: {args.checkpoint}')

model = model.to(device)
model.eval()

print('模型加载完成!')

# 准备数据加载器
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

# 评估
print('\n开始评估整个数据集...')
all_slide_ids = []
all_labels = []
all_probs = []
all_preds = []
attention_data = {}  # {slide_id: {'attention': ..., 'coords_s': ..., 'coords_l': ...}}

start_time = time.time()

with torch.no_grad():
    for idx, (data_s, coord_s, data_l, coords_l, label, slide_id) in enumerate(tqdm(loader, desc='Evaluating', ncols=100)):
        # slide_id 是一个 tuple，取第一个元素
        if isinstance(slide_id, (tuple, list)):
            slide_id = slide_id[0]
        
        data_s = data_s.to(device)
        data_l = data_l.to(device)
        
        # 前向传播（带注意力权重）
        logits, Y_prob, Y_hat, attention_scores_s, attention_scores_l = model.forward_with_attention(
            data_s, coord_s, data_l, coords_l, label
        )
        
        # 记录预测结果
        all_slide_ids.append(slide_id)
        all_labels.append(label.item())
        all_probs.append(Y_prob.cpu().numpy()[0])
        all_preds.append(Y_hat.item())
        
        # 保存注意力数据(用于生成热力图)
        if args.generate_heatmap:
            # coords 可能已经是 [N, 2] (取决于 dataset 实现)
            coords_s_np = coord_s.cpu().numpy()
            coords_l_np = coords_l.cpu().numpy()
            # 如果有 batch 维度,去掉 [batch, N, 2] -> [N, 2]
            if coords_s_np.ndim == 3:
                coords_s_np = coords_s_np[0]
            if coords_l_np.ndim == 3:
                coords_l_np = coords_l_np[0]
            
            attention_data[slide_id] = {
                'attention_s': attention_scores_s.cpu().numpy().flatten(),
                'coords_s': coords_s_np,
                'attention_l': attention_scores_l.cpu().numpy().flatten(),
                'coords_l': coords_l_np
            }

eval_time = time.time() - start_time

# 计算整体指标
all_labels = np.array(all_labels)
all_probs_arr = np.array(all_probs)
all_preds = np.array(all_preds)

# 多分类处理
if args.n_classes == 2:
    # 二分类：取正类概率
    y_score = all_probs_arr[:, 1]
    auc_score = roc_auc_score(all_labels, y_score)
else:
    # 多分类：macro AUC
    auc_score = roc_auc_score(all_labels, all_probs_arr, multi_class='ovr', average='macro')

acc = np.mean(all_labels == all_preds)
f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f'\n{"="*60}')
print(f'全数据集评估结果:')
print(f'{"="*60}')
print(f'样本数: {len(dataset)}')
print(f'AUC:    {auc_score:.4f}')
print(f'ACC:    {acc:.4f}')
print(f'F1:     {f1:.4f}')
print(f'评估用时: {eval_time:.2f} 秒')
print(f'{"="*60}')

# 保存详细结果
results_df = pd.DataFrame({
    'slide_id': all_slide_ids,
    'true_label': all_labels,
    'predicted_label': all_preds,
    'prob_class_0': all_probs_arr[:, 0],
    'prob_class_1': all_probs_arr[:, 1] if args.n_classes == 2 else np.nan
})
results_csv = os.path.join(args.results_dir, 'full_dataset_results.csv')
results_df.to_csv(results_csv, index=False)
print(f'\n详细结果已保存: {results_csv}')

# 保存汇总指标
summary_df = pd.DataFrame({
    'metric': ['AUC', 'ACC', 'F1', 'n_samples', 'eval_time_sec'],
    'value': [auc_score, acc, f1, len(dataset), eval_time]
})
summary_csv = os.path.join(args.results_dir, 'summary_metrics.csv')
summary_df.to_csv(summary_csv, index=False)
print(f'汇总指标已保存: {summary_csv}')

# 生成热力图
if args.generate_heatmap:
    print(f'\n开始生成注意力热力图...')
    heatmap_dir = os.path.join(args.results_dir, 'heatmaps')
    attention_h5_dir = os.path.join(args.results_dir, 'attention_h5')
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(attention_h5_dir, exist_ok=True)
    
    success_count = 0
    for slide_id in tqdm(all_slide_ids, desc='生成热力图', ncols=100):
        wsi_path = os.path.join(args.wsi_root, slide_id + '.svs')
        
        if not os.path.isfile(wsi_path):
            print(f'跳过 {slide_id}: WSI 文件不存在')
            continue
        
        att_data = attention_data[slide_id]
        
        # 生成低倍率热力图（主要）
        heatmap_path_s = os.path.join(heatmap_dir, f'{slide_id}_5x_heatmap.png')
        success_s = create_attention_heatmap(
            attention_scores=att_data['attention_s'],
            coords=att_data['coords_s'],
            wsi_path=wsi_path,
            output_path=heatmap_path_s,
            downsample=args.heatmap_downsample,
            alpha=args.heatmap_alpha,
            level=2  # 5x 对应 level=2
        )
        
        # 生成高倍率热力图（可选）
        heatmap_path_l = os.path.join(heatmap_dir, f'{slide_id}_20x_heatmap.png')
        success_l = create_attention_heatmap(
            attention_scores=att_data['attention_l'],
            coords=att_data['coords_l'],
            wsi_path=wsi_path,
            output_path=heatmap_path_l,
            downsample=args.heatmap_downsample,
            alpha=args.heatmap_alpha,
            level=1  # 20x 对应 level=1
        )
        
        # 保存注意力数据到 h5
        h5_path = os.path.join(attention_h5_dir, f'{slide_id}_attention.h5')
        save_attention_to_h5(
            attention_scores=att_data['attention_s'],
            coords=att_data['coords_s'],
            output_path=h5_path
        )
        
        if success_s or success_l:
            success_count += 1
    
    print(f'\n热力图生成完成!')
    print(f'成功生成: {success_count}/{len(all_slide_ids)}')
    print(f'热力图保存路径: {heatmap_dir}')
    print(f'注意力数据保存路径: {attention_h5_dir}')

print(f'\n✅ 全数据集评估完成!')
print(f'结果保存在: {args.results_dir}')
