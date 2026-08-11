#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热力图生成工具函数
用于从模型注意力权重生成 WSI 级别的热力图可视化
"""

import os
import numpy as np
import cv2
import h5py
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import openslide


def get_attention_colormap(cmap_name='jet'):
    """
    获取注意力热图 colormap。

    - 'paper': 更接近论文/示例图的浅色白底风格
    - 'soft_paper': 更柔和、更不艳丽的浅色风格
    - 其他：交给 matplotlib 默认 colormap
    """
    if cmap_name == 'soft_paper':
        return LinearSegmentedColormap.from_list(
            'soft_paper_attention',
            [
                '#8d8cf4',  # soft lavender
                '#86bfff',  # pale blue
                '#a8f0ff',  # soft cyan
                '#fff2a2',  # light yellow
                '#dca2ad',  # muted pink
            ],
            N=256,
        )
    if cmap_name == 'paper':
        return LinearSegmentedColormap.from_list(
            'paper_attention',
            [
                '#7d7bf2',  # lavender blue
                '#72b7ff',  # light blue
                '#8cf3ff',  # cyan
                '#fff47c',  # yellow
                '#e6a0a6',  # pink
            ],
            N=256,
        )
    return plt.get_cmap(cmap_name)


def create_attention_heatmap_scatter(attention_scores, coords, wsi_path, output_path, 
                                     downsample=64, point_size=20, alpha=0.7, cmap='jet',
                                     use_colorbar=True):
    """
    创建点状注意力热力图（scatter plot风格）
    每个patch用一个彩色点表示，颜色深浅代表注意力强度
    
    参数:
        attention_scores: (N,) 每个 patch 的注意力分数
        coords: (N, 2) 每个 patch 在 level-0 的 (x, y) 坐标
        wsi_path: WSI 文件路径
        output_path: 热力图保存路径
        downsample: 缩略图下采样倍率
        point_size: 散点大小
        alpha: 点的透明度 (0-1)
        cmap: colormap 名称
        use_colorbar: 是否显示颜色条
    """
    try:
        # 打开 WSI
        slide = openslide.OpenSlide(wsi_path)
        w, h = slide.dimensions
        
        # 创建缩略图
        thumbnail_size = (int(w / downsample), int(h / downsample))
        thumbnail = slide.get_thumbnail(thumbnail_size)
        thumbnail_np = np.array(thumbnail.convert('RGB'))
        
        # 归一化注意力分数到 [0, 1]
        scores_norm = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
        
        # 将坐标转换到缩略图尺寸
        coords_scaled = coords / downsample
        
        # 创建matplotlib figure
        fig, ax = plt.subplots(figsize=(thumbnail_size[0]/100, thumbnail_size[1]/100), dpi=100)
        
        # 显示缩略图作为背景
        ax.imshow(thumbnail_np, extent=[0, thumbnail_size[0], thumbnail_size[1], 0])
        
        # 绘制散点图
        scatter = ax.scatter(coords_scaled[:, 0], coords_scaled[:, 1], 
                            c=scores_norm, s=point_size, alpha=alpha, 
                            cmap=cmap, edgecolors='none')
        
        # 添加颜色条
        if use_colorbar:
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Score', rotation=270, labelpad=15)
        
        ax.axis('off')
        ax.set_xlim(0, thumbnail_size[0])
        ax.set_ylim(thumbnail_size[1], 0)
        plt.tight_layout(pad=0)
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        slide.close()
        return True
        
    except Exception as e:
        print(f"生成散点热力图失败: {wsi_path}, 错误: {e}")
        return False


def create_attention_heatmap(attention_scores, coords, wsi_path, output_path, 
                             downsample=64, alpha=0.5, cmap='jet', 
                             level=None, patch_size=256,
                             blank_canvas=False, canvas_color=(255, 255, 255),
                             keep_background_white=True):
    """
    创建注意力热力图并叠加到 WSI 缩略图上
    
    参数:
        attention_scores: (N,) 每个 patch 的注意力分数
        coords: (N, 2) 每个 patch 在 level-0 的 (x, y) 坐标
        wsi_path: WSI 文件路径
        output_path: 热力图保存路径
        downsample: 缩略图下采样倍率
        alpha: 热力图透明度 (0-1)
        cmap: colormap 名称
        level: 用于推断 patch 实际大小的层级（若为 None 则从 coords 推断）
        patch_size: patch 边长（在对应 level 的大小）
    """
    try:
        # 打开 WSI
        slide = openslide.OpenSlide(wsi_path)
        w, h = slide.dimensions
        
        # 创建缩略图或空白画布
        thumbnail_size = (int(w / downsample), int(h / downsample))
        if blank_canvas:
            thumbnail = np.full(
                (thumbnail_size[1], thumbnail_size[0], 3),
                np.asarray(canvas_color, dtype=np.uint8),
                dtype=np.uint8
            )
        else:
            thumbnail = slide.get_thumbnail(thumbnail_size)
            thumbnail = np.array(thumbnail.convert('RGB'))
        
        # 归一化注意力分数到 [0, 1]
        scores_norm = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
        
        # 创建热力图画布
        heatmap = np.zeros((thumbnail_size[1], thumbnail_size[0]), dtype=np.float32)
        count_map = np.zeros((thumbnail_size[1], thumbnail_size[0]), dtype=np.float32)
        
        # 如果提供了 level，计算真实 patch 大小
        if level is not None:
            ds = slide.level_downsamples[level]
            patch_size_level0 = int(patch_size * ds)
        else:
            # 从相邻 coords 推断（假设 step=patch_size）
            if len(coords) > 1:
                diffs = np.abs(coords[1:] - coords[:-1])
                diffs = diffs[diffs > 0]
                patch_size_level0 = int(np.median(diffs)) if len(diffs) > 0 else 256 * 4
            else:
                patch_size_level0 = 256 * 4  # 默认假设 20x (level=1, ds≈4)
        
        # 将注意力分数映射到热力图
        for (x0, y0), score in zip(coords, scores_norm):
            x_thumb = int(x0 / downsample)
            y_thumb = int(y0 / downsample)
            patch_w = max(1, int(patch_size_level0 / downsample))
            patch_h = max(1, int(patch_size_level0 / downsample))
            
            x_end = min(x_thumb + patch_w, thumbnail_size[0])
            y_end = min(y_thumb + patch_h, thumbnail_size[1])
            
            if x_thumb < thumbnail_size[0] and y_thumb < thumbnail_size[1]:
                heatmap[y_thumb:y_end, x_thumb:x_end] += score
                count_map[y_thumb:y_end, x_thumb:x_end] += 1
        
        # 取平均（处理重叠区域）
        heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map != 0)
        
        # 应用 colormap
        cmap_obj = get_attention_colormap(cmap)
        heatmap_colored = (cmap_obj(heatmap) * 255).astype(np.uint8)[:, :, :3]
        
        # 确保热力图和缩略图尺寸完全一致
        if heatmap_colored.shape[:2] != thumbnail.shape[:2]:
            # 调整热力图尺寸以匹配缩略图
            heatmap_colored = cv2.resize(heatmap_colored, (thumbnail.shape[1], thumbnail.shape[0]))
        
        # 若使用白底模式，只在有patch覆盖的区域着色，保持其余区域纯白
        if blank_canvas and keep_background_white:
            overlay = thumbnail.copy()
            valid_mask = count_map > 0
            if np.any(valid_mask):
                blended = cv2.addWeighted(thumbnail, 1 - alpha, heatmap_colored, alpha, 0)
                overlay[valid_mask] = blended[valid_mask]
        else:
            # 叠加热力图到缩略图
            overlay = cv2.addWeighted(thumbnail, 1 - alpha, heatmap_colored, alpha, 0)
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(overlay).save(output_path)
        
        slide.close()
        return True
        
    except Exception as e:
        print(f"生成热力图失败: {wsi_path}, 错误: {e}")
        return False


def save_attention_to_h5(attention_scores, coords, output_path):
    """
    保存注意力分数和坐标到 h5 文件
    
    参数:
        attention_scores: (N,) 注意力分数
        coords: (N, 2) 坐标
        output_path: h5 文件保存路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('attention', data=attention_scores)
        f.create_dataset('coords', data=coords)


def create_heatmap_figure(attention_scores, figsize=(10, 8), title='Attention Distribution'):
    """
    创建注意力分布直方图
    
    参数:
        attention_scores: (N,) 注意力分数
        figsize: 图像大小
        title: 标题
    
    返回:
        fig: matplotlib figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(attention_scores, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Attention Score')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig
