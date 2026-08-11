#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate WSI-overlay attention heatmaps for BiomedCLIP-based ViLa-MIL.

This version keeps the original WSI thumbnail as the background and only
colors patch regions that have attention scores. It also adds a colorbar
to explain normalized attention intensity.

Output:
- overlay_heatmaps/<slide_id>_5x_overlay_heatmap.png
- overlay_heatmaps/<slide_id>_20x_overlay_heatmap.png
- overlay_heatmaps/overlay_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils.heatmap_utils import get_attention_colormap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WSI-overlay attention heatmaps")
    parser.add_argument("--attention_h5_dir", type=str, required=True, help="Directory with *_5x_attention.h5 and *_20x_attention.h5")
    parser.add_argument("--wsi_root", type=str, required=True, help="Directory containing WSI files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--wsi_suffix", type=str, default=".svs", help="Default WSI suffix")
    parser.add_argument("--patch_size", type=int, default=256, help="Patch size on the selected WSI level")
    parser.add_argument("--patch_level_5x", type=int, default=2, help="WSI level for 5x features")
    parser.add_argument("--patch_level_20x", type=int, default=1, help="WSI level for 20x features")
    parser.add_argument(
        "--patch_extent_5x_level0",
        type=int,
        default=None,
        help="Optional explicit 5x patch extent on level-0 coordinates; useful for single-level TIFF pipelines",
    )
    parser.add_argument(
        "--patch_extent_20x_level0",
        type=int,
        default=None,
        help="Optional explicit 20x patch extent on level-0 coordinates; useful for single-level TIFF pipelines",
    )
    parser.add_argument("--heatmap_downsample", type=int, default=64, help="Thumbnail downsample factor")
    parser.add_argument("--heatmap_alpha", type=float, default=0.72, help="Attention overlay strength")
    parser.add_argument("--heatmap_cmap", type=str, default="soft_paper", help="Colormap name")
    parser.add_argument("--saturation_boost", type=float, default=1.0, help="Boost saturation on attended regions")
    parser.add_argument("--value_boost", type=float, default=1.0, help="Boost brightness on attended regions")
    parser.add_argument("--max_slides", type=int, default=None, help="Optional limit for quick testing")
    parser.add_argument("--show_colorbar", action="store_true", default=True, help="Show colorbar legend")
    return parser.parse_args()


def load_attention_h5(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        attention_ds = f["attention"]
        try:
            attention = np.asarray(attention_ds).reshape(-1)
        except (TypeError, ValueError):
            attention = attention_ds.astype("float64")[:].reshape(-1)
        coords = np.asarray(f["coords"])
    return attention, coords


def resolve_wsi_path(wsi_root: str, slide_id: str, suffix: str) -> str | None:
    direct = os.path.join(wsi_root, slide_id + suffix)
    if os.path.isfile(direct):
        return direct
    for candidate in Path(wsi_root).glob(f"{slide_id}.*"):
        if candidate.is_file():
            return str(candidate)
    return None


def scale_for_scale_name(scale_name: str, args: argparse.Namespace) -> int:
    if scale_name == "5x":
        return int(args.patch_level_5x)
    if scale_name == "20x":
        return int(args.patch_level_20x)
    raise ValueError(f"Unknown scale: {scale_name}")


def patch_extent_for_scale(scale_name: str, slide: openslide.OpenSlide, args: argparse.Namespace) -> int:
    if scale_name == "5x" and args.patch_extent_5x_level0 is not None:
        return int(args.patch_extent_5x_level0)
    if scale_name == "20x" and args.patch_extent_20x_level0 is not None:
        return int(args.patch_extent_20x_level0)

    level = scale_for_scale_name(scale_name, args)
    if level >= slide.level_count:
        raise ValueError(
            f"slide only has {slide.level_count} level(s), cannot use level={level} for {scale_name}; "
            "pass --patch_extent_5x_level0/--patch_extent_20x_level0 for single-level WSI pipelines"
        )
    return int(round(args.patch_size * slide.level_downsamples[level]))


def collect_slide_ids(attention_h5_dir: str) -> list[str]:
    ids = set()
    for p in Path(attention_h5_dir).glob("*_5x_attention.h5"):
        ids.add(p.name[: -len("_5x_attention.h5")])
    for p in Path(attention_h5_dir).glob("*_20x_attention.h5"):
        ids.add(p.name[: -len("_20x_attention.h5")])
    return sorted(ids)


def normalize_csv_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv_rows(path: str, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_csv_value(row.get(key, "")) for key in fieldnames})


def build_attention_overlay(
    *,
    slide: openslide.OpenSlide,
    coords: np.ndarray,
    scores: np.ndarray,
    patch_extent: int,
    downsample: int,
    alpha: float,
    cmap_name: str,
    saturation_boost: float,
    value_boost: float,
) -> tuple[np.ndarray, np.ndarray]:
    w, h = slide.dimensions
    thumb_w = max(1, int(np.ceil(w / downsample)))
    thumb_h = max(1, int(np.ceil(h / downsample)))
    background = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
    thumb_h, thumb_w = background.shape[:2]

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    count_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)

    patch_w = max(1, int(np.ceil(patch_extent / downsample)))
    patch_h = max(1, int(np.ceil(patch_extent / downsample)))

    for (x0, y0), score in zip(coords, scores_norm):
        x0 = int(x0)
        y0 = int(y0)
        x_thumb = int(np.floor(x0 / downsample))
        y_thumb = int(np.floor(y0 / downsample))
        x_end = min(thumb_w, max(x_thumb + 1, x_thumb + patch_w))
        y_end = min(thumb_h, max(y_thumb + 1, y_thumb + patch_h))
        if x_thumb >= thumb_w or y_thumb >= thumb_h:
            continue
        heatmap[y_thumb:y_end, x_thumb:x_end] += float(score)
        count_map[y_thumb:y_end, x_thumb:x_end] += 1

    heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map != 0)
    cmap = get_attention_colormap(cmap_name)
    heatmap_rgb = (cmap(heatmap) * 255).astype(np.uint8)[:, :, :3]

    if heatmap_rgb.shape[:2] != background.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_LINEAR)
    if count_map.shape[:2] != background.shape[:2]:
        count_map = cv2.resize(count_map, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)

    overlay = background.copy()
    valid_mask = count_map > 0
    if np.any(valid_mask):
        blended = cv2.addWeighted(background, 1 - alpha, heatmap_rgb, alpha, 0)
        overlay[valid_mask] = blended[valid_mask]

        hsv = cv2.cvtColor(overlay, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        sat[valid_mask] = np.clip(sat[valid_mask] * saturation_boost, 0, 255)
        val[valid_mask] = np.clip(val[valid_mask] * value_boost, 0, 255)
        hsv[:, :, 1] = sat
        hsv[:, :, 2] = val
        overlay = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return overlay, heatmap


def save_figure_with_colorbar(
    overlay: np.ndarray,
    cmap_name: str,
    output_path: str,
    title: str,
    show_colorbar: bool = True,
) -> None:
    fig_w = max(8.0, overlay.shape[1] / 95.0)
    fig_h = max(7.0, overlay.shape[0] / 95.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.imshow(overlay)
    ax.set_axis_off()
    ax.set_title(title, fontsize=18, pad=14)

    if show_colorbar:
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=get_attention_colormap(cmap_name))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Normalized attention", rotation=270, labelpad=18)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.join(args.output_dir, "overlay_heatmaps")
    out_dir_5x = os.path.join(out_dir, "5x")
    out_dir_20x = os.path.join(out_dir, "20x")
    os.makedirs(out_dir_5x, exist_ok=True)
    os.makedirs(out_dir_20x, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, str]] = []

    slide_ids = collect_slide_ids(args.attention_h5_dir)
    if not slide_ids:
        raise FileNotFoundError(f"No attention_h5 files found in {args.attention_h5_dir}")
    if args.max_slides is not None:
        slide_ids = slide_ids[: args.max_slides]

    with tqdm(total=len(slide_ids), desc="Generating overlay heatmaps", ncols=100) as pbar:
        for slide_id in slide_ids:
            for scale_name, suffix in (("5x", "_5x_attention.h5"), ("20x", "_20x_attention.h5")):
                h5_path = os.path.join(args.attention_h5_dir, f"{slide_id}{suffix}")
                if not os.path.isfile(h5_path):
                    continue

                try:
                    attention_scores, coords = load_attention_h5(h5_path)
                    wsi_path = resolve_wsi_path(args.wsi_root, slide_id, args.wsi_suffix)
                    if wsi_path is None:
                        failed_rows.append({"slide_id": slide_id, "scale": scale_name, "reason": "wsi_missing"})
                        continue

                    slide = openslide.OpenSlide(wsi_path)
                    try:
                        patch_extent = patch_extent_for_scale(scale_name, slide, args)
                        overlay, _ = build_attention_overlay(
                            slide=slide,
                            coords=coords,
                            scores=attention_scores,
                            patch_extent=patch_extent,
                            downsample=args.heatmap_downsample,
                            alpha=args.heatmap_alpha,
                            cmap_name=args.heatmap_cmap,
                            saturation_boost=args.saturation_boost,
                            value_boost=args.value_boost,
                        )
                    finally:
                        slide.close()

                    scale_out_dir = out_dir_5x if scale_name == "5x" else out_dir_20x
                    out_path = os.path.join(scale_out_dir, f"{slide_id}_{scale_name}_overlay_heatmap.png")
                    save_figure_with_colorbar(
                        overlay=overlay,
                        cmap_name=args.heatmap_cmap,
                        output_path=out_path,
                        title=f"{slide_id} ({scale_name})",
                        show_colorbar=args.show_colorbar,
                    )
                    manifest_rows.append(
                        {
                            "slide_id": slide_id,
                            "scale": scale_name,
                            "output_path": out_path,
                            "num_patches": int(len(coords)),
                        }
                    )
                except Exception as exc:
                    failed_rows.append({"slide_id": slide_id, "scale": scale_name, "reason": str(exc)})
            pbar.update(1)

    write_csv_rows(
        os.path.join(args.output_dir, "overlay_manifest.csv"),
        manifest_rows,
        ["slide_id", "scale", "output_path", "num_patches"],
    )
    if failed_rows:
        write_csv_rows(
            os.path.join(args.output_dir, "overlay_failed.csv"),
            failed_rows,
            ["slide_id", "scale", "reason"],
        )

    print(f"Saved overlay heatmaps to: {out_dir}")
    print(f"Exported rows: {len(manifest_rows)}")
    print(f"Failed rows: {len(failed_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
