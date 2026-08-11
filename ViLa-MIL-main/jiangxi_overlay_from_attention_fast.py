#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast Jiangxi overlay heatmap generation from existing attention_h5 files.

This is a post-processing script: it does not run the model again. It reuses
attention_h5 exported by jiangxi_generate_attention_heatmaps_and_topk.py and
generates WSI-background overlay PNGs in the same directory layout:

- overlay_heatmaps/5x/<slide_id>_5x_overlay_heatmap.png
- overlay_heatmaps/20x/<slide_id>_20x_overlay_heatmap.png
- overlay_manifest.csv

The key speed-up is that each WSI thumbnail is read only once and reused for
both 5x and 20x overlays.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

from utils.heatmap_utils import get_attention_colormap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast Jiangxi overlay heatmaps from existing attention_h5")
    parser.add_argument("--attention_h5_dir", type=str, required=True)
    parser.add_argument("--wsi_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--wsi_suffix", type=str, default=".tif")
    parser.add_argument("--patch_size_5x_level0", type=int, default=4096)
    parser.add_argument("--patch_size_20x_level0", type=int, default=1024)
    parser.add_argument("--heatmap_downsample", type=int, default=64)
    parser.add_argument("--heatmap_alpha", type=float, default=0.72)
    parser.add_argument("--heatmap_cmap", type=str, default="soft_paper")
    parser.add_argument("--saturation_boost", type=float, default=1.0)
    parser.add_argument("--value_boost", type=float, default=1.0)
    parser.add_argument("--max_slides", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=False)
    return parser.parse_args()


def collect_slide_ids(attention_h5_dir: str) -> list[str]:
    ids = set()
    for path in Path(attention_h5_dir).glob("*_5x_attention.h5"):
        ids.add(path.name[: -len("_5x_attention.h5")])
    for path in Path(attention_h5_dir).glob("*_20x_attention.h5"):
        ids.add(path.name[: -len("_20x_attention.h5")])
    return sorted(ids)


def resolve_wsi_path(wsi_root: str, slide_id: str, suffix: str) -> str | None:
    direct = os.path.join(wsi_root, slide_id + suffix)
    if os.path.isfile(direct):
        return direct
    for candidate in Path(wsi_root).glob(f"{slide_id}.*"):
        if candidate.is_file():
            return str(candidate)
    return None


def load_attention_h5(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        attention = np.asarray(f["attention"]).reshape(-1)
        coords = np.asarray(f["coords"])
    return attention, coords


def load_background(slide_path: str, downsample: int) -> np.ndarray:
    slide = openslide.OpenSlide(slide_path)
    try:
        w, h = slide.dimensions
        thumb_w = max(1, int(np.ceil(w / downsample)))
        thumb_h = max(1, int(np.ceil(h / downsample)))
        return np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
    finally:
        slide.close()


def build_overlay_from_background(
    *,
    background: np.ndarray,
    coords: np.ndarray,
    scores: np.ndarray,
    patch_extent: int,
    downsample: int,
    alpha: float,
    cmap_name: str,
    saturation_boost: float,
    value_boost: float,
) -> np.ndarray:
    thumb_h, thumb_w = background.shape[:2]
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    count_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)

    patch_w = max(1, int(np.ceil(patch_extent / downsample)))
    patch_h = max(1, int(np.ceil(patch_extent / downsample)))

    for (x0, y0), score in zip(coords, scores_norm):
        x_thumb = int(np.floor(int(x0) / downsample))
        y_thumb = int(np.floor(int(y0) / downsample))
        if x_thumb >= thumb_w or y_thumb >= thumb_h:
            continue
        x_end = min(thumb_w, max(x_thumb + 1, x_thumb + patch_w))
        y_end = min(thumb_h, max(y_thumb + 1, y_thumb + patch_h))
        heatmap[y_thumb:y_end, x_thumb:x_end] += float(score)
        count_map[y_thumb:y_end, x_thumb:x_end] += 1.0

    heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map != 0)
    heatmap_rgb = (get_attention_colormap(cmap_name)(heatmap) * 255).astype(np.uint8)[:, :, :3]

    overlay = background.copy()
    valid_mask = count_map > 0
    if np.any(valid_mask):
        blended = cv2.addWeighted(background, 1 - alpha, heatmap_rgb, alpha, 0)
        overlay[valid_mask] = blended[valid_mask]

        if saturation_boost != 1.0 or value_boost != 1.0:
            hsv = cv2.cvtColor(overlay, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1][valid_mask] = np.clip(hsv[:, :, 1][valid_mask] * saturation_boost, 0, 255)
            hsv[:, :, 2][valid_mask] = np.clip(hsv[:, :, 2][valid_mask] * value_boost, 0, 255)
            overlay = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return overlay


def write_csv(path: str, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    out_dir = os.path.join(args.output_dir, "overlay_heatmaps")
    out_5x = os.path.join(out_dir, "5x")
    out_20x = os.path.join(out_dir, "20x")
    os.makedirs(out_5x, exist_ok=True)
    os.makedirs(out_20x, exist_ok=True)

    slide_ids = collect_slide_ids(args.attention_h5_dir)
    if args.max_slides is not None:
        slide_ids = slide_ids[: args.max_slides]

    manifest_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, str]] = []

    for slide_id in tqdm(slide_ids, desc="Fast Jiangxi overlay heatmaps", ncols=100):
        out_path_5x = os.path.join(out_5x, f"{slide_id}_5x_overlay_heatmap.png")
        out_path_20x = os.path.join(out_20x, f"{slide_id}_20x_overlay_heatmap.png")
        if args.skip_existing and os.path.isfile(out_path_5x) and os.path.isfile(out_path_20x):
            continue

        wsi_path = resolve_wsi_path(args.wsi_root, slide_id, args.wsi_suffix)
        if wsi_path is None:
            failed_rows.append({"slide_id": slide_id, "scale": "both", "reason": "wsi_missing"})
            continue

        try:
            background = load_background(wsi_path, args.heatmap_downsample)
        except Exception as exc:
            failed_rows.append({"slide_id": slide_id, "scale": "both", "reason": f"thumbnail_failed:{exc}"})
            continue

        for scale_name, suffix, patch_extent, out_path in (
            ("5x", "_5x_attention.h5", args.patch_size_5x_level0, out_path_5x),
            ("20x", "_20x_attention.h5", args.patch_size_20x_level0, out_path_20x),
        ):
            if args.skip_existing and os.path.isfile(out_path):
                continue

            h5_path = os.path.join(args.attention_h5_dir, f"{slide_id}{suffix}")
            if not os.path.isfile(h5_path):
                failed_rows.append({"slide_id": slide_id, "scale": scale_name, "reason": "attention_h5_missing"})
                continue

            try:
                scores, coords = load_attention_h5(h5_path)
                overlay = build_overlay_from_background(
                    background=background,
                    coords=coords,
                    scores=scores,
                    patch_extent=patch_extent,
                    downsample=args.heatmap_downsample,
                    alpha=args.heatmap_alpha,
                    cmap_name=args.heatmap_cmap,
                    saturation_boost=args.saturation_boost,
                    value_boost=args.value_boost,
                )
                Image.fromarray(overlay).save(out_path)
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

    write_csv(
        os.path.join(args.output_dir, "overlay_manifest.csv"),
        manifest_rows,
        ["slide_id", "scale", "output_path", "num_patches"],
    )
    if failed_rows:
        write_csv(
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
