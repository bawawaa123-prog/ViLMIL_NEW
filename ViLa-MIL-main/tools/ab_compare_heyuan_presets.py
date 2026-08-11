#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small-sample A/B comparison for Heyuan preprocessing presets.

Goal:
- Compare `presets/tcga.csv` vs `presets/heyuan.csv`
- Use the same Heyuan WSI sample slides
- Re-run segmentation + coordinate generation
- Save:
  - mask previews
  - only-mask previews
  - coords h5
  - stitch/graph previews
  - summary tables for patch_level=1 and patch_level=2

Intended usage:
  python tools/ab_compare_heyuan_presets.py \
    --source data/heyuan/wsi \
    --csv_path dataset_csv/all_data_heyuan.csv \
    --output_dir analysis/heyuan_preset_ab \
    --sample_n 12 \
    --seed 2026
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from create_patches_fp import (
    normalize_slide_id,
    patching,
    resolve_wsi_path,
    segment,
    segment_large_single_level,
    stitching,
    vis_wsi_preview_large_single_level,
)
from wsi_core.WholeSlideImage import WholeSlideImage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B compare tcga.csv vs heyuan.csv on a Heyuan sample")
    parser.add_argument("--source", required=True, help="Heyuan WSI root")
    parser.add_argument("--csv_path", required=True, help="CSV with slide_id column")
    parser.add_argument("--output_dir", required=True, help="Output root directory")
    parser.add_argument("--preset_a", default="tcga.csv", help="Preset A under presets/")
    parser.add_argument("--preset_b", default="heyuan.csv", help="Preset B under presets/")
    parser.add_argument("--sample_n", type=int, default=12, help="Number of slides to sample")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--slide_ids", default=None, help="Optional comma-separated slide_id list")
    parser.add_argument("--patch_levels", default="1,2", help="Comma-separated patch levels to compare")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--step_size", type=int, default=256)
    parser.add_argument("--stitch_downscale", type=int, default=64)
    parser.add_argument("--skip_stitch", action="store_true", default=False)
    return parser.parse_args()


def load_preset(preset_name: str) -> tuple[dict, dict, dict, dict]:
    preset_path = REPO_ROOT / "presets" / preset_name
    if not preset_path.is_file():
        raise FileNotFoundError(f"Preset not found: {preset_path}")
    row = pd.read_csv(preset_path).iloc[0]

    seg_params = {
        "seg_level": int(row["seg_level"]),
        "sthresh": int(row["sthresh"]),
        "mthresh": int(row["mthresh"]),
        "close": int(row["close"]),
        "use_otsu": bool(row["use_otsu"]),
        "keep_ids": str(row["keep_ids"]),
        "exclude_ids": str(row["exclude_ids"]),
    }
    filter_params = {
        "a_t": int(row["a_t"]),
        "a_h": int(row["a_h"]),
        "max_n_holes": int(row["max_n_holes"]),
    }
    vis_params = {
        "vis_level": int(row["vis_level"]),
        "line_thickness": int(row["line_thickness"]),
    }
    patch_params = {
        "use_padding": bool(row["use_padding"]),
        "contour_fn": str(row["contour_fn"]),
    }
    return seg_params, filter_params, vis_params, patch_params


def select_slide_ids(csv_path: str, sample_n: int, seed: int, explicit_slide_ids: str | None) -> list[str]:
    df = pd.read_csv(csv_path)
    if "slide_id" not in df.columns:
        raise ValueError(f"slide_id column missing in {csv_path}")
    slide_ids = df["slide_id"].dropna().astype(str).tolist()
    slide_ids = sorted(set(slide_ids))

    if explicit_slide_ids:
        requested = [s.strip() for s in explicit_slide_ids.split(",") if s.strip()]
        missing = [s for s in requested if s not in slide_ids]
        if missing:
            raise ValueError(f"These slide_ids are not in CSV: {missing}")
        return requested

    rng = random.Random(seed)
    if sample_n >= len(slide_ids):
        return slide_ids
    return sorted(rng.sample(slide_ids, sample_n))


def count_coords(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            return 0
        return int(len(f["coords"]))


def maybe_parse_keep_exclude(seg_params: dict) -> dict:
    seg_params = deepcopy(seg_params)

    keep_ids = str(seg_params["keep_ids"])
    if keep_ids != "none" and len(keep_ids) > 0:
        seg_params["keep_ids"] = np.array(keep_ids.split(",")).astype(int)
    else:
        seg_params["keep_ids"] = []

    exclude_ids = str(seg_params["exclude_ids"])
    if exclude_ids != "none" and len(exclude_ids) > 0:
        seg_params["exclude_ids"] = np.array(exclude_ids.split(",")).astype(int)
    else:
        seg_params["exclude_ids"] = []

    return seg_params


def process_one_slide_with_preset(
    slide_id: str,
    source: str,
    preset_name: str,
    patch_levels: list[int],
    patch_size: int,
    step_size: int,
    stitch_downscale: int,
    skip_stitch: bool,
    output_dir: Path,
) -> list[dict]:
    seg_params, filter_params, vis_params, patch_params = load_preset(preset_name)
    preset_dir = output_dir / Path(preset_name).stem
    (preset_dir / "masks").mkdir(parents=True, exist_ok=True)
    (preset_dir / "only_masks").mkdir(parents=True, exist_ok=True)

    wsi_path = resolve_wsi_path(source, slide_id)
    if wsi_path is None:
        return [{
            "slide_id": slide_id,
            "preset": preset_name,
            "status": "missing_slide",
        }]

    slide_obj = WholeSlideImage(wsi_path)
    slide_obj.name = normalize_slide_id(slide_id)

    current_seg_params = maybe_parse_keep_exclude(seg_params)
    current_vis_params = deepcopy(vis_params)
    current_filter_params = deepcopy(filter_params)
    current_patch_params = deepcopy(patch_params)

    if current_vis_params["vis_level"] < 0:
        if len(slide_obj.level_dim) == 1:
            current_vis_params["vis_level"] = 0
        else:
            current_vis_params["vis_level"] = slide_obj.getOpenSlide().get_best_level_for_downsample(64)

    if current_seg_params["seg_level"] < 0:
        if len(slide_obj.level_dim) == 1:
            current_seg_params["seg_level"] = 0
        else:
            current_seg_params["seg_level"] = slide_obj.getOpenSlide().get_best_level_for_downsample(64)

    seg_level = int(current_seg_params["seg_level"])
    vis_level = int(current_vis_params["vis_level"])
    level_w, level_h = slide_obj.level_dim[seg_level]
    is_large_single_level = len(slide_obj.level_dim) == 1 and (level_w * level_h > 1e8)

    t0 = time.time()
    if is_large_single_level:
        slide_obj, seg_time = segment_large_single_level(
            slide_obj,
            current_seg_params,
            current_filter_params,
            wsi_path=wsi_path,
        )
    else:
        slide_obj, seg_time = segment(slide_obj, current_seg_params, current_filter_params)
    seg_wall = time.time() - t0

    if slide_obj.contours_tissue is None or len(slide_obj.contours_tissue) == 0:
        return [{
            "slide_id": slide_id,
            "preset": preset_name,
            "status": "failed_seg",
            "seg_level": seg_level,
            "vis_level": vis_level,
            "seg_time_sec": seg_time,
            "seg_wall_sec": seg_wall,
            "n_contours": 0,
        }]

    try:
        if is_large_single_level:
            mask_img, only_mask_img = vis_wsi_preview_large_single_level(
                slide_obj,
                line_thickness=current_vis_params.get("line_thickness", 100),
                max_thumb_side=4096,
            )
        else:
            mask_img, only_mask_img = slide_obj.visWSI(**current_vis_params)
        mask_path = preset_dir / "masks" / f"{slide_id}.jpg"
        only_mask_path = preset_dir / "only_masks" / f"{slide_id}.png"
        mask_img.save(mask_path)
        only_mask_img.save(only_mask_path)
    except Exception as exc:
        mask_path = None
        only_mask_path = None
        print(f"[Warn] Failed to save mask for {slide_id} with {preset_name}: {exc}")

    rows: list[dict] = []
    n_contours = len(slide_obj.contours_tissue)

    for patch_level in patch_levels:
        level_dir = preset_dir / f"patch_level_{patch_level}"
        patch_save_dir = level_dir / f"patches_{patch_size}"
        graph_dir = level_dir / f"graph_{patch_size}"
        patch_save_dir.mkdir(parents=True, exist_ok=True)
        graph_dir.mkdir(parents=True, exist_ok=True)

        status = "processed"
        reason = ""
        coord_count = 0
        patch_time_sec = -1.0
        stitch_time_sec = -1.0
        h5_path = patch_save_dir / f"{slide_id}.h5"
        graph_path = graph_dir / f"{slide_id}.jpg"

        try:
            current_patch = deepcopy(current_patch_params)
            current_patch.update({
                "patch_level": patch_level,
                "patch_size": patch_size,
                "step_size": step_size,
                "save_path": str(patch_save_dir),
            })
            _, patch_time_sec = patching(slide_obj, **current_patch)
            if h5_path.is_file():
                coord_count = count_coords(h5_path)
            else:
                status = "missing_h5"
        except Exception as exc:
            status = "failed_patch"
            reason = str(exc)

        if status == "processed" and not skip_stitch and h5_path.is_file():
            try:
                heatmap, stitch_time_sec = stitching(str(h5_path), slide_obj, downscale=stitch_downscale)
                heatmap.save(graph_path)
            except Exception as exc:
                reason = f"stitch_failed:{exc}"

        rows.append({
            "slide_id": slide_id,
            "preset": preset_name,
            "status": status,
            "reason": reason,
            "seg_level": seg_level,
            "vis_level": vis_level,
            "patch_level": patch_level,
            "patch_size": patch_size,
            "step_size": step_size,
            "seg_time_sec": seg_time,
            "seg_wall_sec": seg_wall,
            "patch_time_sec": patch_time_sec,
            "stitch_time_sec": stitch_time_sec,
            "n_contours": n_contours,
            "coord_count": coord_count,
            "mask_path": str(mask_path) if mask_path else "",
            "only_mask_path": str(only_mask_path) if only_mask_path else "",
            "coords_h5_path": str(h5_path) if h5_path.is_file() else "",
            "graph_path": str(graph_path) if graph_path.is_file() else "",
        })

    return rows


def build_wide_comparison(raw_df: pd.DataFrame, preset_a: str, preset_b: str) -> pd.DataFrame:
    ok_df = raw_df[raw_df["status"] == "processed"].copy()
    if ok_df.empty:
        return pd.DataFrame()

    pivot = ok_df.pivot_table(
        index=["slide_id", "patch_level"],
        columns="preset",
        values="coord_count",
        aggfunc="first",
    ).reset_index()

    if preset_a in pivot.columns and preset_b in pivot.columns:
        pivot["delta"] = pivot[preset_b] - pivot[preset_a]
        pivot["ratio"] = np.where(pivot[preset_a] > 0, pivot[preset_b] / pivot[preset_a], np.nan)
    return pivot


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_levels = [int(x.strip()) for x in args.patch_levels.split(",") if x.strip()]
    slide_ids = select_slide_ids(args.csv_path, args.sample_n, args.seed, args.slide_ids)

    pd.DataFrame({"slide_id": slide_ids}).to_csv(output_dir / "sample_slides.csv", index=False)

    all_rows: list[dict] = []
    for preset_name in [args.preset_a, args.preset_b]:
        print(f"\n=== Running preset: {preset_name} ===")
        for idx, slide_id in enumerate(slide_ids, start=1):
            print(f"[{idx}/{len(slide_ids)}] {slide_id}")
            rows = process_one_slide_with_preset(
                slide_id=slide_id,
                source=args.source,
                preset_name=preset_name,
                patch_levels=patch_levels,
                patch_size=args.patch_size,
                step_size=args.step_size,
                stitch_downscale=args.stitch_downscale,
                skip_stitch=args.skip_stitch,
                output_dir=output_dir,
            )
            all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    raw_csv = output_dir / "ab_raw_results.csv"
    raw_df.to_csv(raw_csv, index=False)

    wide_df = build_wide_comparison(raw_df, args.preset_a, args.preset_b)
    wide_csv = output_dir / "ab_coord_count_comparison.csv"
    wide_df.to_csv(wide_csv, index=False)

    summary_rows = []
    if not raw_df.empty:
        processed = raw_df[raw_df["status"] == "processed"].copy()
        if not processed.empty:
            grp = processed.groupby(["preset", "patch_level"])["coord_count"]
            for (preset, patch_level), values in grp:
                summary_rows.append({
                    "preset": preset,
                    "patch_level": patch_level,
                    "n_slides": int(values.shape[0]),
                    "coord_count_min": int(values.min()),
                    "coord_count_median": float(values.median()),
                    "coord_count_mean": float(values.mean()),
                    "coord_count_max": int(values.max()),
                })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "ab_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== Done ===")
    print(f"Sample slides: {output_dir / 'sample_slides.csv'}")
    print(f"Raw results:   {raw_csv}")
    print(f"Wide compare:  {wide_csv}")
    print(f"Summary:       {summary_csv}")
    print(f"Masks/graphs:  {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
