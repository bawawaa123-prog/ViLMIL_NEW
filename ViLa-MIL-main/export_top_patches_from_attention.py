#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export top-k patches using saved attention_h5 files.

Two source modes are supported:

1. Crop only the selected top/bottom patches directly from WSI files.
2. Copy/symlink patches from existing patch directories (legacy mode).

Expected inputs:
- attention_h5/<slide_id>_5x_attention.h5
- attention_h5/<slide_id>_20x_attention.h5

WSI mode additionally expects:
- <wsi_root>/<slide_id>.svs

Legacy patch-directory mode additionally expects:
- patches_5x/<slide_id>/<slide_id>_<x>_<y>.png
- patches_20x/<slide_id>/<slide_id>_<x>_<y>.png
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top-k patches from attention results")
    parser.add_argument("--attention_h5_dir", type=str, required=True, help="Directory containing *_5x_attention.h5 and *_20x_attention.h5")
    parser.add_argument("--patches_5x_dir", type=str, default=None, help="Root patches_5x directory (legacy patch-directory mode)")
    parser.add_argument("--patches_20x_dir", type=str, default=None, help="Root patches_20x directory (legacy patch-directory mode)")
    parser.add_argument("--wsi_root", type=str, default=None, help="Crop selected patches directly from WSI files")
    parser.add_argument("--wsi_suffix", type=str, default=".svs", help="Default WSI suffix")
    parser.add_argument("--patch_size", type=int, default=256, help="Output patch size at the selected WSI level")
    parser.add_argument("--patch_level_5x", type=int, default=2, help="WSI level used for 5x patches")
    parser.add_argument("--patch_level_20x", type=int, default=1, help="WSI level used for 20x patches")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k patches per slide per scale")
    parser.add_argument("--slides_csv", type=str, default=None, help="Optional CSV to restrict slide_id list")
    parser.add_argument("--copy_mode", type=str, default="copy", choices=["copy", "symlink"])
    parser.add_argument("--export_bottom_k", type=int, default=0, help="Optional export of bottom-k patches")
    parser.add_argument("--max_slides", type=int, default=None, help="Optional slide limit for testing")
    args = parser.parse_args()

    using_wsi = args.wsi_root is not None
    using_patch_dirs = args.patches_5x_dir is not None or args.patches_20x_dir is not None
    if using_wsi and using_patch_dirs:
        parser.error("Use either --wsi_root or --patches_5x_dir/--patches_20x_dir, not both")
    if not using_wsi and not (args.patches_5x_dir and args.patches_20x_dir):
        parser.error("Pass --wsi_root, or pass both --patches_5x_dir and --patches_20x_dir")
    if args.patch_size <= 0:
        parser.error("--patch_size must be positive")
    if args.top_k < 0 or args.export_bottom_k < 0:
        parser.error("--top_k and --export_bottom_k must be non-negative")
    return args


def load_slide_ids(args: argparse.Namespace) -> list[str]:
    if args.slides_csv:
        df = pd.read_csv(args.slides_csv)
        if "slide_id" not in df.columns:
            raise ValueError(f"slide_id column missing in {args.slides_csv}")
        slide_ids = df["slide_id"].dropna().astype(str).tolist()
        return slide_ids[: args.max_slides] if args.max_slides is not None else slide_ids

    slide_ids = set()
    for path in Path(args.attention_h5_dir).glob("*_5x_attention.h5"):
        slide_ids.add(path.name[: -len("_5x_attention.h5")])
    for path in Path(args.attention_h5_dir).glob("*_20x_attention.h5"):
        slide_ids.add(path.name[: -len("_20x_attention.h5")])
    slide_ids_sorted = sorted(slide_ids)
    return slide_ids_sorted[: args.max_slides] if args.max_slides is not None else slide_ids_sorted


def load_attention_h5(path: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        attention = np.asarray(f["attention"])
        coords = np.asarray(f["coords"])
    return attention.reshape(-1), coords


def patch_filename_candidates(slide_id: str, x: int, y: int) -> list[str]:
    """
    Return filename candidates for different patch-export conventions.

    Supported patterns:
    - <slide_id>_<x>_<y>.png
    - <slide_id>_256_<x>_<y>.png
    """
    x = int(x)
    y = int(y)
    return [
        f"{slide_id}_{x}_{y}.png",
        f"{slide_id}_256_{x}_{y}.png",
    ]


def resolve_wsi_path(wsi_root: str, slide_id: str, suffix: str) -> str | None:
    direct = os.path.join(wsi_root, slide_id + suffix)
    if os.path.isfile(direct):
        return direct

    for candidate in Path(wsi_root).glob(f"{slide_id}.*"):
        if candidate.is_file():
            return str(candidate)
    return None


def export_one_scale_from_patch_dir(
    slide_id: str,
    scale_name: str,
    attention_h5: str,
    patches_root: str,
    output_root: str,
    top_k: int,
    export_bottom_k: int,
    copy_mode: str,
) -> list[dict[str, object]]:
    attention, coords = load_attention_h5(attention_h5)
    if len(attention) != len(coords):
        raise ValueError(f"Length mismatch in {attention_h5}: attn={len(attention)} coords={len(coords)}")

    patch_dir = os.path.join(patches_root, slide_id)
    if not os.path.isdir(patch_dir):
        raise FileNotFoundError(f"Patch dir missing: {patch_dir}")

    sort_idx_desc = np.argsort(-attention)
    records: list[dict[str, object]] = []

    def export_indices(indices: np.ndarray, split_name: str) -> None:
        split_dir = os.path.join(output_root, scale_name, split_name, slide_id)
        os.makedirs(split_dir, exist_ok=True)
        for rank, idx in enumerate(indices, start=1):
            x, y = coords[idx]
            src_path = None
            src_name = None
            for candidate in patch_filename_candidates(slide_id, x, y):
                candidate_path = os.path.join(patch_dir, candidate)
                if os.path.isfile(candidate_path):
                    src_path = candidate_path
                    src_name = candidate
                    break
            if src_path is None or src_name is None:
                continue
            dst_name = f"rank_{rank:03d}_score_{attention[idx]:.6f}_{src_name}"
            dst_path = os.path.join(split_dir, dst_name)
            if copy_mode == "symlink":
                if os.path.lexists(dst_path):
                    os.unlink(dst_path)
                os.symlink(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            records.append(
                {
                    "slide_id": slide_id,
                    "scale": scale_name,
                    "split": split_name,
                    "rank": rank,
                    "attention": float(attention[idx]),
                    "x": int(x),
                    "y": int(y),
                    "src_path": src_path,
                    "dst_path": dst_path,
                }
            )

    top_n = min(top_k, len(sort_idx_desc))
    export_indices(sort_idx_desc[:top_n], "top")

    if export_bottom_k > 0:
        bottom_n = min(export_bottom_k, len(sort_idx_desc))
        export_indices(sort_idx_desc[::-1][:bottom_n], "bottom")

    return records


def export_one_scale_from_wsi(
    slide_id: str,
    scale_name: str,
    attention_h5: str,
    wsi_root: str,
    wsi_suffix: str,
    patch_level: int,
    patch_size: int,
    output_root: str,
    top_k: int,
    export_bottom_k: int,
) -> list[dict[str, object]]:
    attention, coords = load_attention_h5(attention_h5)
    if len(attention) != len(coords):
        raise ValueError(f"Length mismatch in {attention_h5}: attn={len(attention)} coords={len(coords)}")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid coordinate shape in {attention_h5}: {coords.shape}")

    wsi_path = resolve_wsi_path(wsi_root, slide_id, wsi_suffix)
    if wsi_path is None:
        raise FileNotFoundError(f"WSI missing for {slide_id} in {wsi_root}")

    slide = openslide.OpenSlide(wsi_path)
    try:
        if patch_level < 0 or patch_level >= slide.level_count:
            raise ValueError(
                f"Invalid patch level {patch_level} for {slide_id}; "
                f"slide has {slide.level_count} level(s)"
            )

        sort_idx_desc = np.argsort(-attention, kind="stable")
        records: list[dict[str, object]] = []

        def export_indices(indices: np.ndarray, split_name: str) -> None:
            split_dir = os.path.join(output_root, scale_name, split_name, slide_id)
            os.makedirs(split_dir, exist_ok=True)

            for rank, idx in enumerate(indices, start=1):
                x, y = map(int, coords[idx, :2])
                score = float(attention[idx])
                patch = slide.read_region(
                    (x, y),
                    int(patch_level),
                    (int(patch_size), int(patch_size)),
                ).convert("RGB")

                # Some SVS files contain a multi-megabyte ICC payload. Pillow
                # otherwise copies it into every tiny PNG, producing ~9 MB
                # patches and eventually exhausting storage quota.
                patch.info.clear()

                src_name = f"{slide_id}_{x}_{y}.png"
                dst_name = f"rank_{rank:03d}_score_{score:.6f}_{src_name}"
                dst_path = os.path.join(split_dir, dst_name)
                patch.save(dst_path, format="PNG", icc_profile=None)

                records.append(
                    {
                        "slide_id": slide_id,
                        "scale": scale_name,
                        "split": split_name,
                        "rank": rank,
                        "attention": score,
                        "x": x,
                        "y": y,
                        "source_mode": "wsi",
                        "source_path": wsi_path,
                        "patch_level": int(patch_level),
                        "patch_size": int(patch_size),
                        "src_path": wsi_path,
                        "dst_path": dst_path,
                    }
                )

        top_n = min(top_k, len(sort_idx_desc))
        export_indices(sort_idx_desc[:top_n], "top")

        if export_bottom_k > 0:
            bottom_n = min(export_bottom_k, len(sort_idx_desc))
            export_indices(sort_idx_desc[::-1][:bottom_n], "bottom")

        return records
    finally:
        slide.close()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    slide_ids = load_slide_ids(args)

    manifest_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, str]] = []

    using_wsi = args.wsi_root is not None
    scale_specs = [
        ("5x", "_5x_attention.h5", args.patches_5x_dir, args.patch_level_5x),
        ("20x", "_20x_attention.h5", args.patches_20x_dir, args.patch_level_20x),
    ]
    for slide_index, slide_id in enumerate(slide_ids, start=1):
        for scale_name, suffix, patches_root, patch_level in scale_specs:
            attention_h5 = os.path.join(args.attention_h5_dir, f"{slide_id}{suffix}")
            if not os.path.isfile(attention_h5):
                failed_rows.append({"slide_id": slide_id, "scale": scale_name, "reason": "attention_h5_missing"})
                continue
            try:
                if using_wsi:
                    manifest_rows.extend(
                        export_one_scale_from_wsi(
                            slide_id=slide_id,
                            scale_name=scale_name,
                            attention_h5=attention_h5,
                            wsi_root=args.wsi_root,
                            wsi_suffix=args.wsi_suffix,
                            patch_level=patch_level,
                            patch_size=args.patch_size,
                            output_root=args.output_dir,
                            top_k=args.top_k,
                            export_bottom_k=args.export_bottom_k,
                        )
                    )
                else:
                    manifest_rows.extend(
                        export_one_scale_from_patch_dir(
                            slide_id=slide_id,
                            scale_name=scale_name,
                            attention_h5=attention_h5,
                            patches_root=patches_root,
                            output_root=args.output_dir,
                            top_k=args.top_k,
                            export_bottom_k=args.export_bottom_k,
                            copy_mode=args.copy_mode,
                        )
                    )
            except Exception as exc:
                failed_rows.append({"slide_id": slide_id, "scale": scale_name, "reason": str(exc)})

        if slide_index % 20 == 0 or slide_index == len(slide_ids):
            print(f"Processed slides: {slide_index}/{len(slide_ids)}")

    pd.DataFrame(manifest_rows).to_csv(os.path.join(args.output_dir, "top_patches_manifest.csv"), index=False)
    pd.DataFrame(failed_rows, columns=["slide_id", "scale", "reason"]).to_csv(
        os.path.join(args.output_dir, "failed_exports.csv"), index=False
    )

    print(f"Saved top patches to: {args.output_dir}")
    print(f"Exported rows: {len(manifest_rows)}")
    print(f"Failed rows: {len(failed_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
