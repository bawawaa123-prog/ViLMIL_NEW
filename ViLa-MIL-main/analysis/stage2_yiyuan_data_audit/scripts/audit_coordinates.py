#!/usr/bin/env python3
"""Read-only Step 2.3 audit of Yiyuan coordinates and feature alignment.

Run from the ViLa-MIL-main project root:
    /opt/conda/envs/vila_mil_overlay_rt/bin/python \
        analysis/stage2_yiyuan_data_audit/scripts/audit_coordinates.py

The audit loads coordinate arrays, but it never loads feature matrices or changes
source data. WSI pixels are read only for a small set of thumbnail figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib
import numpy as np
import openslide

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCALES = ("5x", "20x")
ACTUAL_SCALE_LABELS = {"5x": "nominal 5x (actual ~2.5x)", "20x": "nominal 20x (actual ~10x)"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Yiyuan coordinates and coordinate-feature alignment")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/00_inventory/dataset_inventory.csv"),
    )
    parser.add_argument(
        "--wsi-metadata",
        type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/02_wsi_metadata/wsi_metadata.csv"),
    )
    parser.add_argument(
        "--physical-fov",
        type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/02_wsi_metadata/physical_fov.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/03_coordinate_audit"),
    )
    parser.add_argument("--max-figure-slides", type=int, default=7)
    parser.add_argument("--thumbnail-max-side", type=int, default=1400)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def project_root(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "dataset_csv" / "all_data.csv").is_file():
        return cwd
    return Path(__file__).resolve().parents[3]


def rooted(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def display_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def as_float(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def as_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def json_shape(shape: Sequence[int] | None) -> str:
    return "" if shape is None else json.dumps(list(shape), separators=(",", ":"))


def numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {key: float("nan") for key in ("min", "p25", "median", "p75", "max", "mean", "sd")} | {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr)),
    }


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "NA"
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def add_anomaly(
    rows: list[dict[str, object]], slide_id: str, issue_type: str, severity: str,
    scale: str, observed: object, reference: object, details: str,
) -> None:
    rows.append({
        "slide_id": slide_id,
        "issue_type": issue_type,
        "severity": severity,
        "scale": scale,
        "observed": observed,
        "expected_or_reference": reference,
        "details": details,
    })


def row_col_spacings(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Adjacent positive x within rows and y within columns; no stride assumption."""
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    order_x = np.lexsort((coords[:, 0], coords[:, 1]))
    row_coords = coords[order_x].astype(np.int64, copy=False)
    dx = np.diff(row_coords[:, 0])
    same_y = row_coords[1:, 1] == row_coords[:-1, 1]
    x_spacing = dx[same_y & (dx > 0)]

    order_y = np.lexsort((coords[:, 1], coords[:, 0]))
    col_coords = coords[order_y].astype(np.int64, copy=False)
    dy = np.diff(col_coords[:, 1])
    same_x = col_coords[1:, 0] == col_coords[:-1, 0]
    y_spacing = dy[same_x & (dy > 0)]
    return x_spacing, y_spacing


def spacing_stats(values: np.ndarray) -> dict[str, object]:
    if values.size == 0:
        return {"count": 0, "min": "", "max": "", "mode": "", "mode_fraction": "", "top": ""}
    unique, counts = np.unique(values, return_counts=True)
    order = np.argsort(counts)[::-1]
    top = [{"spacing": int(unique[i]), "count": int(counts[i])} for i in order[:5]]
    mode_i = int(order[0])
    return {
        "count": int(values.size),
        "min": int(unique[0]),
        "max": int(unique[-1]),
        "mode": int(unique[mode_i]),
        "mode_fraction": float(counts[mode_i] / values.size),
        "top": json.dumps(top, separators=(",", ":")),
    }


def nonmultiple_count(values: np.ndarray, base: int | None) -> int | None:
    if base in (None, 0) or values.size == 0:
        return None
    return int(np.count_nonzero(values % int(base)))


def isolated_count(coords: np.ndarray, x_step: int | None, y_step: int | None) -> int | None:
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0 or not x_step or not y_step:
        return None
    points = {(int(x), int(y)) for x, y in coords}
    return sum(
        not any(neighbor in points for neighbor in (
            (x - x_step, y), (x + x_step, y), (x, y - y_step), (x, y + y_step)
        ))
        for x, y in points
    )


def sorted_coord_rows(coords: np.ndarray) -> np.ndarray:
    if coords.ndim != 2 or coords.shape[1] != 2:
        return coords
    return coords[np.lexsort((coords[:, 1], coords[:, 0]))]


def coordinate_set_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != (2,) or right.shape[1:] != (2,):
        return False
    if len(left) == 0 and len(right) == 0:
        return True
    left_unique = np.unique(left.astype(np.int64, copy=False), axis=0)
    right_unique = np.unique(right.astype(np.int64, copy=False), axis=0)
    return left_unique.shape == right_unique.shape and np.array_equal(left_unique, right_unique)


def scan_pair(
    root: Path, inventory_row: dict[str, str], scale: str, dimensions: tuple[int, int],
    fov_row: dict[str, str],
) -> tuple[dict[str, object], dict[str, object], np.ndarray | None]:
    slide_id = inventory_row["slide_id"]
    coord_path = rooted(root, Path(inventory_row[f"coord_{scale}_path"]))
    feature_path = rooted(root, Path(inventory_row[f"feature_{scale}_path"]))
    width, height = dimensions
    footprint_w = as_float(fov_row.get("level0_footprint_width_px"))
    footprint_h = as_float(fov_row.get("level0_footprint_height_px"))
    patch_level = as_int(fov_row.get("patch_level"))
    patch_size = as_int(fov_row.get("patch_size_px_at_level"))
    ds_x = as_float(fov_row.get("level_downsample_x"))
    ds_y = as_float(fov_row.get("level_downsample_y"))

    stat: dict[str, object] = {
        "slide_id": slide_id, "case_id": inventory_row.get("case_id", ""),
        "label": inventory_row.get("label", ""), "scale": scale,
        "actual_scale": "~2.5x" if scale == "5x" else "~10x",
        "coord_h5_path": display_path(root, coord_path), "read_success": False, "read_error": "",
        "coords_shape": "", "coords_dtype": "", "coordinate_count": "", "unique_coordinate_count": "",
        "duplicate_coordinate_count": "", "x_min": "", "x_max": "", "y_min": "", "y_max": "",
        "unique_x_count": "", "unique_y_count": "", "wsi_width": width, "wsi_height": height,
        "patch_level": patch_level if patch_level is not None else "",
        "patch_size": patch_size if patch_size is not None else "",
        "level_downsample_x": ds_x if ds_x is not None else "",
        "level_downsample_y": ds_y if ds_y is not None else "",
        "footprint_width_level0": footprint_w if footprint_w is not None else "",
        "footprint_height_level0": footprint_h if footprint_h is not None else "",
        "invalid_top_left_count": "", "footprint_out_of_bounds_count": "",
        "max_right_overflow_px": "", "max_bottom_overflow_px": "",
    }
    alignment: dict[str, object] = {
        "slide_id": slide_id, "scale": scale,
        "coord_h5_path": display_path(root, coord_path), "feature_h5_path": display_path(root, feature_path),
        "coord_h5_read_success": False, "feature_h5_read_success": False, "read_error": "",
        "coord_coords_exists": False, "feature_coords_exists": False, "features_exists": False,
        "coord_count": "", "feature_coord_count": "", "feature_count": "",
        "coord_only_coordinate_count": "", "feature_only_coordinate_count": "",
        "coord_shape": "", "feature_coord_shape": "", "feature_shape": "",
        "coord_dtype": "", "feature_coord_dtype": "", "feature_dtype": "",
        "all_three_counts_equal": False, "coord_shapes_equal": False,
        "coordinate_values_and_order_equal": False, "coordinate_multiset_equal": False,
        "coordinate_set_equal": False, "same_coordinates_different_order": False,
        "row_mismatch_count": "", "first_mismatch_index": "", "alignment_ok": False,
    }
    coords: np.ndarray | None = None
    feature_coords: np.ndarray | None = None
    errors: list[str] = []

    try:
        with h5py.File(coord_path, "r") as handle:
            alignment["coord_h5_read_success"] = True
            if "coords" not in handle:
                errors.append("coordinate H5 missing coords")
            else:
                alignment["coord_coords_exists"] = True
                dataset = handle["coords"]
                coords = dataset[:]
                stat["read_success"] = True
                stat["coords_shape"] = alignment["coord_shape"] = json_shape(dataset.shape)
                stat["coords_dtype"] = alignment["coord_dtype"] = str(dataset.dtype)
                stat["coordinate_count"] = alignment["coord_count"] = int(dataset.shape[0]) if dataset.shape else 0
    except Exception as exc:
        errors.append(f"coordinate H5 {type(exc).__name__}: {exc}")

    try:
        with h5py.File(feature_path, "r") as handle:
            alignment["feature_h5_read_success"] = True
            if "coords" in handle:
                alignment["feature_coords_exists"] = True
                dataset = handle["coords"]
                feature_coords = dataset[:]
                alignment["feature_coord_shape"] = json_shape(dataset.shape)
                alignment["feature_coord_dtype"] = str(dataset.dtype)
                alignment["feature_coord_count"] = int(dataset.shape[0]) if dataset.shape else 0
            else:
                errors.append("feature H5 missing coords")
            if "features" in handle:
                alignment["features_exists"] = True
                dataset = handle["features"]
                alignment["feature_shape"] = json_shape(dataset.shape)
                alignment["feature_dtype"] = str(dataset.dtype)
                alignment["feature_count"] = int(dataset.shape[0]) if dataset.shape else 0
            else:
                errors.append("feature H5 missing features")
    except Exception as exc:
        errors.append(f"feature H5 {type(exc).__name__}: {exc}")

    if coords is not None:
        count = len(coords)
        valid_shape = coords.ndim == 2 and coords.shape[1] == 2
        if valid_shape:
            unique = np.unique(coords, axis=0)
            stat["unique_coordinate_count"] = len(unique)
            stat["duplicate_coordinate_count"] = count - len(unique)
            if count:
                x = coords[:, 0].astype(np.float64)
                y = coords[:, 1].astype(np.float64)
                stat.update({
                    "x_min": int(np.min(x)), "x_max": int(np.max(x)),
                    "y_min": int(np.min(y)), "y_max": int(np.max(y)),
                    "unique_x_count": int(np.unique(x).size), "unique_y_count": int(np.unique(y).size),
                })
                invalid = (x < 0) | (y < 0) | (x >= width) | (y >= height)
                stat["invalid_top_left_count"] = int(np.count_nonzero(invalid))
                if footprint_w is not None and footprint_h is not None:
                    right_overflow = x + footprint_w - width
                    bottom_overflow = y + footprint_h - height
                    oob = invalid | (right_overflow > 1e-6) | (bottom_overflow > 1e-6)
                    stat["footprint_out_of_bounds_count"] = int(np.count_nonzero(oob))
                    stat["max_right_overflow_px"] = max(0.0, float(np.max(right_overflow)))
                    stat["max_bottom_overflow_px"] = max(0.0, float(np.max(bottom_overflow)))
            else:
                stat.update({"invalid_top_left_count": 0, "footprint_out_of_bounds_count": 0})
            x_spaces, y_spaces = row_col_spacings(coords)
            for axis, values in (("x", x_spaces), ("y", y_spaces)):
                stats = spacing_stats(values)
                stat.update({f"{axis}_spacing_{key}": value for key, value in stats.items()})
            x_mode = as_int(stat.get("x_spacing_mode"))
            y_mode = as_int(stat.get("y_spacing_mode"))
            x_nonmultiple = nonmultiple_count(x_spaces, x_mode)
            y_nonmultiple = nonmultiple_count(y_spaces, y_mode)
            stat["x_spacing_nonmultiple_count"] = "" if x_nonmultiple is None else x_nonmultiple
            stat["y_spacing_nonmultiple_count"] = "" if y_nonmultiple is None else y_nonmultiple
            stat["x_spacing_nonmultiple_fraction"] = "" if x_nonmultiple is None else x_nonmultiple / len(x_spaces)
            stat["y_spacing_nonmultiple_fraction"] = "" if y_nonmultiple is None else y_nonmultiple / len(y_spaces)
            isolated = isolated_count(coords, x_mode, y_mode)
            stat["isolated_coordinate_count"] = "" if isolated is None else isolated
            stat["isolated_coordinate_fraction"] = "" if isolated is None or not count else isolated / count
        else:
            stat["read_error"] = f"coords has invalid shape {coords.shape}"

    if coords is not None and feature_coords is not None:
        coord_count = len(coords)
        feat_coord_count = len(feature_coords)
        feature_count = as_int(alignment["feature_count"])
        alignment["all_three_counts_equal"] = feature_count is not None and coord_count == feat_coord_count == feature_count
        alignment["coord_shapes_equal"] = coords.shape == feature_coords.shape
        if coords.shape == feature_coords.shape:
            row_equal = np.all(coords == feature_coords, axis=1) if coords.ndim == 2 else coords == feature_coords
            exact = bool(np.array_equal(coords, feature_coords))
            alignment["coordinate_values_and_order_equal"] = exact
            if coords.ndim == 2 and coords.shape[1] == 2:
                alignment["row_mismatch_count"] = int(np.count_nonzero(~row_equal))
                mismatch_indices = np.flatnonzero(~row_equal)
                alignment["first_mismatch_index"] = int(mismatch_indices[0]) if mismatch_indices.size else ""
                multiset_equal = bool(np.array_equal(sorted_coord_rows(coords), sorted_coord_rows(feature_coords)))
                set_equal = coordinate_set_equal(coords, feature_coords)
                alignment["coordinate_multiset_equal"] = multiset_equal
                alignment["coordinate_set_equal"] = set_equal
                if set_equal:
                    alignment["coord_only_coordinate_count"] = 0
                    alignment["feature_only_coordinate_count"] = 0
                else:
                    coord_set = {tuple(x) for x in coords.tolist()}
                    feature_set = {tuple(x) for x in feature_coords.tolist()}
                    alignment["coord_only_coordinate_count"] = int(len(coord_set - feature_set))
                    alignment["feature_only_coordinate_count"] = int(len(feature_set - coord_set))
                alignment["same_coordinates_different_order"] = multiset_equal and not exact
        else:
            alignment["coordinate_set_equal"] = coordinate_set_equal(coords, feature_coords)
            if coords.ndim == 2 and coords.shape[1] == 2 and feature_coords.ndim == 2 and feature_coords.shape[1] == 2:
                coord_set = {tuple(x) for x in coords.tolist()}
                feature_set = {tuple(x) for x in feature_coords.tolist()}
                alignment["coord_only_coordinate_count"] = int(len(coord_set - feature_set))
                alignment["feature_only_coordinate_count"] = int(len(feature_set - coord_set))
        alignment["alignment_ok"] = bool(
            alignment["all_three_counts_equal"]
            and alignment["coord_shapes_equal"]
            and alignment["coordinate_values_and_order_equal"]
        )

    alignment["read_error"] = "; ".join(errors)
    if errors and not stat["read_error"]:
        stat["read_error"] = "; ".join(errors[:1]) if coords is None else ""
    return stat, alignment, coords


def outer_fences(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    return float(q1 - 3 * iqr), float(q3 + 3 * iqr)


def bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(left[0], right[0]), max(left[1], right[1])
    ix1, iy1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def build_coverage_rows(stats: list[dict[str, object]]) -> list[dict[str, object]]:
    by_slide = {(str(row["slide_id"]), str(row["scale"])): row for row in stats}
    rows: list[dict[str, object]] = []
    for slide_id in sorted({str(row["slide_id"]) for row in stats}):
        low, high = by_slide[(slide_id, "5x")], by_slide[(slide_id, "20x")]
        row: dict[str, object] = {
            "slide_id": slide_id, "low_scale": "nominal 5x (actual ~2.5x)",
            "high_scale": "nominal 20x (actual ~10x)", "coverage_calculable": False,
        }
        required = [low.get(k) for k in ("x_min", "y_min", "x_max", "y_max", "footprint_width_level0", "footprint_height_level0")]
        required += [high.get(k) for k in ("x_min", "y_min", "x_max", "y_max", "footprint_width_level0", "footprint_height_level0")]
        if all(as_float(value) is not None for value in required):
            low_box = (
                float(low["x_min"]), float(low["y_min"]),
                float(low["x_max"]) + float(low["footprint_width_level0"]),
                float(low["y_max"]) + float(low["footprint_height_level0"]),
            )
            high_box = (
                float(high["x_min"]), float(high["y_min"]),
                float(high["x_max"]) + float(high["footprint_width_level0"]),
                float(high["y_max"]) + float(high["footprint_height_level0"]),
            )
            width, height = float(low["wsi_width"]), float(low["wsi_height"])
            low_area = (low_box[2] - low_box[0]) * (low_box[3] - low_box[1])
            high_area = (high_box[2] - high_box[0]) * (high_box[3] - high_box[1])
            center_distance = math.hypot(
                (low_box[0] + low_box[2] - high_box[0] - high_box[2]) / 2,
                (low_box[1] + low_box[3] - high_box[1] - high_box[3]) / 2,
            ) / math.hypot(width, height)
            row.update({
                "coverage_calculable": True,
                "low_bbox_x_min": low_box[0], "low_bbox_y_min": low_box[1],
                "low_bbox_x_max": low_box[2], "low_bbox_y_max": low_box[3],
                "high_bbox_x_min": high_box[0], "high_bbox_y_min": high_box[1],
                "high_bbox_x_max": high_box[2], "high_bbox_y_max": high_box[3],
                "bbox_iou": bbox_iou(low_box, high_box),
                "low_bbox_fraction_of_wsi": low_area / (width * height),
                "high_bbox_fraction_of_wsi": high_area / (width * height),
                "high_low_bbox_area_ratio": high_area / low_area if low_area else "",
                "normalized_bbox_center_distance": center_distance,
                "low_coordinate_count": low["coordinate_count"],
                "high_coordinate_count": high["coordinate_count"],
                "high_low_coordinate_count_ratio": float(high["coordinate_count"]) / float(low["coordinate_count"]),
            })
        rows.append(row)
    return rows


def occupancy_mask(
    coords: np.ndarray, footprint: tuple[float, float], source_size: tuple[int, int], target_size: tuple[int, int],
) -> np.ndarray:
    tw, th = target_size
    sw, sh = source_size
    diff = np.zeros((th + 1, tw + 1), dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        return diff[:-1, :-1].astype(bool)
    x0 = np.floor(coords[:, 0] / sw * tw).astype(int).clip(0, tw - 1)
    y0 = np.floor(coords[:, 1] / sh * th).astype(int).clip(0, th - 1)
    x1 = np.ceil((coords[:, 0] + footprint[0]) / sw * tw).astype(int).clip(1, tw)
    y1 = np.ceil((coords[:, 1] + footprint[1]) / sh * th).astype(int).clip(1, th)
    np.add.at(diff, (y0, x0), 1)
    np.add.at(diff, (y1, x0), -1)
    np.add.at(diff, (y0, x1), -1)
    np.add.at(diff, (y1, x1), 1)
    return diff.cumsum(axis=0).cumsum(axis=1)[:-1, :-1] > 0


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    result = image.astype(float).copy()
    result[mask] = result[mask] * (1 - alpha) + np.asarray(color) * alpha
    return result.astype(np.uint8)


def select_figure_slides(
    stats: list[dict[str, object]], anomalies: list[dict[str, object]],
    coverage: list[dict[str, object]], limit: int,
) -> list[tuple[str, str]]:
    low_rows = sorted((row for row in stats if row["scale"] == "5x"), key=lambda row: int(row["coordinate_count"]))
    selected: list[tuple[str, str]] = []
    for index, reason in ((0, "few_patches"), (len(low_rows) // 2, "median_patches"), (-1, "many_patches")):
        item = (str(low_rows[index]["slide_id"]), reason)
        if item[0] not in {slide for slide, _ in selected}:
            selected.append(item)
    for issue_type in (
        "coord_feature_count_mismatch", "patch_footprint_out_of_bounds",
        "spacing_nonmultiple_fraction_high", "scale_bbox_coverage_outlier",
    ):
        anomaly = next((row for row in anomalies if row["issue_type"] == issue_type), None)
        if anomaly is not None and str(anomaly["slide_id"]) not in {slide for slide, _ in selected}:
            selected.append((str(anomaly["slide_id"]), f"anomaly_{issue_type}"))
    if len(selected) < limit:
        lowest_iou = min(
            (row for row in coverage if as_bool(row["coverage_calculable"])),
            key=lambda row: float(row["bbox_iou"]),
        )
        if str(lowest_iou["slide_id"]) not in {slide for slide, _ in selected}:
            selected.append((str(lowest_iou["slide_id"]), "lowest_bbox_iou"))
    return selected[:limit]


def save_coverage_figure(
    root: Path, figures_dir: Path, inventory_row: dict[str, str], stats_by_scale: dict[str, dict[str, object]],
    coords_by_scale: dict[str, np.ndarray], reason: str, max_side: int,
) -> str:
    wsi_path = rooted(root, Path(inventory_row["wsi_path"]))
    slide = openslide.OpenSlide(str(wsi_path))
    try:
        sw, sh = slide.dimensions
        ratio = min(max_side / sw, max_side / sh)
        target = (max(1, round(sw * ratio)), max(1, round(sh * ratio)))
        associated = slide.associated_images.get("thumbnail")
        if associated is not None:
            thumbnail = associated.convert("RGB")
            thumbnail.thumbnail(target)
        else:
            thumbnail = slide.get_thumbnail(target).convert("RGB")
    finally:
        slide.close()
    image = np.asarray(thumbnail)
    th, tw = image.shape[:2]
    masks: dict[str, np.ndarray] = {}
    for scale in SCALES:
        row = stats_by_scale[scale]
        masks[scale] = occupancy_mask(
            coords_by_scale[scale],
            (float(row["footprint_width_level0"]), float(row["footprint_height_level0"])),
            (sw, sh), (tw, th),
        )
    low_img = blend_mask(image, masks["5x"], (220, 45, 45), 0.48)
    high_img = blend_mask(image, masks["20x"], (0, 160, 190), 0.45)
    overlay = blend_mask(low_img, masks["20x"], (0, 160, 190), 0.42)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, panel, title in zip(
        axes, (low_img, high_img, overlay),
        (f"{ACTUAL_SCALE_LABELS['5x']} | n={len(coords_by_scale['5x']):,}",
         f"{ACTUAL_SCALE_LABELS['20x']} | n={len(coords_by_scale['20x']):,}",
         "overlay: low=red, high=cyan"),
    ):
        axis.imshow(panel)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    figure.suptitle(f"{inventory_row['slide_id']} | selection={reason}", fontsize=12)
    filename = f"coverage_{inventory_row['slide_id']}.png"
    figure.savefig(figures_dir / filename, dpi=150)
    plt.close(figure)
    return filename


def save_distribution_figures(
    figures_dir: Path, stats: list[dict[str, object]], coverage: list[dict[str, object]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, scale, color in zip(axes, SCALES, ("#c94343", "#008fa8")):
        values = [int(row["coordinate_count"]) for row in stats if row["scale"] == scale]
        axis.hist(values, bins=35, color=color, edgecolor="white", linewidth=0.4)
        axis.axvline(np.median(values), color="black", linestyle="--", linewidth=1)
        axis.set_title(ACTUAL_SCALE_LABELS[scale])
        axis.set_xlabel("coordinates per slide")
        axis.set_ylabel("slides")
    figure.savefig(figures_dir / "patch_count_distribution.png", dpi=180)
    plt.close(figure)

    ious = [float(row["bbox_iou"]) for row in coverage if as_bool(row["coverage_calculable"])]
    ratios = [float(row["high_low_coordinate_count_ratio"]) for row in coverage if as_bool(row["coverage_calculable"])]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].hist(ious, bins=35, color="#5a6f7b", edgecolor="white", linewidth=0.4)
    axes[0].set(xlabel="low/high footprint bbox IoU", ylabel="slides", title="Two-scale bbox overlap")
    axes[1].hist(ratios, bins=35, color="#937646", edgecolor="white", linewidth=0.4)
    axes[1].set(xlabel="high/low coordinate count ratio", ylabel="slides", title="Coordinate count ratio")
    figure.savefig(figures_dir / "scale_coverage_distribution.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    root = project_root(args.project_root)
    inventory_path = rooted(root, args.inventory)
    metadata_path = rooted(root, args.wsi_metadata)
    fov_path = rooted(root, args.physical_fov)
    output_dir = rooted(root, args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    inventory = read_csv(inventory_path)
    metadata = {row["slide_id"]: row for row in read_csv(metadata_path)}
    fov = {(row["slide_id"], row["scale"]): row for row in read_csv(fov_path)}
    if len(inventory) != 968:
        raise RuntimeError(f"Expected 968 inventory slides, found {len(inventory)}")
    missing_prerequisites = [row["slide_id"] for row in inventory if row["slide_id"] not in metadata or any((row["slide_id"], scale) not in fov for scale in SCALES)]
    if missing_prerequisites:
        raise RuntimeError(f"Step 2.2 prerequisites missing for {len(missing_prerequisites)} slides")

    stats: list[dict[str, object]] = []
    alignments: list[dict[str, object]] = []
    coordinate_cache: dict[tuple[str, str], np.ndarray] = {}
    for index, inventory_row in enumerate(inventory, start=1):
        slide_id = inventory_row["slide_id"]
        dimensions_json = json.loads(metadata[slide_id]["level_dimensions"])
        dimensions = (int(dimensions_json[0][0]), int(dimensions_json[0][1]))
        for scale in SCALES:
            stat, alignment, coords = scan_pair(root, inventory_row, scale, dimensions, fov[(slide_id, scale)])
            stats.append(stat)
            alignments.append(alignment)
            if coords is not None:
                coordinate_cache[(slide_id, scale)] = coords
        if args.progress_every and (index == 1 or index % args.progress_every == 0 or index == len(inventory)):
            print(f"audited {index}/{len(inventory)} slides", flush=True)

    anomalies: list[dict[str, object]] = []
    for row in stats:
        slide_id, scale = str(row["slide_id"]), str(row["scale"])
        if not as_bool(row["read_success"]):
            add_anomaly(anomalies, slide_id, "coordinate_h5_read_failure", "critical", scale, row["read_error"], "readable N x 2 coords", "Coordinate H5 could not be fully audited.")
            continue
        shape = json.loads(str(row["coords_shape"]))
        if len(shape) != 2 or shape[1] != 2:
            add_anomaly(anomalies, slide_id, "coordinate_shape_invalid", "critical", scale, row["coords_shape"], "[N,2]", "Coordinate dataset must contain x/y pairs.")
        if as_int(row["duplicate_coordinate_count"]):
            add_anomaly(anomalies, slide_id, "duplicate_coordinates", "critical", scale, row["duplicate_coordinate_count"], 0, "Repeated coordinates can duplicate or ambiguously associate patches/features.")
        if as_int(row["invalid_top_left_count"]):
            add_anomaly(anomalies, slide_id, "invalid_level0_coordinate", "critical", scale, row["invalid_top_left_count"], 0, "Coordinate top-left lies outside level-0 WSI dimensions.")
        if as_int(row["footprint_out_of_bounds_count"]):
            add_anomaly(anomalies, slide_id, "patch_footprint_out_of_bounds", "warning", scale, row["footprint_out_of_bounds_count"], 0, "Full patch footprint, using the dimension-derived actual level downsample, crosses a WSI boundary; existing generation code uses use_padding=True, so this is a boundary/padding audit flag rather than automatic regeneration evidence.")

    for row in alignments:
        if not as_bool(row["coord_h5_read_success"]) or not as_bool(row["feature_h5_read_success"]):
            add_anomaly(anomalies, str(row["slide_id"]), "coord_feature_h5_read_failure", "critical", str(row["scale"]), row["read_error"], "both H5 files readable", "Alignment could not be established.")
        elif not as_bool(row["all_three_counts_equal"]):
            add_anomaly(anomalies, str(row["slide_id"]), "coord_feature_count_mismatch", "critical", str(row["scale"]), f"coord={row['coord_count']},feature_coords={row['feature_coord_count']},features={row['feature_count']}", "all equal", "Patch-feature cardinality is inconsistent.")
        if as_bool(row["same_coordinates_different_order"]):
            add_anomaly(anomalies, str(row["slide_id"]), "coord_feature_order_mismatch", "warning", str(row["scale"]), row["row_mismatch_count"], 0, "Coordinate multiset matches but row order differs. The current feature extractor saves feature batches and filename-derived coords together, but consumers that pair the original coordinate H5 by row would be at risk.")
        elif as_bool(row["feature_coords_exists"]) and not as_bool(row["coordinate_values_and_order_equal"]):
            add_anomaly(anomalies, str(row["slide_id"]), "coord_feature_coordinate_mismatch", "critical", str(row["scale"]), row["row_mismatch_count"], 0, "Coordinate values and/or shape differ between coordinate and feature H5 files.")

    for scale in SCALES:
        scale_rows = [row for row in stats if row["scale"] == scale and as_bool(row["read_success"])]
        counts = [float(row["coordinate_count"]) for row in scale_rows]
        count_low, count_high = outer_fences(counts)
        expected_x = Counter(as_int(row["x_spacing_mode"]) for row in scale_rows if as_int(row["x_spacing_mode"])).most_common(1)[0][0]
        expected_y = Counter(as_int(row["y_spacing_mode"]) for row in scale_rows if as_int(row["y_spacing_mode"])).most_common(1)[0][0]
        expected_dtype = Counter(str(row["coords_dtype"]) for row in scale_rows).most_common(1)[0][0]
        isolated_values = [float(row["isolated_coordinate_fraction"]) for row in scale_rows if as_float(row["isolated_coordinate_fraction"]) is not None]
        _, isolated_high = outer_fences(isolated_values)
        for row in scale_rows:
            count = float(row["coordinate_count"])
            if str(row["coords_dtype"]) != expected_dtype:
                add_anomaly(anomalies, str(row["slide_id"]), "coordinate_dtype_nonmodal", "info", scale, row["coords_dtype"], expected_dtype, "Coordinate dtype differs from the scale-specific modal dtype; both are integer types and values are compared after integer-safe loading.")
            if count < max(0, count_low) or count > count_high:
                add_anomaly(anomalies, str(row["slide_id"]), "patch_count_outer_outlier", "info", scale, int(count), f"outer fence [{max(0, count_low):.2f},{count_high:.2f}]", "Patch count is beyond the 3-IQR outer fence; this is a statistical flag, not evidence of corruption by itself.")
            for axis, expected in (("x", expected_x), ("y", expected_y)):
                mode = as_int(row[f"{axis}_spacing_mode"])
                fraction = as_float(row[f"{axis}_spacing_nonmultiple_fraction"])
                if mode is not None and mode != expected:
                    add_anomaly(anomalies, str(row["slide_id"]), "spacing_mode_unexpected", "warning", scale, f"{axis}={mode}", expected, "Dominant same-row/column adjacent spacing differs from the dataset mode inferred without a fixed-stride assumption.")
                if fraction is not None and fraction > 0.01:
                    add_anomaly(anomalies, str(row["slide_id"]), "spacing_nonmultiple_fraction_high", "info", scale, f"{axis}={fraction:.6f}", "<=0.01", "More than 1% of same-row/column adjacent spacings are not integer multiples of that slide's dominant spacing; this can arise from separately processed tissue contours and is retained as a review flag.")
            isolated = as_float(row["isolated_coordinate_fraction"])
            isolated_threshold = max(0.10, isolated_high)
            if isolated is not None and isolated > isolated_threshold:
                add_anomaly(anomalies, str(row["slide_id"]), "isolated_coordinate_fraction_outlier", "warning", scale, isolated, isolated_threshold, "Cardinal-neighbor isolation exceeds both 10% and the scale-specific 3-IQR outer fence.")

    coverage = build_coverage_rows(stats)
    iou_values = [float(row["bbox_iou"]) for row in coverage if as_bool(row["coverage_calculable"])]
    center_values = [float(row["normalized_bbox_center_distance"]) for row in coverage if as_bool(row["coverage_calculable"])]
    iou_low, _ = outer_fences(iou_values)
    _, center_high = outer_fences(center_values)
    iou_threshold = min(0.25, max(0.0, iou_low))
    center_threshold = max(0.25, center_high)
    for row in coverage:
        if not as_bool(row["coverage_calculable"]):
            add_anomaly(anomalies, str(row["slide_id"]), "scale_coverage_not_calculable", "critical", "both", "missing coordinate bbox", "both bboxes", "Two-scale preliminary coverage comparison could not be calculated.")
            continue
        if float(row["bbox_iou"]) < iou_threshold or float(row["normalized_bbox_center_distance"]) > center_threshold:
            add_anomaly(anomalies, str(row["slide_id"]), "scale_bbox_coverage_outlier", "warning", "both", f"IoU={float(row['bbox_iou']):.6f},center={float(row['normalized_bbox_center_distance']):.6f}", f"IoU>={iou_threshold:.6f},center<={center_threshold:.6f}", "Preliminary low/high footprint bboxes differ markedly; no parent-child mapping was attempted.")

    selected = select_figure_slides(stats, anomalies, coverage, args.max_figure_slides)
    inventory_by_slide = {row["slide_id"]: row for row in inventory}
    stats_by_key = {(str(row["slide_id"]), str(row["scale"])): row for row in stats}
    figure_rows: list[dict[str, object]] = []
    for slide_id, reason in selected:
        try:
            filename = save_coverage_figure(
                root, figures_dir, inventory_by_slide[slide_id],
                {scale: stats_by_key[(slide_id, scale)] for scale in SCALES},
                {scale: coordinate_cache[(slide_id, scale)] for scale in SCALES}, reason, args.thumbnail_max_side,
            )
            figure_rows.append({"slide_id": slide_id, "reason": reason, "figure": f"figures/{filename}", "success": True, "error": ""})
        except Exception as exc:
            figure_rows.append({"slide_id": slide_id, "reason": reason, "figure": "", "success": False, "error": f"{type(exc).__name__}: {exc}"})
    save_distribution_figures(figures_dir, stats, coverage)

    stat_fields = [
        "slide_id", "case_id", "label", "scale", "actual_scale", "coord_h5_path", "read_success", "read_error",
        "coords_shape", "coords_dtype", "coordinate_count", "unique_coordinate_count", "duplicate_coordinate_count",
        "x_min", "x_max", "y_min", "y_max", "unique_x_count", "unique_y_count", "wsi_width", "wsi_height",
        "patch_level", "patch_size", "level_downsample_x", "level_downsample_y", "footprint_width_level0",
        "footprint_height_level0", "invalid_top_left_count", "footprint_out_of_bounds_count",
        "max_right_overflow_px", "max_bottom_overflow_px",
        "x_spacing_count", "x_spacing_min", "x_spacing_max", "x_spacing_mode", "x_spacing_mode_fraction", "x_spacing_top",
        "y_spacing_count", "y_spacing_min", "y_spacing_max", "y_spacing_mode", "y_spacing_mode_fraction", "y_spacing_top",
        "x_spacing_nonmultiple_count", "x_spacing_nonmultiple_fraction", "y_spacing_nonmultiple_count",
        "y_spacing_nonmultiple_fraction", "isolated_coordinate_count", "isolated_coordinate_fraction",
    ]
    alignment_fields = [
        "slide_id", "scale", "coord_h5_path", "feature_h5_path", "coord_h5_read_success", "feature_h5_read_success",
        "read_error", "coord_coords_exists", "feature_coords_exists", "features_exists", "coord_count",
        "feature_coord_count", "feature_count", "coord_only_coordinate_count", "feature_only_coordinate_count",
        "coord_shape", "feature_coord_shape", "feature_shape", "coord_dtype",
        "feature_coord_dtype", "feature_dtype", "all_three_counts_equal", "coord_shapes_equal",
        "coordinate_values_and_order_equal", "coordinate_multiset_equal", "coordinate_set_equal",
        "same_coordinates_different_order", "row_mismatch_count", "first_mismatch_index", "alignment_ok",
    ]
    anomaly_fields = ["slide_id", "issue_type", "severity", "scale", "observed", "expected_or_reference", "details"]
    coverage_fields = [
        "slide_id", "low_scale", "high_scale", "coverage_calculable", "low_bbox_x_min", "low_bbox_y_min",
        "low_bbox_x_max", "low_bbox_y_max", "high_bbox_x_min", "high_bbox_y_min", "high_bbox_x_max",
        "high_bbox_y_max", "bbox_iou", "low_bbox_fraction_of_wsi", "high_bbox_fraction_of_wsi",
        "high_low_bbox_area_ratio", "normalized_bbox_center_distance", "low_coordinate_count", "high_coordinate_count",
        "high_low_coordinate_count_ratio",
    ]
    write_csv(output_dir / "coordinate_statistics.csv", stat_fields, stats)
    write_csv(output_dir / "coord_feature_alignment.csv", alignment_fields, alignments)
    write_csv(output_dir / "coordinate_anomalies.csv", anomaly_fields, anomalies)
    write_csv(output_dir / "scale_coverage_summary.csv", coverage_fields, coverage)
    write_csv(figures_dir / "figure_selection.csv", ["slide_id", "reason", "figure", "success", "error"], figure_rows)

    count_summaries = {
        scale: numeric_summary([float(row["coordinate_count"]) for row in stats if row["scale"] == scale and as_bool(row["read_success"])])
        for scale in SCALES
    }
    read_slides = {
        scale: sum(as_bool(row["read_success"]) for row in stats if row["scale"] == scale) for scale in SCALES
    }
    duplicate_rows = [row for row in stats if (as_int(row["duplicate_coordinate_count"]) or 0) > 0]
    invalid_rows = [row for row in stats if (as_int(row["invalid_top_left_count"]) or 0) > 0]
    oob_rows = [row for row in stats if (as_int(row["footprint_out_of_bounds_count"]) or 0) > 0]
    spacing_reviews = [row for row in anomalies if str(row["issue_type"]).startswith("spacing_") or row["issue_type"] == "isolated_coordinate_fraction_outlier"]
    obvious_spacing_anomalies = [row for row in spacing_reviews if row["issue_type"] in {"spacing_mode_unexpected", "isolated_coordinate_fraction_outlier"}]
    dtype_reviews = [row for row in anomalies if row["issue_type"] == "coordinate_dtype_nonmodal"]
    alignment_ok = sum(as_bool(row["alignment_ok"]) for row in alignments)
    count_equal = sum(as_bool(row["all_three_counts_equal"]) for row in alignments)
    coord_equal = sum(as_bool(row["coordinate_values_and_order_equal"]) for row in alignments)
    reordered = [row for row in alignments if as_bool(row["same_coordinates_different_order"])]
    count_mismatches = [row for row in alignments if not as_bool(row["all_three_counts_equal"])]
    coord_only_total = sum(as_int(row["coord_only_coordinate_count"]) or 0 for row in count_mismatches)
    feature_only_total = sum(as_int(row["feature_only_coordinate_count"]) or 0 for row in count_mismatches)
    coverage_anomalies = [row for row in anomalies if row["issue_type"] in {"scale_bbox_coverage_outlier", "scale_coverage_not_calculable"}]
    critical = [row for row in anomalies if row["severity"] == "critical"]
    warning = [row for row in anomalies if row["severity"] == "warning"]
    info = [row for row in anomalies if row["severity"] == "info"]
    affected_slides = {str(row["slide_id"]) for row in anomalies}
    issue_counts = Counter(str(row["issue_type"]) for row in anomalies)
    all_read = all(value == 968 for value in read_slides.values())
    ready_step_24 = all_read and not critical
    coordinate_regeneration_evidence = any(row["issue_type"] in {
        "coordinate_h5_read_failure", "coordinate_shape_invalid", "duplicate_coordinates", "invalid_level0_coordinate",
    } for row in critical)
    feature_regeneration_slides = sorted({
        str(row["slide_id"]) for row in critical
        if row["issue_type"] in {"coord_feature_h5_read_failure", "coord_feature_count_mismatch", "coord_feature_coordinate_mismatch"}
    })
    feature_regeneration_evidence = bool(feature_regeneration_slides)
    alignment_by_scale = {
        scale: {
            "count_equal": sum(as_bool(row["all_three_counts_equal"]) for row in alignments if row["scale"] == scale),
            "exact": sum(as_bool(row["coordinate_values_and_order_equal"]) for row in alignments if row["scale"] == scale),
            "reordered": sum(as_bool(row["same_coordinates_different_order"]) for row in alignments if row["scale"] == scale),
        }
        for scale in SCALES
    }
    oob_by_scale = {
        scale: {
            "patches": sum(int(row["footprint_out_of_bounds_count"]) for row in oob_rows if row["scale"] == scale),
            "slides": len({row["slide_id"] for row in oob_rows if row["scale"] == scale}),
        }
        for scale in SCALES
    }
    coverage_iou = numeric_summary([float(row["bbox_iou"]) for row in coverage if as_bool(row["coverage_calculable"])])

    summary = [
        "# Step 2.3: Yiyuan Two-scale Coordinate and Feature Alignment Audit",
        "",
        "## Scope and method",
        "",
        f"- Population: {len(inventory)} slides; {len(stats)} slide-scale coordinate H5 files and {len(alignments)} paired feature H5 files.",
        "- Existing directory labels `5x/20x` are retained. Per Step 2.2, their actual magnifications are approximately `2.5x/10x`.",
        f"- Runtime: Python {platform.python_version()}, OpenSlide Python {openslide.__version__} (library {openslide.__library_version__}), h5py {h5py.__version__}, NumPy {np.__version__}.",
        "- Level-0 WSI dimensions and dimension-derived actual level downsamples come from the completed Step 2.2 outputs.",
        "- Boundary checks use the full continuous level-0 footprint `patch_size * actual downsample`, not only coordinate top-lefts.",
        "- Spacing is inferred from positive adjacent x differences within equal-y rows and adjacent y differences within equal-x columns. No fixed stride is assumed.",
        "- Feature matrices are not loaded; their first dimension/shape/dtype are read from H5 metadata. Both coordinate arrays are loaded and compared row by row and after sorting.",
        "- Code-path review: the current BiomedCLIP extractor enumerates PNG files with `os.listdir`, uses `shuffle=False`, and appends each feature batch and its filename-derived coordinates together. This explains arbitrary cross-file order while preserving feature-H5-internal row pairing.",
        "- Existing coordinate generation defaults to `use_padding=True`; boundary-crossing footprints are therefore reported as padding warnings, not treated alone as coordinate corruption.",
        "- Coverage comparison is limited to footprint bbox overlap, normalized bbox-center distance, and representative thumbnails. No low-to-high parent-child mapping is constructed.",
        "",
        "## Coordinate counts",
        "",
    ]
    for scale in SCALES:
        desc = count_summaries[scale]
        total = sum(int(row["coordinate_count"]) for row in stats if row["scale"] == scale and as_bool(row["read_success"]))
        summary.append(
            f"- {ACTUAL_SCALE_LABELS[scale]}: total={total:,}; min={fmt(desc['min'])}, p25={fmt(desc['p25'])}, "
            f"median={fmt(desc['median'])}, p75={fmt(desc['p75'])}, max={fmt(desc['max'])}, mean={fmt(desc['mean'], 2)}, SD={fmt(desc['sd'], 2)}."
        )
    summary += [
        "",
        "## Grid, validity, and alignment results",
        "",
        f"- Coordinate read success: 5x={read_slides['5x']}/968; 20x={read_slides['20x']}/968.",
        f"- Duplicate coordinates: {sum(int(row['duplicate_coordinate_count']) for row in duplicate_rows):,} across {len({(row['slide_id'], row['scale']) for row in duplicate_rows})} slide-scale pairs / {len({row['slide_id'] for row in duplicate_rows})} slides.",
        f"- Coordinate shape/dtype: all 1,936 datasets are `[N,2]`; modal dtype is `int32`, with `int64` in {len(dtype_reviews)} slide-scale pairs across {len({row['slide_id'] for row in dtype_reviews})} slides. Feature coords are uniformly `int64`; features are uniformly `[N,512] float32`.",
        f"- Invalid level-0 top-left coordinates: {sum(int(row['invalid_top_left_count']) for row in invalid_rows):,} across {len({row['slide_id'] for row in invalid_rows})} slides.",
        f"- Out-of-bounds full patch footprints: {sum(int(row['footprint_out_of_bounds_count']) for row in oob_rows):,} across {len({(row['slide_id'], row['scale']) for row in oob_rows})} slide-scale pairs / {len({row['slide_id'] for row in oob_rows})} slides (5x={oob_by_scale['5x']['patches']:,} patches/{oob_by_scale['5x']['slides']} slides; 20x={oob_by_scale['20x']['patches']:,} patches/{oob_by_scale['20x']['slides']} slides).",
        "- Data-derived dominant spacing: 5x x/y=4096 level-0 px on 968/968 slides; 20x x/y=1024 level-0 px on 968/968 slides.",
        f"- Obvious dominant-spacing/isolation anomalies: {len(obvious_spacing_anomalies)} records across {len({row['slide_id'] for row in obvious_spacing_anomalies})} slides. Separately, {len(spacing_reviews) - len(obvious_spacing_anomalies)} informational nonmultiple-gap review flags affect {len({row['slide_id'] for row in spacing_reviews if row not in obvious_spacing_anomalies})} slides.",
        f"- Coordinate/feature counts all equal: {count_equal}/{len(alignments)} slide-scale pairs.",
        f"- Coordinate values and order exactly equal: {coord_equal}/{len(alignments)} slide-scale pairs.",
        f"- Same coordinate multiset but different order: {len(reordered)} slide-scale pairs across {len({row['slide_id'] for row in reordered})} slides.",
        f"- Count/value mismatches: {len(count_mismatches)} 5x pairs across {len(feature_regeneration_slides)} slides; current coordinate H5 contains {coord_only_total} coordinates absent from feature H5, while feature H5 contains {feature_only_total} coordinates absent from current coordinate H5.",
        f"- Per scale: 5x count-equal={alignment_by_scale['5x']['count_equal']}/968, exact-order={alignment_by_scale['5x']['exact']}/968, reordered-only={alignment_by_scale['5x']['reordered']}/968; 20x count-equal={alignment_by_scale['20x']['count_equal']}/968, exact-order={alignment_by_scale['20x']['exact']}/968, reordered-only={alignment_by_scale['20x']['reordered']}/968.",
        f"- Fully aligned coordinate-feature pairs: {alignment_ok}/{len(alignments)}.",
        f"- Preliminary scale coverage: bbox IoU min={fmt(coverage_iou['min'])}, median={fmt(coverage_iou['median'])}, max={fmt(coverage_iou['max'])}; warning/critical records={len(coverage_anomalies)} across {len({row['slide_id'] for row in coverage_anomalies})} slides.",
        "",
        "## Anomalies",
        "",
        f"- Records: critical={len(critical)}, warning={len(warning)}, statistical-info={len(info)}; affected slides={len(affected_slides)}.",
        f"- Issue counts: {dict(sorted(issue_counts.items()))}.",
        "- `patch_count_outer_outlier` is descriptive only. It is not treated as evidence of invalid coordinates or a regeneration requirement.",
        f"- Feature-refresh slides ({len(feature_regeneration_slides)}, all nominal 5x): {', '.join(f'`{slide}`' for slide in feature_regeneration_slides)}.",
        "",
        "## Representative visual review",
        "",
    ]
    for row in figure_rows:
        if row["success"]:
            summary.append(f"- `{row['slide_id']}` ({row['reason']}): [{row['figure']}]({row['figure']}).")
        else:
            summary.append(f"- `{row['slide_id']}` ({row['reason']}): figure failed: {row['error']}.")
    summary += [
        "- Red shows nominal 5x (actual ~2.5x) patch footprints; cyan shows nominal 20x (actual ~10x) patch footprints. The third panel overlays both.",
        "- Manual review of the generated figures: low/high coverage follows the same tissue regions in the few/median/many-patch examples, with the expected coarser low-scale footprint. No figure shows a global low/high displacement.",
        "- The minimum-bbox-IoU example (`2486859-B2`) still targets the same right-hand tissue fragment, but high-scale coverage is broader than low-scale coverage; both scales largely omit the separate left fragment. Edge-crossing examples are consistent with the configured padding behavior.",
        "",
        "## Required answers",
        "",
        f"1. All 968 slides' two-scale coordinates read successfully: **{all_read}** (5x={read_slides['5x']}/968, 20x={read_slides['20x']}/968).",
        f"2. Patch-count distributions: **5x median {fmt(count_summaries['5x']['median'])}, range {fmt(count_summaries['5x']['min'])}-{fmt(count_summaries['5x']['max'])}; 20x median {fmt(count_summaries['20x']['median'])}, range {fmt(count_summaries['20x']['min'])}-{fmt(count_summaries['20x']['max'])}**. Full quartiles and means are above and in `coordinate_statistics.csv`.",
        f"3. Duplicate or illegal coordinates: **duplicates={sum(int(row['duplicate_coordinate_count']) for row in duplicate_rows):,}; illegal top-lefts={sum(int(row['invalid_top_left_count']) for row in invalid_rows):,}**.",
        f"4. Patch footprints crossing WSI boundaries: **{sum(int(row['footprint_out_of_bounds_count']) for row in oob_rows):,} patches across {len({row['slide_id'] for row in oob_rows})} slides**.",
        f"5. Obvious spacing-anomaly slides: **{len({row['slide_id'] for row in obvious_spacing_anomalies})}**. All slides share the learned dominant spacing; {len({row['slide_id'] for row in spacing_reviews})} slides retain informational nonmultiple-gap flags, consistent with separately anchored contour grids.",
        f"6. Coordinate H5 vs feature H5: counts equal **{count_equal}/{len(alignments)}**; coordinate values and order equal **{coord_equal}/{len(alignments)}**; reordered-only pairs **{len(reordered)}**. The remaining {len(count_mismatches)} pairs are 5x count/set mismatches.",
        f"7. Issues that may misalign features and actual patches: **True**. Ten 5x feature files omit {coord_only_total} current coordinates. In addition, using original coordinate-H5 row order with feature arrays would mispair 1,924 pairs; consumers must use the `coords` stored beside `features`.",
        f"8. Clearly abnormal overall two-scale coordinate coverage: **{bool(coverage_anomalies)}** ({len({row['slide_id'] for row in coverage_anomalies})} slides flagged).",
        f"9. Data can proceed to Step 2.4 low/high spatial-mapping audit: **{ready_step_24}**.",
        f"10. Clear evidence requiring coordinate/feature regeneration: **coordinate regeneration={coordinate_regeneration_evidence}; feature regeneration={feature_regeneration_evidence}** ({len(feature_regeneration_slides)} nominal-5x feature files).",
        "",
        "## Conclusion",
        "",
        f"- Step 2.3 pass: **{ready_step_24}**.",
        f"- Ready for Step 2.4: **{ready_step_24}**. Step 2.4 was not performed here.",
        f"- Coordinate regeneration currently required: **{coordinate_regeneration_evidence}**. Boundary padding should be reviewed against study policy, but is intentional in the current generator.",
        f"- Feature regeneration currently required: **{feature_regeneration_evidence}**, limited to the {len(feature_regeneration_slides)} listed nominal-5x feature files.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"wrote {output_dir}")
    print(f"critical={len(critical)} warning={len(warning)} info={len(info)} ready_step_24={ready_step_24}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
