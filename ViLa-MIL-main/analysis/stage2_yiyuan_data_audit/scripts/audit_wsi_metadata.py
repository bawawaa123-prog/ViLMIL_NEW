#!/usr/bin/env python3
"""Read-only Step 2.2 audit of Yiyuan WSI scale and coordinate H5 metadata.

Run from the ViLa-MIL-main root with the environment that provides OpenSlide:
    /opt/conda/envs/vila_mil_overlay_rt/bin/python \
        analysis/stage2_yiyuan_data_audit/scripts/audit_wsi_metadata.py

Coordinates are not loaded into memory. Only dataset shape/dtype and attributes
are read from H5 files; WSI pixel regions are never decoded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib
import numpy as np
import openslide

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


NOMINAL_MAGNIFICATIONS = {"5x": 5.0, "20x": 20.0}
SCALES = ("5x", "20x")
RELATIVE_TOLERANCE = 0.05
H5_WSI_DOWNSAMPLE_TOLERANCE = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Yiyuan WSI metadata and physical scale")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=Path("dataset_csv/all_data.csv"))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/00_inventory/dataset_inventory.csv"),
    )
    parser.add_argument("--wsi-dir", type=Path, default=Path("data/yiyuan/wsi"))
    parser.add_argument("--coord-5x-dir", type=Path, default=Path("data/yiyuan/patches_coords_5x/patches_256"))
    parser.add_argument("--coord-20x-dir", type=Path, default=Path("data/yiyuan/patches_coords_20x/patches_256"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/02_wsi_metadata"),
    )
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def resolve_project_root(value: Path | None) -> Path:
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
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    return fields, rows


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def to_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def json_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        value = list(value)
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def attr_value(attrs: h5py.AttributeManager, key: str) -> object | None:
    try:
        value = attrs[key]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value
    except Exception:
        return None


def pair(value: object) -> tuple[float | None, float | None]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) >= 2:
            return to_float(value[0]), to_float(value[1])
        if len(value) == 1:
            number = to_float(value[0])
            return number, number
    number = to_float(value)
    return number, number


def explicit_property(
    properties: openslide._PropertyMap,
    candidates: Sequence[str],
) -> tuple[float | None, str, str]:
    for key in candidates:
        raw = properties.get(key)
        number = to_float(raw)
        if number is not None:
            return number, key, str(raw)
    return None, "missing", ""


def resolve_wsi_path(root: Path, wsi_dir: Path, inventory_row: dict[str, str], slide_id: str) -> Path | None:
    recorded = inventory_row.get("wsi_path", "")
    if recorded:
        candidate = Path(recorded)
        candidate = candidate if candidate.is_absolute() else root / candidate
        if candidate.is_file():
            return candidate
    for suffix in (".svs", ".tif", ".tiff", ".ome.tif", ".ome.tiff"):
        candidate = wsi_dir / f"{slide_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def scan_h5(path: Path, scale: str, slide_id: str, root: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "slide_id": slide_id,
        "scale": scale,
        "h5_path": display_path(root, path),
        "h5_exists": path.is_file(),
        "h5_read_success": False,
        "h5_error": "",
        "coords_exists": False,
        "coords_count": "",
        "coords_shape": "",
        "coords_dtype": "",
        "patch_level": "",
        "patch_size": "",
        "h5_downsample_x": "",
        "h5_downsample_y": "",
        "h5_level_dim": "",
        "h5_downsampled_level_dim": "",
        "h5_name": "",
        "h5_save_path": "",
        "required_metadata_complete": False,
    }
    if not path.is_file():
        result["h5_error"] = "file_missing"
        return result
    try:
        with h5py.File(path, "r") as handle:
            if "coords" not in handle:
                result["h5_error"] = "coords_dataset_missing"
                return result
            coords = handle["coords"]
            result["coords_exists"] = True
            result["coords_count"] = coords.shape[0] if coords.shape else 0
            result["coords_shape"] = json_value(coords.shape)
            result["coords_dtype"] = str(coords.dtype)
            attrs = coords.attrs
            patch_level = attr_value(attrs, "patch_level")
            patch_size = attr_value(attrs, "patch_size")
            downsample = attr_value(attrs, "downsample")
            ds_x, ds_y = pair(downsample)
            result.update({
                "patch_level": "" if patch_level is None else to_int(patch_level),
                "patch_size": "" if patch_size is None else to_int(patch_size),
                "h5_downsample_x": "" if ds_x is None else ds_x,
                "h5_downsample_y": "" if ds_y is None else ds_y,
                "h5_level_dim": json_value(attr_value(attrs, "level_dim")),
                "h5_downsampled_level_dim": json_value(attr_value(attrs, "downsampled_level_dim")),
                "h5_name": attr_value(attrs, "name") or "",
                "h5_save_path": attr_value(attrs, "save_path") or "",
                "h5_read_success": True,
            })
            result["required_metadata_complete"] = (
                to_int(patch_level) is not None and to_int(patch_size) is not None
                and ds_x is not None and ds_y is not None
            )
    except Exception as exc:
        result["h5_error"] = f"{type(exc).__name__}: {exc}"
    return result


def numeric_values(rows: Iterable[dict[str, object]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = to_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("count", "min", "p25", "median", "p75", "max", "mean", "sd", "cv_pct")}
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr))
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": mean,
        "sd": sd,
        "cv_pct": (sd / mean * 100.0) if mean else None,
    }


def stat_line(name: str, values: Sequence[float], unit: str = "") -> str:
    stats = describe(values)
    if not values:
        return f"- {name}: unavailable"
    suffix = f" {unit}" if unit else ""
    return (
        f"- {name}: n={stats['count']}, min={stats['min']:.6g}, median={stats['median']:.6g}, "
        f"max={stats['max']:.6g}, mean={stats['mean']:.6g}, SD={stats['sd']:.6g}, "
        f"CV={stats['cv_pct']:.4g}%{suffix}"
    )


def rel_deviation(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0):
        return None
    return abs(value - reference) / abs(reference)


def add_anomaly(
    rows: list[dict[str, object]], slide_id: str, issue_type: str, severity: str,
    scale: str, observed: object, expected: object, details: str,
) -> None:
    rows.append({
        "slide_id": slide_id,
        "issue_type": issue_type,
        "severity": severity,
        "scale": scale,
        "observed": observed,
        "expected_or_reference": expected,
        "details": details,
    })


def save_figures(
    figures_dir: Path,
    metadata_rows: list[dict[str, object]],
    mag_rows: list[dict[str, object]],
    fov_rows: list[dict[str, object]],
) -> None:
    plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight"})

    objective = Counter(str(row["objective_power"]) for row in metadata_rows if row["objective_power"] != "")
    missing_objective = sum(row["objective_power"] == "" for row in metadata_rows)
    labels = list(objective.keys()) + (["missing"] if missing_objective else [])
    counts = list(objective.values()) + ([missing_objective] if missing_objective else [])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(labels, counts, color="#3572A5")
    ax.set(title="WSI objective power", xlabel="Objective power", ylabel="Slides")
    for index, value in enumerate(counts):
        ax.text(index, value, str(value), ha="center", va="bottom", fontsize=8)
    fig.savefig(figures_dir / "objective_power_distribution.png")
    plt.close(fig)

    mpp_x = numeric_values(metadata_rows, "mpp_x")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(mpp_x, bins=min(40, max(5, int(math.sqrt(len(mpp_x))))) if mpp_x else 5, color="#2A9D8F")
    ax.set(title="Level-0 MPP-X distribution", xlabel="MPP-X (um/pixel)", ylabel="Slides")
    fig.savefig(figures_dir / "mpp_x_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for scale, color in (("5x", "#457B9D"), ("20x", "#E76F51")):
        values = numeric_values(mag_rows, f"actual_magnification_{scale}")
        ax.hist(values, bins=30, alpha=0.65, label=scale, color=color)
        ax.axvline(NOMINAL_MAGNIFICATIONS[scale], color=color, linestyle="--", linewidth=1)
    ax.set(title="Actual branch magnification", xlabel="Effective magnification", ylabel="Slides")
    ax.legend()
    fig.savefig(figures_dir / "actual_magnification_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for scale, color in (("5x", "#457B9D"), ("20x", "#E76F51")):
        values = [to_float(row["fov_geomean_um"]) for row in fov_rows if row["scale"] == scale]
        values = [value for value in values if value is not None]
        ax.hist(values, bins=30, alpha=0.65, label=scale, color=color)
    ax.set(title="Physical patch FOV", xlabel="Geometric-mean FOV (um)", ylabel="Slides")
    ax.legend()
    fig.savefig(figures_dir / "physical_fov_distribution.png")
    plt.close(fig)

    signatures = Counter(str(row["pyramid_topology_signature"]) for row in metadata_rows if row["metadata_read_success"])
    fig, ax = plt.subplots(figsize=(8.0, max(3.5, min(8.0, 0.45 * len(signatures) + 1.5))))
    items = signatures.most_common(10)
    ax.barh([item[0] for item in items][::-1], [item[1] for item in items][::-1], color="#6A994E")
    ax.set(title="Top pyramid topology signatures", xlabel="Slides", ylabel="level_count | downsamples rounded to 0.1")
    fig.savefig(figures_dir / "pyramid_topology_distribution.png")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = resolve_project_root(args.project_root)
    csv_path = rooted(root, args.csv)
    inventory_path = rooted(root, args.inventory)
    output_dir = rooted(root, args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_fields, dataset_rows = read_csv(csv_path)
    if not {"slide_id", "case_id", "label"}.issubset(csv_fields):
        raise ValueError("Dataset CSV must contain slide_id, case_id, and label")
    inventory_fields, inventory_rows = read_csv(inventory_path)
    if "slide_id" not in inventory_fields:
        raise ValueError("Step 2.0 inventory lacks slide_id")
    if [row["slide_id"] for row in inventory_rows] != [row["slide_id"] for row in dataset_rows]:
        raise ValueError("Dataset slide population/order differs from Step 2.0 inventory")
    inventory_by_slide = {row["slide_id"]: row for row in inventory_rows}
    wsi_dir = rooted(root, args.wsi_dir)

    coord_dirs = {
        "5x": rooted(root, args.coord_5x_dir),
        "20x": rooted(root, args.coord_20x_dir),
    }
    metadata_rows: list[dict[str, object]] = []
    pyramid_rows: list[dict[str, object]] = []
    h5_rows: list[dict[str, object]] = []
    wsi_cache: dict[str, dict[str, object]] = {}

    for index, row in enumerate(dataset_rows, start=1):
        slide_id = row["slide_id"]
        wsi_path = resolve_wsi_path(root, wsi_dir, inventory_by_slide[slide_id], slide_id)
        metadata: dict[str, object] = {
            "slide_id": slide_id,
            "case_id": row["case_id"],
            "label": row["label"],
            "wsi_path": display_path(root, wsi_path) if wsi_path else "",
            "metadata_read_success": False,
            "metadata_error": "",
            "vendor": "",
            "vendor_source": "openslide.vendor",
            "objective_power": "",
            "objective_power_source": "missing",
            "objective_power_raw": "",
            "mpp_x": "",
            "mpp_y": "",
            "mpp_x_source": "missing",
            "mpp_y_source": "missing",
            "mpp_x_raw": "",
            "mpp_y_raw": "",
            "level_count": "",
            "level_dimensions": "",
            "level_downsamples": "",
            "pyramid_topology_signature": "",
            "pyramid_exact_downsample_signature": "",
        }
        if wsi_path is None:
            metadata["metadata_error"] = "wsi_path_not_resolved"
        else:
            try:
                slide = openslide.OpenSlide(str(wsi_path))
                properties = slide.properties
                vendor = properties.get("openslide.vendor", "")
                objective, objective_source, objective_raw = explicit_property(
                    properties,
                    ("openslide.objective-power", "aperio.AppMag", "hamamatsu.SourceLens"),
                )
                mpp_x, mpp_x_source, mpp_x_raw = explicit_property(
                    properties, ("openslide.mpp-x", "aperio.MPP")
                )
                mpp_y, mpp_y_source, mpp_y_raw = explicit_property(
                    properties, ("openslide.mpp-y", "aperio.MPP")
                )
                dimensions = [(int(width), int(height)) for width, height in slide.level_dimensions]
                downsamples = [float(value) for value in slide.level_downsamples]
                level0_width, level0_height = dimensions[0]
                ds_xy = [
                    (level0_width / width, level0_height / height)
                    for width, height in dimensions
                ]
                topology = f"{slide.level_count}|" + ",".join(f"{value:.1f}" for value in downsamples)
                exact_signature = f"{slide.level_count}|" + ",".join(f"{value:.6f}" for value in downsamples)
                metadata.update({
                    "metadata_read_success": True,
                    "vendor": vendor,
                    "objective_power": "" if objective is None else objective,
                    "objective_power_source": objective_source,
                    "objective_power_raw": objective_raw,
                    "mpp_x": "" if mpp_x is None else mpp_x,
                    "mpp_y": "" if mpp_y is None else mpp_y,
                    "mpp_x_source": mpp_x_source,
                    "mpp_y_source": mpp_y_source,
                    "mpp_x_raw": mpp_x_raw,
                    "mpp_y_raw": mpp_y_raw,
                    "level_count": slide.level_count,
                    "level_dimensions": json_value(dimensions),
                    "level_downsamples": json_value(downsamples),
                    "pyramid_topology_signature": topology,
                    "pyramid_exact_downsample_signature": exact_signature,
                })
                for level, ((width, height), scalar_ds, (ds_x, ds_y)) in enumerate(zip(dimensions, downsamples, ds_xy)):
                    pyramid_rows.append({
                        "slide_id": slide_id,
                        "level": level,
                        "width_px": width,
                        "height_px": height,
                        "openslide_downsample": scalar_ds,
                        "downsample_x_from_dimensions": ds_x,
                        "downsample_y_from_dimensions": ds_y,
                        "effective_magnification": "" if objective is None else objective / scalar_ds,
                        "effective_magnification_x": "" if objective is None else objective / ds_x,
                        "effective_magnification_y": "" if objective is None else objective / ds_y,
                        "effective_mpp_x": "" if mpp_x is None else mpp_x * ds_x,
                        "effective_mpp_y": "" if mpp_y is None else mpp_y * ds_y,
                    })
                wsi_cache[slide_id] = {
                    "objective": objective,
                    "mpp_x": mpp_x,
                    "mpp_y": mpp_y,
                    "dimensions": dimensions,
                    "downsamples": downsamples,
                    "ds_xy": ds_xy,
                    "level_count": slide.level_count,
                }
                slide.close()
            except Exception as exc:
                metadata["metadata_error"] = f"{type(exc).__name__}: {exc}"
        metadata_rows.append(metadata)

        for scale in SCALES:
            h5_rows.append(scan_h5(coord_dirs[scale] / f"{slide_id}.h5", scale, slide_id, root))
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(dataset_rows)):
            print(f"Scanned {index}/{len(dataset_rows)} slides", flush=True)

    h5_by_slide_scale = {(str(row["slide_id"]), str(row["scale"])): row for row in h5_rows}
    magnification_rows: list[dict[str, object]] = []
    fov_rows: list[dict[str, object]] = []
    anomalies: list[dict[str, object]] = []

    for metadata in metadata_rows:
        slide_id = str(metadata["slide_id"])
        if not metadata["metadata_read_success"]:
            add_anomaly(anomalies, slide_id, "wsi_metadata_read_failure", "critical", "", metadata["metadata_error"], "readable WSI metadata", "OpenSlide could not read metadata.")
            continue
        if not metadata["vendor"]:
            add_anomaly(anomalies, slide_id, "vendor_missing", "warning", "", "missing", "explicit vendor metadata", "No vendor string was exposed by OpenSlide.")
        if metadata["objective_power"] == "":
            add_anomaly(anomalies, slide_id, "objective_power_missing", "critical", "", "missing", "explicit objective power", "Effective magnification cannot be calculated; no value was guessed.")
        if metadata["mpp_x"] == "" or metadata["mpp_y"] == "":
            add_anomaly(anomalies, slide_id, "mpp_missing", "critical", "", f"x={metadata['mpp_x']},y={metadata['mpp_y']}", "explicit MPP-X and MPP-Y", "Physical FOV cannot be fully calculated; no value was guessed.")

        cache = wsi_cache[slide_id]
        mag_row: dict[str, object] = {
            "slide_id": slide_id,
            "case_id": metadata["case_id"],
            "label": metadata["label"],
            "objective_power": metadata["objective_power"],
            "magnification_calculable": cache["objective"] is not None,
        }
        branch_magnifications: dict[str, float | None] = {}
        branch_fov: dict[str, float | None] = {}
        for scale in SCALES:
            h5 = h5_by_slide_scale[(slide_id, scale)]
            level = to_int(h5["patch_level"])
            patch_size = to_int(h5["patch_size"])
            valid_level = level is not None and 0 <= level < int(cache["level_count"])
            if not h5["h5_read_success"]:
                add_anomaly(anomalies, slide_id, "coordinate_h5_read_failure", "critical", scale, h5["h5_error"], "readable coordinate H5", "Coordinate metadata could not be audited.")
            elif not h5["required_metadata_complete"]:
                add_anomaly(anomalies, slide_id, "coordinate_h5_metadata_missing", "critical", scale, "incomplete", "patch_level, patch_size, downsample", "Required H5 attributes are incomplete.")
            if not valid_level:
                add_anomaly(anomalies, slide_id, "patch_level_invalid", "critical", scale, level, f"0..{int(cache['level_count']) - 1}", "H5 patch_level is unavailable or outside the WSI pyramid.")
                scalar_ds = ds_x = ds_y = None
            else:
                scalar_ds = float(cache["downsamples"][level])
                ds_x, ds_y = cache["ds_xy"][level]
                attr_ds_x = to_float(h5["h5_downsample_x"])
                attr_ds_y = to_float(h5["h5_downsample_y"])
                mismatch_x = rel_deviation(attr_ds_x, ds_x)
                mismatch_y = rel_deviation(attr_ds_y, ds_y)
                if (mismatch_x is not None and mismatch_x > H5_WSI_DOWNSAMPLE_TOLERANCE) or (mismatch_y is not None and mismatch_y > H5_WSI_DOWNSAMPLE_TOLERANCE):
                    add_anomaly(anomalies, slide_id, "h5_wsi_downsample_mismatch", "warning", scale, f"h5=({attr_ds_x},{attr_ds_y})", f"wsi=({ds_x},{ds_y})", "H5 downsample differs from the current WSI pyramid by more than 0.1%.")
            objective = cache["objective"]
            actual_mag = objective / scalar_ds if objective is not None and scalar_ds else None
            branch_magnifications[scale] = actual_mag
            nominal = NOMINAL_MAGNIFICATIONS[scale]
            deviation_pct = ((actual_mag - nominal) / nominal * 100.0) if actual_mag is not None else None
            mag_row.update({
                f"patch_level_{scale}": "" if level is None else level,
                f"patch_size_{scale}": "" if patch_size is None else patch_size,
                f"downsample_{scale}": "" if scalar_ds is None else scalar_ds,
                f"downsample_x_{scale}": "" if ds_x is None else ds_x,
                f"downsample_y_{scale}": "" if ds_y is None else ds_y,
                f"actual_magnification_{scale}": "" if actual_mag is None else actual_mag,
                f"actual_magnification_x_{scale}": "" if objective is None or ds_x is None else objective / ds_x,
                f"actual_magnification_y_{scale}": "" if objective is None or ds_y is None else objective / ds_y,
                f"nominal_magnification_{scale}": nominal,
                f"signed_deviation_from_nominal_pct_{scale}": "" if deviation_pct is None else deviation_pct,
                f"absolute_deviation_from_nominal_pct_{scale}": "" if deviation_pct is None else abs(deviation_pct),
            })

            mpp_x = cache["mpp_x"]
            mpp_y = cache["mpp_y"]
            fov_x = patch_size * ds_x * mpp_x if patch_size and ds_x and mpp_x is not None else None
            fov_y = patch_size * ds_y * mpp_y if patch_size and ds_y and mpp_y is not None else None
            fov_geomean = math.sqrt(fov_x * fov_y) if fov_x is not None and fov_y is not None else None
            branch_fov[scale] = fov_geomean
            fov_rows.append({
                "slide_id": slide_id,
                "case_id": metadata["case_id"],
                "label": metadata["label"],
                "scale": scale,
                "patch_level": "" if level is None else level,
                "patch_size_px_at_level": "" if patch_size is None else patch_size,
                "level_downsample_x": "" if ds_x is None else ds_x,
                "level_downsample_y": "" if ds_y is None else ds_y,
                "level0_footprint_width_px": "" if patch_size is None or ds_x is None else patch_size * ds_x,
                "level0_footprint_height_px": "" if patch_size is None or ds_y is None else patch_size * ds_y,
                "level0_mpp_x": "" if mpp_x is None else mpp_x,
                "level0_mpp_y": "" if mpp_y is None else mpp_y,
                "effective_mpp_x_at_level": "" if mpp_x is None or ds_x is None else mpp_x * ds_x,
                "effective_mpp_y_at_level": "" if mpp_y is None or ds_y is None else mpp_y * ds_y,
                "fov_width_um": "" if fov_x is None else fov_x,
                "fov_height_um": "" if fov_y is None else fov_y,
                "fov_geomean_um": "" if fov_geomean is None else fov_geomean,
                "fov_area_mm2": "" if fov_x is None or fov_y is None else fov_x * fov_y / 1_000_000.0,
                "fov_calculable": fov_geomean is not None,
            })
        low_mag, high_mag = branch_magnifications["5x"], branch_magnifications["20x"]
        ratio = high_mag / low_mag if low_mag not in (None, 0) and high_mag is not None else None
        low_fov, high_fov = branch_fov["5x"], branch_fov["20x"]
        fov_ratio = low_fov / high_fov if high_fov not in (None, 0) and low_fov is not None else None
        mag_row.update({
            "high_low_actual_magnification_ratio": "" if ratio is None else ratio,
            "high_low_ratio_deviation_from_4_pct": "" if ratio is None else (ratio - 4.0) / 4.0 * 100.0,
            "low_high_physical_fov_ratio": "" if fov_ratio is None else fov_ratio,
        })
        magnification_rows.append(mag_row)

    objective_values = numeric_values(metadata_rows, "objective_power")
    mpp_x_values = numeric_values(metadata_rows, "mpp_x")
    mpp_y_values = numeric_values(metadata_rows, "mpp_y")
    medians = {
        "objective_power": statistics.median(objective_values) if objective_values else None,
        "mpp_x": statistics.median(mpp_x_values) if mpp_x_values else None,
        "mpp_y": statistics.median(mpp_y_values) if mpp_y_values else None,
        "actual_magnification_5x": statistics.median(numeric_values(magnification_rows, "actual_magnification_5x")) if numeric_values(magnification_rows, "actual_magnification_5x") else None,
        "actual_magnification_20x": statistics.median(numeric_values(magnification_rows, "actual_magnification_20x")) if numeric_values(magnification_rows, "actual_magnification_20x") else None,
        "high_low_actual_magnification_ratio": statistics.median(numeric_values(magnification_rows, "high_low_actual_magnification_ratio")) if numeric_values(magnification_rows, "high_low_actual_magnification_ratio") else None,
    }
    for scale in SCALES:
        values = [to_float(row["fov_geomean_um"]) for row in fov_rows if row["scale"] == scale]
        values = [value for value in values if value is not None]
        medians[f"fov_{scale}"] = statistics.median(values) if values else None

    topology_counts = Counter(str(row["pyramid_topology_signature"]) for row in metadata_rows if row["metadata_read_success"])
    level_count_values = [to_int(row["level_count"]) for row in metadata_rows if row["metadata_read_success"]]
    level_count_values = [value for value in level_count_values if value is not None]
    modal_level_count = Counter(level_count_values).most_common(1)[0][0] if level_count_values else None
    downsample_median_by_level: dict[int, float] = {}
    for level in range(modal_level_count or 0):
        values = [
            to_float(row["openslide_downsample"])
            for row in pyramid_rows
            if to_int(row["level"]) == level
        ]
        values = [value for value in values if value is not None]
        if values:
            downsample_median_by_level[level] = statistics.median(values)
    for metadata in metadata_rows:
        slide_id = str(metadata["slide_id"])
        if not metadata["metadata_read_success"]:
            continue
        for field in ("objective_power", "mpp_x", "mpp_y"):
            value = to_float(metadata[field])
            reference = medians[field]
            deviation = rel_deviation(value, reference)
            if deviation is not None and deviation > RELATIVE_TOLERANCE:
                add_anomaly(anomalies, slide_id, f"{field}_cross_wsi_outlier", "warning", "", value, reference, "Value differs from the dataset median by more than 5%.")
        level_count = to_int(metadata["level_count"])
        if modal_level_count is not None and level_count != modal_level_count:
            add_anomaly(anomalies, slide_id, "pyramid_level_count_outlier", "warning", "", level_count, modal_level_count, "Level count differs from the dataset mode.")
        elif level_count is not None:
            slide_levels = [row for row in pyramid_rows if row["slide_id"] == slide_id]
            outlying_levels = []
            for level_row in slide_levels:
                level = to_int(level_row["level"])
                value = to_float(level_row["openslide_downsample"])
                reference = downsample_median_by_level.get(level) if level is not None else None
                if rel_deviation(value, reference) is not None and rel_deviation(value, reference) > RELATIVE_TOLERANCE:
                    outlying_levels.append(f"L{level}:{value} vs {reference}")
            if outlying_levels:
                add_anomaly(anomalies, slide_id, "pyramid_downsample_outlier", "warning", "", ";".join(outlying_levels), "per-level dataset medians", "One or more level downsamples differ from the corresponding dataset median by more than 5%.")

    for row in magnification_rows:
        slide_id = str(row["slide_id"])
        for scale in SCALES:
            actual = to_float(row[f"actual_magnification_{scale}"])
            nominal = NOMINAL_MAGNIFICATIONS[scale]
            if actual is not None and rel_deviation(actual, nominal) > 0.10:
                add_anomaly(anomalies, slide_id, "systematic_nominal_magnification_deviation", "warning", scale, actual, nominal, "Actual magnification differs from the branch name by more than 10%; evaluate as a systematic naming issue versus cross-WSI variability.")
            reference = medians[f"actual_magnification_{scale}"]
            if rel_deviation(actual, reference) is not None and rel_deviation(actual, reference) > RELATIVE_TOLERANCE:
                add_anomaly(anomalies, slide_id, "actual_magnification_cross_wsi_outlier", "warning", scale, actual, reference, "Actual branch magnification differs from its dataset median by more than 5%.")
        ratio = to_float(row["high_low_actual_magnification_ratio"])
        if ratio is not None and rel_deviation(ratio, 4.0) > RELATIVE_TOLERANCE:
            add_anomaly(anomalies, slide_id, "high_low_ratio_deviation", "warning", "both", ratio, 4.0, "High/low magnification ratio differs from 4 by more than 5%.")

    for row in fov_rows:
        value = to_float(row["fov_geomean_um"])
        reference = medians[f"fov_{row['scale']}"]
        if rel_deviation(value, reference) is not None and rel_deviation(value, reference) > RELATIVE_TOLERANCE:
            add_anomaly(anomalies, str(row["slide_id"]), "physical_fov_cross_wsi_outlier", "warning", str(row["scale"]), value, reference, "Physical FOV differs from the branch median by more than 5%.")

    metadata_fields = [
        "slide_id", "case_id", "label", "wsi_path", "metadata_read_success", "metadata_error",
        "vendor", "vendor_source", "objective_power", "objective_power_source", "objective_power_raw",
        "mpp_x", "mpp_y", "mpp_x_source", "mpp_y_source", "mpp_x_raw", "mpp_y_raw",
        "level_count", "level_dimensions", "level_downsamples", "pyramid_topology_signature",
        "pyramid_exact_downsample_signature",
    ]
    pyramid_fields = [
        "slide_id", "level", "width_px", "height_px", "openslide_downsample",
        "downsample_x_from_dimensions", "downsample_y_from_dimensions", "effective_magnification",
        "effective_magnification_x", "effective_magnification_y", "effective_mpp_x", "effective_mpp_y",
    ]
    h5_fields = [
        "slide_id", "scale", "h5_path", "h5_exists", "h5_read_success", "h5_error", "coords_exists",
        "coords_count", "coords_shape", "coords_dtype", "patch_level", "patch_size", "h5_downsample_x",
        "h5_downsample_y", "h5_level_dim", "h5_downsampled_level_dim", "h5_name", "h5_save_path",
        "required_metadata_complete",
    ]
    mag_fields = [
        "slide_id", "case_id", "label", "objective_power", "magnification_calculable",
        "patch_level_5x", "patch_size_5x", "downsample_5x", "downsample_x_5x", "downsample_y_5x",
        "actual_magnification_5x", "actual_magnification_x_5x", "actual_magnification_y_5x",
        "nominal_magnification_5x", "signed_deviation_from_nominal_pct_5x", "absolute_deviation_from_nominal_pct_5x",
        "patch_level_20x", "patch_size_20x", "downsample_20x", "downsample_x_20x", "downsample_y_20x",
        "actual_magnification_20x", "actual_magnification_x_20x", "actual_magnification_y_20x",
        "nominal_magnification_20x", "signed_deviation_from_nominal_pct_20x", "absolute_deviation_from_nominal_pct_20x",
        "high_low_actual_magnification_ratio", "high_low_ratio_deviation_from_4_pct", "low_high_physical_fov_ratio",
    ]
    fov_fields = [
        "slide_id", "case_id", "label", "scale", "patch_level", "patch_size_px_at_level",
        "level_downsample_x", "level_downsample_y", "level0_footprint_width_px", "level0_footprint_height_px",
        "level0_mpp_x", "level0_mpp_y", "effective_mpp_x_at_level", "effective_mpp_y_at_level",
        "fov_width_um", "fov_height_um", "fov_geomean_um", "fov_area_mm2", "fov_calculable",
    ]
    anomaly_fields = ["slide_id", "issue_type", "severity", "scale", "observed", "expected_or_reference", "details"]
    write_csv(output_dir / "wsi_metadata.csv", metadata_fields, metadata_rows)
    write_csv(output_dir / "pyramid_levels.csv", pyramid_fields, pyramid_rows)
    write_csv(output_dir / "coordinate_h5_metadata.csv", h5_fields, h5_rows)
    write_csv(output_dir / "magnification_audit.csv", mag_fields, magnification_rows)
    write_csv(output_dir / "physical_fov.csv", fov_fields, fov_rows)
    write_csv(output_dir / "wsi_metadata_anomalies.csv", anomaly_fields, anomalies)
    save_figures(figures_dir, metadata_rows, magnification_rows, fov_rows)

    success_count = sum(bool(row["metadata_read_success"]) for row in metadata_rows)
    vendor_counts = Counter(str(row["vendor"]) if row["vendor"] else "missing" for row in metadata_rows)
    objective_counts = Counter(str(row["objective_power"]) if row["objective_power"] != "" else "missing" for row in metadata_rows)
    level_count_counts = Counter(str(row["level_count"]) if row["level_count"] != "" else "missing" for row in metadata_rows)
    h5_configs = {
        scale: Counter((str(row["patch_level"]), str(row["patch_size"])) for row in h5_rows if row["scale"] == scale)
        for scale in SCALES
    }
    mag_5 = numeric_values(magnification_rows, "actual_magnification_5x")
    mag_20 = numeric_values(magnification_rows, "actual_magnification_20x")
    ratios = numeric_values(magnification_rows, "high_low_actual_magnification_ratio")
    fov_5 = [to_float(row["fov_geomean_um"]) for row in fov_rows if row["scale"] == "5x"]
    fov_20 = [to_float(row["fov_geomean_um"]) for row in fov_rows if row["scale"] == "20x"]
    fov_5 = [value for value in fov_5 if value is not None]
    fov_20 = [value for value in fov_20 if value is not None]
    scale_issue_types = {
        "objective_power_cross_wsi_outlier", "mpp_x_cross_wsi_outlier", "mpp_y_cross_wsi_outlier",
        "actual_magnification_cross_wsi_outlier", "physical_fov_cross_wsi_outlier", "high_low_ratio_deviation",
    }
    structural_issue_types = {"pyramid_level_count_outlier", "pyramid_downsample_outlier"}
    cross_wsi_scale_anomaly_slides = {str(row["slide_id"]) for row in anomalies if row["issue_type"] in scale_issue_types}
    structural_anomaly_slides = {str(row["slide_id"]) for row in anomalies if row["issue_type"] in structural_issue_types}
    metadata_critical_slides = {str(row["slide_id"]) for row in anomalies if row["severity"] == "critical"}
    systematic_nominal_slides = {str(row["slide_id"]) for row in anomalies if row["issue_type"] == "systematic_nominal_magnification_deviation"}

    enough_metadata = success_count == len(dataset_rows) and not metadata_critical_slides
    cross_wsi_consistent = not cross_wsi_scale_anomaly_slides
    nominal_accurate = all(rel_deviation(statistics.median(values), NOMINAL_MAGNIFICATIONS[scale]) <= 0.10 for scale, values in (("5x", mag_5), ("20x", mag_20)) if values)
    if not enough_metadata:
        category = "metadata insufficient to judge all slides"
    elif cross_wsi_consistent and not nominal_accurate:
        category = "5x/20x naming is imprecise, but the underlying data are scale-consistent"
    elif cross_wsi_consistent and nominal_accurate:
        category = "scale definition is reliable"
    else:
        category = "substantive cross-WSI scale inconsistency is present"

    can_step_23 = success_count == len(dataset_rows) and all(bool(row["h5_read_success"]) and bool(row["required_metadata_complete"]) for row in h5_rows)
    regeneration_evidence = bool(metadata_critical_slides or cross_wsi_scale_anomaly_slides)
    summary = [
        "# Step 2.2: Yiyuan WSI Metadata and Physical Scale Audit",
        "",
        "## Scope and formulas",
        "",
        f"- Population: {len(dataset_rows)} slides, dynamically matched to Step 2.0 inventory.",
        f"- OpenSlide: python {openslide.__version__}, library {openslide.__library_version__}; h5py {h5py.__version__}.",
        "- No WSI pixels or coordinate arrays were read. Only OpenSlide metadata and H5 dataset shape/dtype/attributes were accessed.",
        "- Coordinates are level-0 positions: `create_patches_fp.py` generates candidates in level-0 space and downstream `read_region((x,y), level, size)` consumes level-0 locations.",
        "- Effective magnification = objective power / actual OpenSlide level downsample.",
        "- Effective MPP-X/Y = level-0 MPP-X/Y x dimension-derived level downsample-X/Y.",
        "- Physical FOV-X/Y = patch size at selected level x dimension-derived downsample-X/Y x level-0 MPP-X/Y.",
        "- Explicit fallback chains only: objective uses `openslide.objective-power`, then `aperio.AppMag`, then `hamamatsu.SourceLens`; MPP uses `openslide.mpp-x/y`, then explicit `aperio.MPP`. Missing values are never inferred from a nominal magnification.",
        "- Cross-WSI anomaly threshold: >5% from the dataset median. H5/WSI downsample mismatch threshold: >0.1%. Nominal branch-name warning: >10%.",
        "",
        "## WSI metadata completeness",
        "",
        f"- Successfully read: {success_count}/{len(dataset_rows)}",
        f"- Metadata/coordinate critical-anomaly slides: {len(metadata_critical_slides)}",
        f"- Vendor counts: {dict(vendor_counts)}",
        f"- Objective power counts: {dict(objective_counts)}",
        f"- Level-count distribution: {dict(level_count_counts)}",
        f"- Pyramid topology signatures (level count and downsamples rounded to 0.1): unique={len(topology_counts)}, top 10={dict(topology_counts.most_common(10))}",
        f"- Exact downsample sequences: {len({row['pyramid_exact_downsample_signature'] for row in metadata_rows if row['metadata_read_success']})} (small dimension-rounding differences are retained in CSV and are not by themselves topology anomalies).",
        stat_line("MPP-X", mpp_x_values, "um/pixel"),
        stat_line("MPP-Y", mpp_y_values, "um/pixel"),
        "",
        "## Coordinate H5 configuration",
        "",
        f"- 5x `(patch_level, patch_size)` counts: {dict(h5_configs['5x'])}",
        f"- 20x `(patch_level, patch_size)` counts: {dict(h5_configs['20x'])}",
        f"- Readable and complete H5 metadata rows: {sum(bool(row['h5_read_success']) and bool(row['required_metadata_complete']) for row in h5_rows)}/{len(h5_rows)}",
        "",
        "## Actual magnification",
        "",
        stat_line("Nominal 5x branch actual magnification", mag_5, "x"),
        stat_line("Nominal 20x branch actual magnification", mag_20, "x"),
        stat_line("High/low actual magnification ratio", ratios),
        f"- Slides with systematic nominal-magnification warnings: {len(systematic_nominal_slides)}",
        "",
        "## Physical FOV",
        "",
        stat_line("Nominal 5x branch geometric-mean FOV", fov_5, "um"),
        stat_line("Nominal 20x branch geometric-mean FOV", fov_20, "um"),
        stat_line("Low/high physical FOV ratio", numeric_values(magnification_rows, "low_high_physical_fov_ratio")),
        "",
        "## Anomalies",
        "",
        f"- Total anomaly records: {len(anomalies)}",
        f"- Slides with cross-WSI scale anomalies: {len(cross_wsi_scale_anomaly_slides)}",
        f"- Slides with pyramid structural anomalies: {len(structural_anomaly_slides)}",
        f"- Slides with critical missing/unreadable metadata or H5 configuration: {len(metadata_critical_slides)}",
        f"- Anomaly type counts: {dict(Counter(str(row['issue_type']) for row in anomalies))}",
        "",
        "## Required answers",
        "",
        f"1. Metadata read success: **{success_count}/{len(dataset_rows)}**.",
        f"2. Current 5x configuration: **{h5_configs['5x'].most_common(1)[0][0] if h5_configs['5x'] else 'unavailable'}** as `(patch_level, patch_size)`.",
        f"3. Current 20x configuration: **{h5_configs['20x'].most_common(1)[0][0] if h5_configs['20x'] else 'unavailable'}** as `(patch_level, patch_size)`.",
        f"4. Nominal 5x uniformly close to 5x: **{bool(mag_5) and all(rel_deviation(value, 5.0) <= 0.10 for value in mag_5)}**.",
        f"5. Nominal 20x uniformly close to 20x: **{bool(mag_20) and all(rel_deviation(value, 20.0) <= 0.10 for value in mag_20)}**.",
        f"6. High/low ratio stably close to 4: **{bool(ratios) and all(rel_deviation(value, 4.0) <= RELATIVE_TOLERANCE for value in ratios)}**.",
        f"7. Physical FOV cross-WSI consistency within 5% of branch medians: **{not any(row['issue_type'] == 'physical_fov_cross_wsi_outlier' for row in anomalies)}**.",
        f"8. Pyramid structure consistent within the 5% per-level criterion: **{not any(row['issue_type'] in {'pyramid_level_count_outlier', 'pyramid_downsample_outlier'} for row in anomalies)}**.",
        f"9. Metadata/magnification/FOV anomaly slides: critical={len(metadata_critical_slides)}, cross-WSI scale={len(cross_wsi_scale_anomaly_slides)}, pyramid structure={len(structural_anomaly_slides)}, systematic nominal-name deviation={len(systematic_nominal_slides)}.",
        f"10. Classification: **{category}**.",
        f"11. Ready for Step 2.3 coordinate audit: **{can_step_23}** (Step 2.3 was not executed).",
        f"12. Clear current evidence that patches must be regenerated: **{regeneration_evidence}**. A systematic naming mismatch alone is not regeneration evidence when physical scales are internally consistent.",
        "",
        "## Current conclusion",
        "",
        f"- {category}.",
        f"- Step 2.3 readiness: {can_step_23}.",
        f"- Patch-regeneration evidence from this metadata audit: {regeneration_evidence}.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Outputs written to {display_path(root, output_dir)}")
    print(f"WSI metadata success: {success_count}/{len(dataset_rows)}; anomalies: {len(anomalies)}")
    print(f"Classification: {category}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
