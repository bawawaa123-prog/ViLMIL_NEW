#!/usr/bin/env python3
"""Validate staged Step 2.3 nominal-5x BiomedCLIP feature repairs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--alignment-csv", type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/03_coordinate_audit/coord_feature_alignment.csv"),
    )
    parser.add_argument(
        "--coord-dir", type=Path, default=Path("data/yiyuan/patches_coords_5x/patches_256")
    )
    parser.add_argument(
        "--candidate-dir", type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/03_coordinate_audit/feature_repair_staging_5x"),
    )
    parser.add_argument(
        "--output-csv", type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/03_coordinate_audit/feature_repair_verification.csv"),
    )
    return parser.parse_args()


def root_path(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    cwd = Path.cwd().resolve()
    return cwd if (cwd / "dataset_csv" / "all_data.csv").is_file() else Path(__file__).resolve().parents[3]


def rooted(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def targets_from_alignment(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            row["slide_id"] for row in csv.DictReader(handle)
            if row.get("scale") == "5x" and row.get("all_three_counts_equal", "").lower() != "true"
        ]


def validate_one(slide_id: str, coord_path: Path, candidate_path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "slide_id": slide_id, "candidate_exists": candidate_path.is_file(), "read_success": False,
        "coord_count": "", "feature_coord_count": "", "feature_count": "",
        "feature_shape": "", "feature_dtype": "", "counts_equal": False,
        "coord_shape_equal": False, "coord_order_exact": False, "coord_set_equal": False,
        "features_all_finite": False, "feature_abs_mean": "", "feature_abs_max": "",
        "validation_ok": False, "error": "",
    }
    try:
        with h5py.File(coord_path, "r") as handle:
            expected = handle["coords"][:]
        row["coord_count"] = len(expected)
        with h5py.File(candidate_path, "r") as handle:
            actual = handle["coords"][:]
            features = handle["features"]
            row.update({
                "read_success": True, "feature_coord_count": len(actual), "feature_count": len(features),
                "feature_shape": str(tuple(features.shape)), "feature_dtype": str(features.dtype),
            })
            row["counts_equal"] = len(expected) == len(actual) == len(features)
            row["coord_shape_equal"] = expected.shape == actual.shape
            row["coord_order_exact"] = expected.shape == actual.shape and np.array_equal(expected, actual)
            if expected.ndim == actual.ndim == 2 and expected.shape[1:] == actual.shape[1:] == (2,):
                left = np.unique(expected.astype(np.int64, copy=False), axis=0)
                right = np.unique(actual.astype(np.int64, copy=False), axis=0)
                row["coord_set_equal"] = left.shape == right.shape and np.array_equal(left, right)
            finite = True
            absolute_sum = 0.0
            absolute_max = 0.0
            element_count = 0
            for start in range(0, len(features), 256):
                block = features[start:start + 256]
                finite = finite and bool(np.isfinite(block).all())
                absolute_sum += float(np.abs(block).sum(dtype=np.float64))
                absolute_max = max(absolute_max, float(np.abs(block).max(initial=0)))
                element_count += block.size
            row["features_all_finite"] = finite
            row["feature_abs_mean"] = absolute_sum / element_count if element_count else math.nan
            row["feature_abs_max"] = absolute_max
            row["validation_ok"] = bool(
                row["counts_equal"] and row["coord_order_exact"] and
                features.shape == (len(expected), 512) and features.dtype == np.dtype("float32") and finite
            )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    args = parse_args()
    root = root_path(args.project_root)
    alignment = rooted(root, args.alignment_csv)
    coord_dir = rooted(root, args.coord_dir)
    candidate_dir = rooted(root, args.candidate_dir)
    output_csv = rooted(root, args.output_csv)
    targets = targets_from_alignment(alignment)
    rows = [
        validate_one(sid, coord_dir / f"{sid}.h5", candidate_dir / f"{sid}.h5") for sid in targets
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(bool(row["validation_ok"]) for row in rows)
    print(f"Validated {len(rows)} repair targets: {passed} passed, {len(rows) - passed} failed")
    for row in rows:
        print(
            f"{row['slide_id']}: ok={row['validation_ok']} coords={row['coord_count']} "
            f"features={row['feature_count']} exact_order={row['coord_order_exact']} error={row['error']}"
        )
    print(f"CSV: {output_csv}")
    return 0 if rows and passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
