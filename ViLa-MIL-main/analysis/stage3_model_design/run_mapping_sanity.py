"""Run a small read-only mapping sanity check against Stage 2 Yiyuan files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cross_scale_mapping import build_cross_scale_mapping


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    root = args.project_root.resolve()
    inventory = rows(root / "analysis/stage2_yiyuan_data_audit/00_inventory/dataset_inventory.csv")
    fov = {(r["slide_id"], r["scale"]): r for r in rows(root / "analysis/stage2_yiyuan_data_audit/02_wsi_metadata/physical_fov.csv")}
    metadata = {r["slide_id"]: r for r in rows(root / "analysis/stage2_yiyuan_data_audit/02_wsi_metadata/wsi_metadata.csv")}
    for row in inventory[: max(0, args.limit)]:
        slide_id = row["slide_id"]

        def load(path):
            with h5py.File(root / path, "r") as handle:
                return np.asarray(handle["coords"][:])

        # Use coordinates stored in feature H5 files: Stage 2 found that the
        # standalone coordinate H5 row order usually differs from feature H5.
        low = load(row["feature_5x_path"])
        high = load(row["feature_20x_path"])
        low_fov, high_fov = fov[(slide_id, "5x")], fov[(slide_id, "20x")]
        dimensions = json.loads(metadata[slide_id]["level_dimensions"])[0]
        mapping = build_cross_scale_mapping(
            low,
            high,
            low_patch_size=float(low_fov["patch_size_px_at_level"]),
            high_patch_size=float(high_fov["patch_size_px_at_level"]),
            low_downsample=float(low_fov["level_downsample_x"]),
            high_downsample=float(high_fov["level_downsample_x"]),
            wsi_dimensions=(float(dimensions[0]), float(dimensions[1])),
        )
        # Every CSR edge must reconstruct a positive-area bbox relation.
        for low_index in range(len(low)):
            low_box = mapping.low_bboxes[low_index]
            for high_index in mapping.children(low_index):
                high_box = mapping.high_bboxes[int(high_index)]
                overlap = max(0.0, min(low_box[2], high_box[2]) - max(low_box[0], high_box[0]))
                overlap *= max(0.0, min(low_box[3], high_box[3]) - max(low_box[1], high_box[1]))
                if overlap <= 0.0:
                    raise AssertionError(f"non-overlap CSR edge: {slide_id} low={low_index} high={high_index}")
        if set(mapping.child_indices.tolist()).union(mapping.unmapped_high_indices.tolist()) != set(range(len(high))):
            raise AssertionError(f"high coverage is incomplete for {slide_id}")
        print(
            f"{slide_id}: low={len(low)} high={len(high)} edges={len(mapping.child_indices)} "
            f"low_coverage={mapping.low_coverage:.6f} high_coverage={mapping.parent_coverage:.6f} "
            f"unmapped_high={len(mapping.unmapped_high_indices)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
