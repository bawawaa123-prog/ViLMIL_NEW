"""Build Stage 3.1 per-slide variable cross-scale mapping caches.

The command is read-only with respect to source H5/metadata files. Outputs are
written only below the requested Stage 3.1 output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cross_scale_mapping import build_cross_scale_mapping


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_feature_coords(root: Path, relative_path: str) -> np.ndarray:
    with h5py.File(root / relative_path, "r") as handle:
        # Coordinates are read from the feature H5, preserving feature-row pairing.
        return np.asarray(handle["coords"][:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis/stage3_model_design/01_cross_scale_mapping/output",
    )
    parser.add_argument("--limit", type=int, default=None, help="process only the first N slides")
    parser.add_argument("--summary-name", default="summary.md")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    cache_dir = output_dir / "mappings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    inventory = read_csv(root / "analysis/stage2_yiyuan_data_audit/00_inventory/dataset_inventory.csv")
    fov_rows = read_csv(root / "analysis/stage2_yiyuan_data_audit/02_wsi_metadata/physical_fov.csv")
    fov = {(row["slide_id"], row["scale"]): row for row in fov_rows}
    metadata = {row["slide_id"]: row for row in read_csv(root / "analysis/stage2_yiyuan_data_audit/02_wsi_metadata/wsi_metadata.csv")}
    rows = inventory if args.limit is None else inventory[: max(0, args.limit)]
    stats = []
    for row in rows:
        slide_id = row["slide_id"]
        low = load_feature_coords(root, row["feature_5x_path"])
        high = load_feature_coords(root, row["feature_20x_path"])
        low_fov, high_fov = fov[(slide_id, "5x")], fov[(slide_id, "20x")]
        dimensions = json.loads(metadata[slide_id]["level_dimensions"])[0]
        mapping = build_cross_scale_mapping(
            low,
            high,
            low_patch_size=float(low_fov["patch_size_px_at_level"]),
            high_patch_size=float(high_fov["patch_size_px_at_level"]),
            low_downsample=(float(low_fov["level_downsample_x"]), float(low_fov["level_downsample_y"])),
            high_downsample=(float(high_fov["level_downsample_x"]), float(high_fov["level_downsample_y"])),
            wsi_dimensions=(float(dimensions[0]), float(dimensions[1])),
        )
        mapping.save_npz(str(cache_dir / f"{slide_id}.npz"))
        stats.append({
            "slide_id": slide_id,
            "low_count": len(low),
            "high_count": len(high),
            "edges": len(mapping.child_indices),
            "low_with_child_count": int((mapping.low_child_counts > 0).sum()),
            "low_child_count_mean": float(mapping.low_child_counts.mean()) if len(low) else 0.0,
            "low_child_count_median": float(np.median(mapping.low_child_counts)) if len(low) else 0.0,
            "high_with_parent_count": int(mapping.high_has_parent_mask.sum()),
            "unmapped_high_count": len(mapping.unmapped_high_indices),
            "high_parent_coverage": mapping.parent_coverage,
            "low_child_coverage": mapping.low_coverage,
            "padding_low_gt50_count": int((mapping.low_padding_ratio > 0.5).sum()),
            "padding_high_gt50_count": int((mapping.high_padding_ratio > 0.5).sum()),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "mapping_statistics.csv"
    fieldnames = list(stats[0]) if stats else ["slide_id"]
    with stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)

    total_low = sum(row["low_count"] for row in stats)
    total_high = sum(row["high_count"] for row in stats)
    total_unmapped = sum(row["unmapped_high_count"] for row in stats)
    summary = output_dir / args.summary_name
    summary.write_text(
        "# Stage 3.1 Mapping Run\n\n"
        f"- Slides processed: {len(stats)}\n"
        f"- Total low patches: {total_low}\n"
        f"- Total high patches: {total_high}\n"
        f"- Unmapped high patches: {total_unmapped}\n"
        f"- Unmapped fraction: {(total_unmapped / total_high) if total_high else 0.0:.6f}\n"
        f"- Cache directory: `{cache_dir}`\n\n"
        "The source feature H5 files, standalone coordinate H5 files, splits, and Stage 1 outputs are unchanged.\n",
        encoding="utf-8",
    )
    print(f"processed={len(stats)} output={output_dir} unmapped_high={total_unmapped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
