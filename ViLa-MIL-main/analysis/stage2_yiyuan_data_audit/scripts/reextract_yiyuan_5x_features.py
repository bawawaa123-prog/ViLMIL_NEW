#!/usr/bin/env python3
"""Stage or commit the Step 2.3 Yiyuan nominal-5x feature repairs.

The default mode is a read-only preflight. ``--execute`` performs BiomedCLIP
inference but writes only to the audit staging directory. ``--commit-staged``
requires ``--yes`` and backs up every old feature before atomically replacing it.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import openslide


DEFAULT_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--alignment-csv", type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/03_coordinate_audit/coord_feature_alignment.csv"),
    )
    parser.add_argument("--wsi-dir", type=Path, default=Path("data/yiyuan/wsi"))
    parser.add_argument(
        "--coord-dir", type=Path, default=Path("data/yiyuan/patches_coords_5x/patches_256")
    )
    parser.add_argument(
        "--feature-dir", type=Path, default=Path("data/yiyuan/features_biomedclip_5x")
    )
    parser.add_argument(
        "--staging-dir", type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/03_coordinate_audit/feature_repair_staging_5x"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--execute", action="store_true", help="Run inference and create staged H5 files")
    parser.add_argument("--overwrite-staging", action="store_true")
    parser.add_argument("--commit-staged", action="store_true", help="Back up and replace production files")
    parser.add_argument("--yes", action="store_true", help="Required confirmation for --commit-staged")
    return parser.parse_args()


def project_root(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "dataset_csv" / "all_data.csv").is_file():
        return cwd
    return Path(__file__).resolve().parents[3]


def rooted(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def load_targets(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = [
        row["slide_id"] for row in rows
        if row.get("scale") == "5x" and row.get("all_three_counts_equal", "").lower() != "true"
    ]
    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate nominal-5x repair targets in alignment CSV")
    if not targets:
        raise ValueError("No nominal-5x count-mismatch targets found in alignment CSV")
    return targets


def resolve_wsi(wsi_dir: Path, slide_id: str) -> Path:
    matches = sorted(path for path in wsi_dir.glob(f"{slide_id}.*") if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one WSI for {slide_id}, found {len(matches)}: {matches}")
    return matches[0]


def load_coord_spec(path: Path) -> tuple[np.ndarray, int, int]:
    with h5py.File(path, "r") as handle:
        if "coords" not in handle:
            raise KeyError(f"Missing coords dataset: {path}")
        dataset = handle["coords"]
        coords = dataset[:]
        patch_level = int(dataset.attrs["patch_level"])
        patch_size = int(dataset.attrs["patch_size"])
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
        raise ValueError(f"Invalid coordinate shape {coords.shape}: {path}")
    if not np.issubdtype(coords.dtype, np.integer):
        raise TypeError(f"Coordinates must be integer, got {coords.dtype}: {path}")
    return coords, patch_level, patch_size


def validate_feature(path: Path, expected_coords: np.ndarray) -> tuple[bool, str]:
    try:
        with h5py.File(path, "r") as handle:
            if not {"coords", "features"}.issubset(handle.keys()):
                return False, "missing coords or features dataset"
            coords = handle["coords"][:]
            features = handle["features"]
            if coords.shape != expected_coords.shape or not np.array_equal(coords, expected_coords):
                return False, "coordinates are not exactly equal in coordinate-H5 order"
            if features.shape != (len(expected_coords), 512):
                return False, f"unexpected feature shape {features.shape}"
            if features.dtype != np.dtype("float32"):
                return False, f"unexpected feature dtype {features.dtype}"
            for start in range(0, len(features), 256):
                if not np.isfinite(features[start:start + 256]).all():
                    return False, "features contain NaN or infinity"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"


def preflight(
    targets: list[str], wsi_dir: Path, coord_dir: Path, feature_dir: Path, staging_dir: Path
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slide_id in targets:
        coord_path = coord_dir / f"{slide_id}.h5"
        old_feature = feature_dir / f"{slide_id}.h5"
        if not coord_path.is_file() or not old_feature.is_file():
            raise FileNotFoundError(f"Missing coordinate or current feature for {slide_id}")
        wsi_path = resolve_wsi(wsi_dir, slide_id)
        coords, patch_level, patch_size = load_coord_spec(coord_path)
        with openslide.OpenSlide(str(wsi_path)) as slide:
            if patch_level < 0 or patch_level >= slide.level_count:
                raise ValueError(f"Invalid patch level {patch_level} for {slide_id}")
            # One real read per slide catches unsupported/corrupt WSI input cheaply.
            slide.read_region(tuple(map(int, coords[0])), patch_level, (patch_size, patch_size)).load()
        with h5py.File(old_feature, "r") as handle:
            old_count = int(handle["features"].shape[0])
        staged = staging_dir / f"{slide_id}.h5"
        staged_ok, staged_detail = (False, "not present")
        if staged.is_file():
            staged_ok, staged_detail = validate_feature(staged, coords)
        rows.append({
            "slide_id": slide_id, "coordinate_count": len(coords), "current_feature_count": old_count,
            "missing_count": len(coords) - old_count, "patch_level": patch_level,
            "patch_size": patch_size, "wsi": wsi_path, "staged_ok": staged_ok,
            "staged_detail": staged_detail,
        })
    return rows


def print_preflight(rows: list[dict[str, object]], staging_dir: Path) -> None:
    print("slide_id\tcoords\tcurrent\tmissing\tlevel\tsize\tstaged")
    for row in rows:
        print(
            f"{row['slide_id']}\t{row['coordinate_count']}\t{row['current_feature_count']}\t"
            f"{row['missing_count']}\t{row['patch_level']}\t{row['patch_size']}\t{row['staged_ok']}"
        )
    print(f"Targets: {len(rows)}; patches to extract: {sum(int(r['coordinate_count']) for r in rows)}")
    print(f"Staging directory: {staging_dir}")


def extract_all(
    rows: list[dict[str, object]], coord_dir: Path, staging_dir: Path, batch_size: int,
    model_path: str, cache_dir: Path | None, overwrite: bool,
) -> None:
    import torch
    from PIL import ImageFile
    from torch.utils.data import DataLoader, Dataset

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from feature_extraction.patch_extraction_utils_biomedclip import (  # noqa: PLC0415
        _load_biomedclip_model,
        get_biomedclip_transforms,
    )

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this repair run; refusing slow CPU inference")
    model, _, _ = _load_biomedclip_model(
        model_path=model_path, cache_dir=str(cache_dir) if cache_dir else None, load_tokenizer=False
    )
    model = model.to(device).eval()
    transform = get_biomedclip_transforms()
    staging_dir.mkdir(parents=True, exist_ok=True)

    class SlideDataset(Dataset):
        def __init__(self, wsi_path: Path, coords: np.ndarray, level: int, size: int) -> None:
            self.slide = openslide.OpenSlide(str(wsi_path))
            self.coords = coords
            self.level = level
            self.size = size

        def __len__(self) -> int:
            return len(self.coords)

        def __getitem__(self, index: int):
            xy = tuple(map(int, self.coords[index]))
            image = self.slide.read_region(xy, self.level, (self.size, self.size)).convert("RGB")
            return transform(image)

        def close(self) -> None:
            self.slide.close()

    for row in rows:
        slide_id = str(row["slide_id"])
        destination = staging_dir / f"{slide_id}.h5"
        coords, level, size = load_coord_spec(coord_dir / f"{slide_id}.h5")
        if destination.exists() and not overwrite:
            ok, detail = validate_feature(destination, coords)
            if ok:
                print(f"SKIP valid staged file: {slide_id}")
                continue
            raise FileExistsError(f"Invalid staged file exists ({detail}); use --overwrite-staging: {destination}")

        dataset = SlideDataset(Path(row["wsi"]), coords, level, size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        batches: list[np.ndarray] = []
        try:
            print(f"EXTRACT {slide_id}: {len(coords)} patches")
            with torch.inference_mode():
                for batch in loader:
                    output = model.encode_image(batch.to(device, non_blocking=True))
                    batches.append(output.float().cpu().numpy())
        finally:
            dataset.close()
        features = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        temporary = destination.with_suffix(".h5.tmp")
        if temporary.exists():
            temporary.unlink()
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset("features", data=features, compression="gzip")
            handle.create_dataset("coords", data=coords.astype(np.int64, copy=False), compression="gzip")
            handle.attrs["step23_repair"] = True
            handle.attrs["source_wsi"] = Path(row["wsi"]).name
            handle.attrs["patch_level"] = level
            handle.attrs["patch_size"] = size
            handle.attrs["model"] = model_path
        os.replace(temporary, destination)
        ok, detail = validate_feature(destination, coords)
        if not ok:
            raise RuntimeError(f"Post-write validation failed for {slide_id}: {detail}")
        print(f"STAGED {destination}")


def commit_all(
    rows: list[dict[str, object]], coord_dir: Path, feature_dir: Path, staging_dir: Path,
    audit_dir: Path,
) -> Path:
    failures = []
    for row in rows:
        slide_id = str(row["slide_id"])
        coords, _, _ = load_coord_spec(coord_dir / f"{slide_id}.h5")
        ok, detail = validate_feature(staging_dir / f"{slide_id}.h5", coords)
        if not ok:
            failures.append(f"{slide_id}: {detail}")
    if failures:
        raise RuntimeError("Refusing commit; staged validation failed:\n" + "\n".join(failures))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = audit_dir / f"feature_repair_backup_5x_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    for row in rows:
        slide_id = str(row["slide_id"])
        current = feature_dir / f"{slide_id}.h5"
        backup = backup_dir / current.name
        shutil.copy2(current, backup)
        manifest_rows.append((slide_id, str(current), str(backup)))
    with (backup_dir / "commit_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slide_id", "replacement_path", "backup_path"])
        writer.writerows(manifest_rows)
    for row in rows:
        slide_id = str(row["slide_id"])
        current = feature_dir / f"{slide_id}.h5"
        staged = staging_dir / current.name
        os.replace(staged, current)
    return backup_dir


def main() -> int:
    args = parse_args()
    if args.execute and args.commit_staged:
        raise SystemExit("Choose either --execute or --commit-staged, not both")
    if args.commit_staged and not args.yes:
        raise SystemExit("--commit-staged requires explicit --yes")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    root = project_root(args.project_root)
    alignment = rooted(root, args.alignment_csv)
    wsi_dir = rooted(root, args.wsi_dir)
    coord_dir = rooted(root, args.coord_dir)
    feature_dir = rooted(root, args.feature_dir)
    staging_dir = rooted(root, args.staging_dir)
    targets = load_targets(alignment)
    rows = preflight(targets, wsi_dir, coord_dir, feature_dir, staging_dir)
    print_preflight(rows, staging_dir)

    if args.execute:
        extract_all(
            rows, coord_dir, staging_dir, args.batch_size, args.model_path,
            args.cache_dir.expanduser().resolve() if args.cache_dir else None, args.overwrite_staging,
        )
        print("Extraction completed in staging only; production feature files were not changed.")
    elif args.commit_staged:
        backup_dir = commit_all(rows, coord_dir, feature_dir, staging_dir, alignment.parent)
        print(f"Committed {len(rows)} repaired files. Backups: {backup_dir}")
    else:
        print("Read-only preflight complete. Add --execute to create staged candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
