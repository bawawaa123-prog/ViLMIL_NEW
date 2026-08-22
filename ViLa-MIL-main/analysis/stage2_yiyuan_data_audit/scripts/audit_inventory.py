#!/usr/bin/env python3
"""Build the read-only Step 2.0 inventory for the Yiyuan dataset.

Run from the ViLa-MIL-main directory:
    python analysis/stage2_yiyuan_data_audit/scripts/audit_inventory.py

The script only reads the dataset CSV and the five configured data trees. It
does not open WSI pixel data or inspect coordinate/feature contents.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WSI_SUFFIXES = (".ome.tiff", ".ome.tif", ".svs", ".tiff", ".tif", ".ndpi", ".mrxs")
H5_SUFFIXES = (".h5", ".hdf5")
FEATURE_SUFFIXES = (".h5", ".hdf5", ".pt", ".pth")

CORE_INVENTORY_FIELDS = [
    "slide_id",
    "case_id",
    "label",
    "wsi_exists",
    "wsi_path",
    "coord_5x_exists",
    "coord_5x_path",
    "coord_20x_exists",
    "coord_20x_path",
    "feature_5x_exists",
    "feature_5x_path",
    "feature_20x_exists",
    "feature_20x_path",
    "is_complete",
]


@dataclass(frozen=True)
class FileKind:
    key: str
    label: str
    root: Path
    suffixes: tuple[str, ...]


@dataclass(frozen=True)
class DataFile:
    kind: str
    path: Path
    derived_slide_id: str
    normalized_slide_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the Yiyuan dataset file inventory")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="ViLa-MIL-main root (default: current directory, or inferred from this script)",
    )
    parser.add_argument("--csv", type=Path, default=Path("dataset_csv/all_data.csv"))
    parser.add_argument("--wsi-dir", type=Path, default=Path("data/yiyuan/wsi"))
    parser.add_argument("--coord-5x-dir", type=Path, default=Path("data/yiyuan/patches_coords_5x"))
    parser.add_argument("--coord-20x-dir", type=Path, default=Path("data/yiyuan/patches_coords_20x"))
    parser.add_argument("--feature-5x-dir", type=Path, default=Path("data/yiyuan/features_biomedclip_5x"))
    parser.add_argument("--feature-20x-dir", type=Path, default=Path("data/yiyuan/features_biomedclip_20x"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/00_inventory"),
    )
    return parser.parse_args()


def resolve_project_root(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "dataset_csv" / "all_data.csv").is_file():
        return cwd
    return Path(__file__).resolve().parents[3]


def under_root(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else project_root / value


def display_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def strip_known_suffix(filename: str, suffixes: Iterable[str]) -> str:
    name = filename.strip()
    lower_name = name.casefold()
    for suffix in sorted(suffixes, key=len, reverse=True):
        if lower_name.endswith(suffix.casefold()):
            return name[: -len(suffix)]
    return name


def normalize_slide_id(value: object, suffixes: Iterable[str] = ()) -> str:
    """Normalize only whitespace, a recognized extension, and character case."""
    slide_id = strip_known_suffix(str(value).strip(), suffixes)
    return slide_id.casefold()


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"CSV contains duplicate column names: {fields}")
        required = {"slide_id", "case_id", "label"}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    return fields, rows


def coordinate_data_dir(root: Path) -> Path:
    """Use the create_patches_fp.py coordinate output, excluding preview artifacts."""
    preferred = root / "patches_256"
    return preferred if preferred.is_dir() else root


def scan_kind(kind: FileKind) -> list[DataFile]:
    if not kind.root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {kind.root}")
    suffixes = tuple(s.casefold() for s in kind.suffixes)
    results = []
    for path in sorted(kind.root.rglob("*")):
        if not path.is_file() or not path.name.casefold().endswith(suffixes):
            continue
        derived = strip_known_suffix(path.name, kind.suffixes)
        results.append(
            DataFile(
                kind=kind.key,
                path=path,
                derived_slide_id=derived,
                normalized_slide_id=normalize_slide_id(derived),
            )
        )
    return results


def index_files(files: list[DataFile]) -> dict[str, list[DataFile]]:
    result: dict[str, list[DataFile]] = defaultdict(list)
    for item in files:
        result[item.normalized_slide_id].append(item)
    return result


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def joined_paths(matches: list[DataFile], project_root: Path) -> str:
    return ";".join(display_path(item.path, project_root) for item in matches)


def build_duplicate_rows(
    csv_fields: list[str],
    csv_rows: list[dict[str, str]],
    csv_key_rows: dict[str, list[int]],
    files_by_kind: dict[str, list[DataFile]],
    indexes: dict[str, dict[str, list[DataFile]]],
    project_root: Path,
) -> list[dict[str, object]]:
    duplicates: list[dict[str, object]] = []

    exact_slide_rows: dict[str, list[int]] = defaultdict(list)
    for row_number, row in enumerate(csv_rows, start=2):
        exact_slide_rows[row["slide_id"]].append(row_number)
    for slide_id, row_numbers in sorted(exact_slide_rows.items()):
        if len(row_numbers) > 1:
            duplicates.append({
                "issue_type": "csv_duplicate_slide_id",
                "file_type": "csv",
                "normalized_slide_id": normalize_slide_id(slide_id, WSI_SUFFIXES),
                "slide_ids": slide_id,
                "count": len(row_numbers),
                "paths": "",
                "row_numbers": ";".join(map(str, row_numbers)),
                "details": "The same literal slide_id occurs in multiple CSV rows.",
            })

    record_rows: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for row_number, row in enumerate(csv_rows, start=2):
        record_rows[tuple(row[field] for field in csv_fields)].append(row_number)
    for values, row_numbers in sorted(record_rows.items()):
        if len(row_numbers) > 1:
            record = dict(zip(csv_fields, values))
            duplicates.append({
                "issue_type": "csv_duplicate_record",
                "file_type": "csv",
                "normalized_slide_id": normalize_slide_id(record["slide_id"], WSI_SUFFIXES),
                "slide_ids": record["slide_id"],
                "count": len(row_numbers),
                "paths": "",
                "row_numbers": ";".join(map(str, row_numbers)),
                "details": "All CSV field values are identical.",
            })

    for normalized, row_indexes in sorted(csv_key_rows.items()):
        literal_ids = sorted({csv_rows[index]["slide_id"] for index in row_indexes})
        if len(literal_ids) > 1:
            duplicates.append({
                "issue_type": "normalized_csv_slide_id_collision",
                "file_type": "csv",
                "normalized_slide_id": normalized,
                "slide_ids": ";".join(literal_ids),
                "count": len(row_indexes),
                "paths": "",
                "row_numbers": ";".join(str(index + 2) for index in row_indexes),
                "details": "Distinct CSV slide_id values normalize to the same key.",
            })

    csv_keys = set(csv_key_rows)
    for kind, files in files_by_kind.items():
        by_key = indexes[kind]
        for normalized, matches in sorted(by_key.items()):
            if len(matches) <= 1:
                continue
            literal_ids = sorted({item.derived_slide_id for item in matches})
            common = {
                "file_type": kind,
                "normalized_slide_id": normalized,
                "slide_ids": ";".join(literal_ids),
                "count": len(matches),
                "paths": joined_paths(matches, project_root),
                "row_numbers": "",
            }
            if normalized in csv_keys:
                duplicates.append({
                    **common,
                    "issue_type": "multiple_files_same_type",
                    "details": "One CSV slide matches multiple files of this type.",
                })
            duplicates.append({
                **common,
                "issue_type": "normalized_filename_collision",
                "details": "Multiple filenames normalize to the same slide key.",
            })
    return duplicates


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    csv_path = under_root(project_root, args.csv)
    output_dir = under_root(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_fields, csv_rows = read_csv_rows(csv_path)
    csv_key_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(csv_rows):
        normalized = normalize_slide_id(row["slide_id"], WSI_SUFFIXES + H5_SUFFIXES)
        if not normalized:
            raise ValueError(f"CSV row {index + 2} has an empty slide_id")
        csv_key_rows[normalized].append(index)

    coord_5x_root = coordinate_data_dir(under_root(project_root, args.coord_5x_dir))
    coord_20x_root = coordinate_data_dir(under_root(project_root, args.coord_20x_dir))
    kinds = [
        FileKind("wsi", "WSI", under_root(project_root, args.wsi_dir), WSI_SUFFIXES),
        FileKind("coord_5x", "5x coordinates", coord_5x_root, H5_SUFFIXES),
        FileKind("coord_20x", "20x coordinates", coord_20x_root, H5_SUFFIXES),
        FileKind("feature_5x", "5x feature", under_root(project_root, args.feature_5x_dir), FEATURE_SUFFIXES),
        FileKind("feature_20x", "20x feature", under_root(project_root, args.feature_20x_dir), FEATURE_SUFFIXES),
    ]
    files_by_kind = {kind.key: scan_kind(kind) for kind in kinds}
    indexes = {key: index_files(files) for key, files in files_by_kind.items()}

    inventory_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for csv_row_number, row in enumerate(csv_rows, start=2):
        normalized = normalize_slide_id(row["slide_id"], WSI_SUFFIXES + H5_SUFFIXES)
        output: dict[str, object] = dict(row)
        output["normalized_slide_id"] = normalized
        missing_items = []
        warnings = []
        for kind in kinds:
            matches = indexes[kind.key].get(normalized, [])
            output[f"{kind.key}_exists"] = bool(matches)
            output[f"{kind.key}_path"] = joined_paths(matches, project_root)
            output[f"{kind.key}_match_count"] = len(matches)
            if not matches:
                missing_items.append(kind.label)
            elif len(matches) > 1:
                warnings.append(f"{kind.key}:multiple_matches={len(matches)}")
            elif matches[0].derived_slide_id != row["slide_id"]:
                warnings.append(
                    f"{kind.key}:normalized_match={matches[0].derived_slide_id!r}"
                )
        output["is_complete"] = not missing_items
        output["audit_warnings"] = ";".join(warnings)
        inventory_rows.append(output)
        if missing_items:
            missing_rows.append({
                "csv_row_number": csv_row_number,
                "slide_id": row["slide_id"],
                "case_id": row["case_id"],
                "label": row["label"],
                "missing_items": ";".join(missing_items),
                "missing_count": len(missing_items),
                "audit_warnings": ";".join(warnings),
            })

    inventory_extra_fields = [field for field in csv_fields if field not in {"slide_id", "case_id", "label"}]
    inventory_fields = CORE_INVENTORY_FIELDS + inventory_extra_fields + [
        "normalized_slide_id",
        "wsi_match_count",
        "coord_5x_match_count",
        "coord_20x_match_count",
        "feature_5x_match_count",
        "feature_20x_match_count",
        "audit_warnings",
    ]
    write_csv(output_dir / "dataset_inventory.csv", inventory_fields, inventory_rows)
    write_csv(
        output_dir / "missing_files.csv",
        ["csv_row_number", "slide_id", "case_id", "label", "missing_items", "missing_count", "audit_warnings"],
        missing_rows,
    )

    csv_keys = set(csv_key_rows)
    orphan_rows = []
    for kind in kinds:
        for item in files_by_kind[kind.key]:
            if item.normalized_slide_id not in csv_keys:
                orphan_rows.append({
                    "file_type": kind.key,
                    "path": display_path(item.path, project_root),
                    "filename": item.path.name,
                    "derived_slide_id": item.derived_slide_id,
                    "normalized_slide_id": item.normalized_slide_id,
                    "reason": "No normalized slide_id match in all_data.csv.",
                })
    write_csv(
        output_dir / "orphan_files.csv",
        ["file_type", "path", "filename", "derived_slide_id", "normalized_slide_id", "reason"],
        orphan_rows,
    )

    duplicate_rows = build_duplicate_rows(
        csv_fields, csv_rows, csv_key_rows, files_by_kind, indexes, project_root
    )
    duplicate_fields = [
        "issue_type", "file_type", "normalized_slide_id", "slide_ids", "count",
        "paths", "row_numbers", "details",
    ]
    write_csv(output_dir / "duplicate_slides.csv", duplicate_fields, duplicate_rows)

    label_counts = Counter(row["label"] for row in csv_rows)
    missing_counts = {
        kind.key: sum(not bool(indexes[kind.key].get(normalize_slide_id(row["slide_id"], WSI_SUFFIXES + H5_SUFFIXES))) for row in csv_rows)
        for kind in kinds
    }
    exact_slide_counts = Counter(row["slide_id"] for row in csv_rows)
    duplicate_slide_groups = sum(count > 1 for count in exact_slide_counts.values())
    duplicate_slide_excess = sum(count - 1 for count in exact_slide_counts.values() if count > 1)
    records = [tuple(row[field] for field in csv_fields) for row in csv_rows]
    duplicate_record_excess = len(records) - len(set(records))
    normalized_filename_collisions = sum(
        1 for row in duplicate_rows if row["issue_type"] == "normalized_filename_collision"
    )
    multiple_file_groups = sum(
        1 for row in duplicate_rows if row["issue_type"] == "multiple_files_same_type"
    )
    normalized_csv_collisions = sum(
        1 for row in duplicate_rows if row["issue_type"] == "normalized_csv_slide_id_collision"
    )
    warning_rows = sum(bool(row["audit_warnings"]) for row in inventory_rows)

    lines = [
        "Step 2.0 - Yiyuan dataset inventory and integrity audit",
        "=" * 64,
        f"Project root: {project_root}",
        f"Dataset CSV: {display_path(csv_path, project_root)}",
        f"CSV fields: {', '.join(csv_fields)}",
        "",
        f"CSV total records: {len(csv_rows)}",
        f"Unique slide_id count (literal): {len(exact_slide_counts)}",
        f"Unique slide_id count (normalized): {len(csv_key_rows)}",
        f"Unique case_id count: {len({row['case_id'] for row in csv_rows})}",
        "Slides by label:",
    ]
    lines.extend(f"  {label or '<empty>'}: {count}" for label, count in sorted(label_counts.items()))
    lines.extend([
        "",
        f"WSI file total: {len(files_by_kind['wsi'])}",
        f"5x coordinate file total: {len(files_by_kind['coord_5x'])}",
        f"20x coordinate file total: {len(files_by_kind['coord_20x'])}",
        f"5x feature file total: {len(files_by_kind['feature_5x'])}",
        f"20x feature file total: {len(files_by_kind['feature_20x'])}",
        "",
        f"Complete slide records: {sum(bool(row['is_complete']) for row in inventory_rows)}",
        f"Missing WSI count: {missing_counts['wsi']}",
        f"Missing 5x coordinate count: {missing_counts['coord_5x']}",
        f"Missing 20x coordinate count: {missing_counts['coord_20x']}",
        f"Missing 5x feature count: {missing_counts['feature_5x']}",
        f"Missing 20x feature count: {missing_counts['feature_20x']}",
        "",
        f"Orphan file count: {len(orphan_rows)}",
        f"Duplicate literal slide_id groups: {duplicate_slide_groups}",
        f"Duplicate slide_id excess rows: {duplicate_slide_excess}",
        f"Completely duplicate CSV excess records: {duplicate_record_excess}",
        f"Normalized CSV slide_id collision groups: {normalized_csv_collisions}",
        f"Multiple same-type file groups: {multiple_file_groups}",
        f"Normalized filename collision groups: {normalized_filename_collisions}",
        f"Inventory rows with naming/match warnings: {warning_rows}",
        "",
        "Major anomalies:",
    ])
    anomaly_lines = []
    if missing_rows:
        anomaly_lines.append(f"- {len(missing_rows)} CSV slide records have one or more missing file types.")
    if orphan_rows:
        anomaly_lines.append(f"- {len(orphan_rows)} files do not match a CSV slide_id.")
    if duplicate_rows:
        anomaly_lines.append(f"- {len(duplicate_rows)} duplicate/collision issue rows were recorded.")
    if warning_rows:
        anomaly_lines.append(f"- {warning_rows} inventory rows required non-exact or ambiguous matching.")
    lines.extend(anomaly_lines or ["- None."])
    lines.extend([
        "",
        "Scope note: this audit checks file presence and slide_id correspondence only;",
        "it does not validate WSI magnification/MPP, coordinate geometry, or feature contents.",
    ])
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote audit outputs to {display_path(output_dir, project_root)}")
    print(f"CSV records: {len(csv_rows)}; complete records: {sum(bool(row['is_complete']) for row in inventory_rows)}")
    print(f"Missing rows: {len(missing_rows)}; orphan files: {len(orphan_rows)}; duplicate issues: {len(duplicate_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
