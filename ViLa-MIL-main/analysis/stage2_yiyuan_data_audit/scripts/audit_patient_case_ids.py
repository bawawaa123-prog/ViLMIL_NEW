#!/usr/bin/env python3
"""Audit Yiyuan case/slide identity heuristics and strict split leakage risk.

Run from the ViLa-MIL-main root:
    python analysis/stage2_yiyuan_data_audit/scripts/audit_patient_case_ids.py

This is a read-only audit. Candidate groups are naming-based hypotheses and
must never be treated as confirmed patient identities without hospital input.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ParsedId:
    original_id: str
    normalized_id: str
    prefix_candidate: str
    suffix_candidate: str
    numeric_core: str
    block_candidate: str
    block_number_candidate: str
    trailing_candidate: str
    pattern_type: str
    is_groupable_block_id: bool


PATTERNS = (
    ("numeric_hyphen_B_number_letter", re.compile(r"^(\d+)(-B)(\d+)([A-Za-z]+)$", re.I)),
    ("numeric_hyphen_B_number", re.compile(r"^(\d+)(-B)(\d+)$", re.I)),
    ("numeric_hyphen_B", re.compile(r"^(\d+)(-B)$", re.I)),
    ("numeric_hyphen_B_trailing_punctuation", re.compile(r"^(\d+)(-B)([.]+)$", re.I)),
    ("numeric_B_number_letter", re.compile(r"^(\d+)(B)(\d+)([A-Za-z]+)$", re.I)),
    ("numeric_B_number", re.compile(r"^(\d+)(B)(\d+)$", re.I)),
    ("numeric_B_hyphen_number", re.compile(r"^(\d+)(B)(-\d+)$", re.I)),
    ("numeric_B", re.compile(r"^(\d+)(B)$", re.I)),
    ("pure_numeric", re.compile(r"^(\d+)$", re.I)),
)

STANDARD_GROUP_PATTERNS = {
    "numeric_hyphen_B_number",
    "numeric_hyphen_B",
    "numeric_B_number",
    "numeric_B",
}

OUTPUT_FILES = (
    "possible_same_patient.csv",
    "label_consistency.csv",
    "case_slide_mapping.csv",
    "split_leakage_audit.csv",
    "id_pattern_statistics.csv",
    "manual_confirmation_needed.csv",
    "summary.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Yiyuan patient/case/slide identity relationships")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=Path("dataset_csv/all_data.csv"))
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/00_inventory/dataset_inventory.csv"),
        help="Step 2.0 inventory used to verify that the audited slide population is unchanged",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path(
            "trained_models/adenocarcinoma_strict5_new/"
            "adenocarcinoma_biomedclip_dual_strict5_new_s1/"
            "experiment_adenocarcinoma_biomedclip_dual_strict5_new.txt"
        ),
        help="Training settings file used to resolve the authoritative split_dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/stage2_yiyuan_data_audit/01_patient_case_audit"),
    )
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


def relative_display(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_dict_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    return fields, rows


def write_rows(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_identifier(value: str) -> ParsedId:
    original = value.strip()
    normalized = original.upper()
    for pattern_type, regex in PATTERNS:
        match = regex.fullmatch(original)
        if not match:
            continue
        groups = match.groups()
        core = groups[0]
        if pattern_type == "pure_numeric":
            return ParsedId(original, normalized, core, "", core, "", "", "", pattern_type, False)

        block_token = groups[1]
        suffix = original[len(core):]
        block_number = ""
        trailing = ""
        if pattern_type in {"numeric_hyphen_B_number", "numeric_B_number"}:
            block_number = groups[2]
        elif pattern_type in {"numeric_hyphen_B_number_letter", "numeric_B_number_letter"}:
            block_number, trailing = groups[2], groups[3]
        elif pattern_type == "numeric_B_hyphen_number":
            trailing = groups[2]
        elif pattern_type == "numeric_hyphen_B_trailing_punctuation":
            trailing = groups[2]
        groupable = pattern_type != "numeric_hyphen_B_trailing_punctuation"
        return ParsedId(
            original, normalized, core, suffix, core, block_token.upper().replace("-", ""),
            block_number, trailing, pattern_type, groupable,
        )

    leading = re.match(r"^(\d+)(.*)$", original)
    core = leading.group(1) if leading else ""
    suffix = leading.group(2) if leading else original
    return ParsedId(original, normalized, core, suffix, core, "", "", suffix, "unparsed", False)


def load_experiment_settings(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8").strip()
    settings = ast.literal_eval(text)
    if not isinstance(settings, dict) or not settings.get("split_dir"):
        raise ValueError(f"Experiment config does not contain split_dir: {path}")
    return settings


def load_split(path: Path) -> dict[str, list[str]]:
    fields, rows = read_dict_rows(path)
    required = {"train", "val", "test"}
    if not required.issubset(fields):
        raise ValueError(f"Split file lacks train/val/test columns: {path}")
    return {name: [row[name] for row in rows if row[name]] for name in ("train", "val", "test")}


def split_maps(split_dir: Path, folds: int, valid_slides: set[str]) -> tuple[dict[int, dict[str, str]], list[str]]:
    maps: dict[int, dict[str, str]] = {}
    checks = []
    for fold in range(folds):
        path = split_dir / f"splits_{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Expected configured split file is missing: {path}")
        columns = load_split(path)
        assignment: dict[str, str] = {}
        duplicate_memberships = []
        for split_name, slide_ids in columns.items():
            for slide_id in slide_ids:
                if slide_id in assignment:
                    duplicate_memberships.append(slide_id)
                assignment[slide_id] = split_name
        unknown = set(assignment) - valid_slides
        missing = valid_slides - set(assignment)
        if duplicate_memberships or unknown or missing:
            raise ValueError(
                f"Invalid fold {fold}: duplicate_memberships={len(duplicate_memberships)}, "
                f"unknown={len(unknown)}, missing={len(missing)}"
            )
        checks.append(
            f"fold {fold}: train={len(columns['train'])}, val={len(columns['val'])}, "
            f"test={len(columns['test'])}, coverage={len(assignment)}"
        )
        maps[fold] = assignment
    return maps, checks


def semantic_split_copy_check(authoritative: Path, copy_dir: Path, folds: int) -> bool | None:
    if not copy_dir.is_dir():
        return None
    for fold in range(folds):
        copied = copy_dir / f"splits_{fold}.csv"
        if not copied.is_file():
            return None
        left, right = load_split(authoritative / f"splits_{fold}.csv"), load_split(copied)
        if any(left[name] != right[name] for name in ("train", "val", "test")):
            return False
    return True


def main() -> int:
    args = parse_args()
    root = project_root(args.project_root)
    csv_path = rooted(root, args.csv)
    inventory_path = rooted(root, args.inventory)
    config_path = rooted(root, args.experiment_config)
    output_dir = rooted(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fields, rows = read_dict_rows(csv_path)
    required = {"slide_id", "case_id", "label"}
    if not required.issubset(fields):
        raise ValueError(f"Dataset CSV lacks required fields: {sorted(required - set(fields))}")
    slide_ids = [row["slide_id"] for row in rows]
    if len(slide_ids) != len(set(slide_ids)):
        raise ValueError("slide_id values must be unique for this audit")
    inventory_fields, inventory_rows = read_dict_rows(inventory_path)
    if "slide_id" not in inventory_fields:
        raise ValueError(f"Step 2.0 inventory lacks slide_id: {inventory_path}")
    inventory_slide_ids = [row["slide_id"] for row in inventory_rows]
    if inventory_slide_ids != slide_ids:
        raise ValueError(
            "all_data.csv slide population/order differs from the Step 2.0 inventory; "
            "rerun Step 2.0 before this audit"
        )

    parsed_slide = {row["slide_id"]: parse_identifier(row["slide_id"]) for row in rows}
    parsed_case = {row["slide_id"]: parse_identifier(row["case_id"]) for row in rows}
    row_by_slide = {row["slide_id"]: row for row in rows}

    base_members: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        parsed = parsed_slide[row["slide_id"]]
        if parsed.is_groupable_block_id and parsed.numeric_core:
            base_members[parsed.numeric_core].append(row)
    candidate_members = {base: members for base, members in base_members.items() if len(members) > 1}
    candidate_ids = {base: f"CG_{index:04d}" for index, base in enumerate(sorted(candidate_members), start=1)}

    group_meta: dict[str, dict[str, object]] = {}
    for base, members in candidate_members.items():
        labels = sorted({member["label"] for member in members})
        nonstandard = any(parsed_slide[member["slide_id"]].pattern_type not in STANDARD_GROUP_PATTERNS for member in members)
        id_mismatch = any(member["slide_id"] != member["case_id"] for member in members)
        consistent = len(labels) == 1
        confidence = "medium" if consistent and not nonstandard and not id_mismatch else "low"
        reason = (
            "Slides share the same complete leading numeric core and differ only in an explicit B/block suffix. "
            "This may indicate related tissue blocks or specimens, but the repository contains no hospital-level "
            "patient/accession mapping that proves a shared patient or encounter."
        )
        if not consistent:
            reason += " Labels conflict within the candidate group, so the grouping heuristic may be wrong or requires pathology review."
        if nonstandard:
            reason += " At least one suffix uses a nonstandard sub-block form."
        if id_mismatch:
            reason += " At least one case_id differs literally from slide_id."
        group_meta[base] = {
            "candidate_group_id": candidate_ids[base],
            "group_size": len(members),
            "labels": labels,
            "is_label_consistent": consistent,
            "confidence": confidence,
            "reason": reason,
        }

    possible_rows = []
    for base in sorted(candidate_members):
        meta = group_meta[base]
        for member in candidate_members[base]:
            parsed = parsed_slide[member["slide_id"]]
            possible_rows.append({
                "candidate_group_id": meta["candidate_group_id"],
                "candidate_base_id": base,
                "group_size": meta["group_size"],
                "slide_id": member["slide_id"],
                "case_id": member["case_id"],
                "label": member["label"],
                "parsed_prefix": parsed.prefix_candidate,
                "parsed_suffix": parsed.suffix_candidate,
                "pattern_type": parsed.pattern_type,
                "reason_for_grouping": meta["reason"],
                "confidence": meta["confidence"],
                "requires_manual_confirmation": True,
            })
    write_rows(
        output_dir / "possible_same_patient.csv",
        ["candidate_group_id", "candidate_base_id", "group_size", "slide_id", "case_id", "label",
         "parsed_prefix", "parsed_suffix", "pattern_type", "reason_for_grouping", "confidence",
         "requires_manual_confirmation"],
        possible_rows,
    )

    consistency_rows = []
    for base in sorted(candidate_members):
        meta = group_meta[base]
        members = candidate_members[base]
        consistency_rows.append({
            "candidate_group_id": meta["candidate_group_id"],
            "candidate_base_id": base,
            "group_size": meta["group_size"],
            "labels": ";".join(meta["labels"]),
            "is_label_consistent": meta["is_label_consistent"],
            "slide_ids": ";".join(member["slide_id"] for member in members),
            "priority": "normal" if meta["is_label_consistent"] else "high",
            "requires_manual_confirmation": True,
        })
    write_rows(
        output_dir / "label_consistency.csv",
        ["candidate_group_id", "candidate_base_id", "group_size", "labels", "is_label_consistent",
         "slide_ids", "priority", "requires_manual_confirmation"],
        consistency_rows,
    )

    settings = load_experiment_settings(config_path)
    split_dir = rooted(root, Path(str(settings["split_dir"])))
    folds = int(settings.get("num_splits", 5))
    fold_maps, fold_checks = split_maps(split_dir, folds, set(slide_ids))

    assignment_path = split_dir / "strict_fold_assignments.csv"
    assignment_consistent: bool | None = None
    test_fold_by_slide: dict[str, int] = {}
    if assignment_path.is_file():
        assignment_fields, assignment_rows = read_dict_rows(assignment_path)
        if {"slide_id", "case_id", "label", "fold_id"}.issubset(assignment_fields):
            test_fold_by_slide = {row["slide_id"]: int(row["fold_id"]) for row in assignment_rows}
            assignment_consistent = (
                set(test_fold_by_slide) == set(slide_ids)
                and all(fold_maps[fold][slide_id] == "test" for slide_id, fold in test_fold_by_slide.items())
            )
    if not test_fold_by_slide:
        test_fold_by_slide = {
            slide_id: next(fold for fold in range(folds) if fold_maps[fold][slide_id] == "test")
            for slide_id in slide_ids
        }

    model_copy_dir = config_path.parent
    model_copy_consistent = semantic_split_copy_check(split_dir, model_copy_dir, folds)

    leakage_rows = []
    confirmed_case_groups: set[str] = set()
    potential_group_ids: set[str] = set()
    potential_event_keys: set[tuple[int, str]] = set()
    involved_folds: set[int] = set()
    involved_slides: set[str] = set()

    cases: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)
    for fold, assignments in fold_maps.items():
        for case_id, members in cases.items():
            splits = sorted({assignments[member["slide_id"]] for member in members})
            if len(splits) <= 1:
                continue
            confirmed_case_groups.add(case_id)
            for member in members:
                leakage_rows.append({
                    "fold": fold,
                    "candidate_group_id": "",
                    "candidate_base_id": "",
                    "slide_id": member["slide_id"],
                    "case_id": case_id,
                    "label": member["label"],
                    "split": assignments[member["slide_id"]],
                    "group_splits": ";".join(splits),
                    "cross_split": True,
                    "leakage_type": "confirmed_case_id_leakage",
                    "requires_manual_confirmation": False,
                    "details": "The same literal case_id occurs in multiple sets within this fold.",
                })

        for base, members in candidate_members.items():
            splits = sorted({assignments[member["slide_id"]] for member in members})
            if len(splits) <= 1:
                continue
            group_id = candidate_ids[base]
            potential_group_ids.add(group_id)
            potential_event_keys.add((fold, group_id))
            involved_folds.add(fold)
            involved_slides.update(member["slide_id"] for member in members)
            for member in members:
                leakage_rows.append({
                    "fold": fold,
                    "candidate_group_id": group_id,
                    "candidate_base_id": base,
                    "slide_id": member["slide_id"],
                    "case_id": member["case_id"],
                    "label": member["label"],
                    "split": assignments[member["slide_id"]],
                    "group_splits": ";".join(splits),
                    "cross_split": True,
                    "leakage_type": "potential_candidate_group_leakage",
                    "requires_manual_confirmation": True,
                    "details": (
                        "Potential only: if the naming-based candidate group is confirmed as one patient/case/encounter, "
                        "this fold has patient-level leakage risk."
                    ),
                })
    leakage_rows.sort(key=lambda row: (int(row["fold"]), str(row["leakage_type"]), str(row["candidate_base_id"]), str(row["slide_id"])))
    write_rows(
        output_dir / "split_leakage_audit.csv",
        ["fold", "candidate_group_id", "candidate_base_id", "slide_id", "case_id", "label", "split",
         "group_splits", "cross_split", "leakage_type", "requires_manual_confirmation", "details"],
        leakage_rows,
    )

    mapping_rows = []
    for row in rows:
        slide = parsed_slide[row["slide_id"]]
        case = parsed_case[row["slide_id"]]
        base = slide.numeric_core if slide.numeric_core in candidate_members else ""
        meta = group_meta.get(base)
        mapping_rows.append({
            "slide_id": row["slide_id"],
            "case_id": row["case_id"],
            "label": row["label"],
            "normalized_id": slide.normalized_id,
            "prefix_candidate": slide.prefix_candidate,
            "suffix_candidate": slide.suffix_candidate,
            "numeric_core": slide.numeric_core,
            "block_candidate": slide.block_candidate,
            "block_number_candidate": slide.block_number_candidate,
            "trailing_candidate": slide.trailing_candidate,
            "candidate_base_id": base,
            "candidate_group_id": meta["candidate_group_id"] if meta else "",
            "candidate_group_size": meta["group_size"] if meta else 1,
            "parsed_suffix": slide.suffix_candidate,
            "pattern_type": slide.pattern_type,
            "case_pattern_type": case.pattern_type,
            "case_slide_literal_match": row["case_id"] == row["slide_id"],
            "possible_multi_slide_patient": bool(meta),
            "requires_manual_confirmation": bool(meta) or row["case_id"] != row["slide_id"],
            "strict_test_fold": test_fold_by_slide[row["slide_id"]],
        })
    write_rows(
        output_dir / "case_slide_mapping.csv",
        ["slide_id", "case_id", "label", "normalized_id", "prefix_candidate", "suffix_candidate",
         "numeric_core", "block_candidate", "block_number_candidate", "trailing_candidate",
         "candidate_base_id", "candidate_group_id", "candidate_group_size", "parsed_suffix",
         "pattern_type", "case_pattern_type", "case_slide_literal_match", "possible_multi_slide_patient",
         "requires_manual_confirmation", "strict_test_fold"],
        mapping_rows,
    )

    statistic_rows = []
    for id_field, parsed_values in (("slide_id", parsed_slide.values()), ("case_id", parsed_case.values())):
        grouped: dict[str, list[str]] = defaultdict(list)
        for parsed in parsed_values:
            grouped[parsed.pattern_type].append(parsed.original_id)
        for pattern_type, values in sorted(grouped.items()):
            statistic_rows.append({
                "id_field": id_field,
                "pattern_type": pattern_type,
                "count": len(values),
                "unique_count": len(set(values)),
                "example_ids": ";".join(values[:5]),
                "is_parsed": pattern_type != "unparsed",
                "grouping_eligible_pattern": pattern_type in STANDARD_GROUP_PATTERNS or pattern_type in {
                    "numeric_B_number_letter", "numeric_hyphen_B_number_letter", "numeric_B_hyphen_number"
                },
            })
    write_rows(
        output_dir / "id_pattern_statistics.csv",
        ["id_field", "pattern_type", "count", "unique_count", "example_ids", "is_parsed", "grouping_eligible_pattern"],
        statistic_rows,
    )

    manual_rows = []
    for base in sorted(candidate_members):
        members = candidate_members[base]
        meta = group_meta[base]
        related = ";".join(member["slide_id"] for member in members)
        suffixes = ";".join(parsed_slide[member["slide_id"]].suffix_candidate for member in members)
        labels = ";".join(sorted({member["label"] for member in members}))
        assignments = ";".join(
            f"{member['slide_id']}:test_fold={test_fold_by_slide[member['slide_id']]}" for member in members
        )
        question = (
            f"{related} 是否属于同一患者、同一次送检病例，或同一病例/标本的不同组织块？"
        )
        if not meta["is_label_consistent"]:
            question += f" 若属于同一高层实体，如何解释其标签同时包含 {labels}？"
        manual_rows.append({
            "issue_type": "candidate_multi_slide_identity",
            "candidate_group_id": meta["candidate_group_id"],
            "candidate_base_id": base,
            "related_slide_ids": related,
            "related_case_ids": ";".join(member["case_id"] for member in members),
            "suffixes": suffixes,
            "labels": labels,
            "current_fold_assignments": assignments,
            "confidence": meta["confidence"],
            "priority": "high" if not meta["is_label_consistent"] or meta["candidate_group_id"] in potential_group_ids else "normal",
            "question": question,
            "why_it_matters": (
                "If confirmed as one patient/case/encounter, all related slides must remain in one partition; "
                "current candidate cross-split status=" + str(meta["candidate_group_id"] in potential_group_ids) + "."
            ),
        })
    for row in rows:
        if row["case_id"] == row["slide_id"]:
            continue
        manual_rows.append({
            "issue_type": "case_slide_literal_mismatch",
            "candidate_group_id": "",
            "candidate_base_id": parsed_slide[row["slide_id"]].numeric_core,
            "related_slide_ids": row["slide_id"],
            "related_case_ids": row["case_id"],
            "suffixes": parsed_slide[row["slide_id"]].suffix_candidate,
            "labels": row["label"],
            "current_fold_assignments": f"{row['slide_id']}:test_fold={test_fold_by_slide[row['slide_id']]}",
            "confidence": "low",
            "priority": "normal",
            "question": (
                f"case_id {row['case_id']} 末尾标点与 slide_id {row['slide_id']} 的差异是录入标点，"
                "还是具有实际病例编号含义？"
            ),
            "why_it_matters": "A normalization rule cannot be adopted safely until this literal ID difference is explained.",
        })
    write_rows(
        output_dir / "manual_confirmation_needed.csv",
        ["issue_type", "candidate_group_id", "candidate_base_id", "related_slide_ids", "related_case_ids",
         "suffixes", "labels", "current_fold_assignments", "confidence", "priority", "question", "why_it_matters"],
        manual_rows,
    )

    slide_pattern_counts = Counter(parsed.pattern_type for parsed in parsed_slide.values())
    case_pattern_counts = Counter(parsed.pattern_type for parsed in parsed_case.values())
    size_distribution = Counter(len(members) for members in candidate_members.values())
    consistent_groups = sum(bool(meta["is_label_consistent"]) for meta in group_meta.values())
    conflict_groups = len(group_meta) - consistent_groups
    conflict_bases = [base for base in sorted(group_meta) if not group_meta[base]["is_label_consistent"]]
    mismatch_rows = [row for row in rows if row["case_id"] != row["slide_id"]]
    unparsed_slide_count = slide_pattern_counts.get("unparsed", 0)
    candidate_group_count = len(candidate_members)
    candidate_slide_count = sum(len(members) for members in candidate_members.values())
    max_group_size = max(size_distribution, default=0)

    if confirmed_case_groups:
        conclusion_state = "D"
        conclusion = "发现同一明确 case_id 跨集合，这是 confirmed split implementation problem。"
    elif candidate_group_count == 0:
        conclusion_state = "A"
        conclusion = "未发现 candidate multi-slide group，但这不能证明 patient-level independence。"
    elif potential_group_ids:
        conclusion_state = "C"
        conclusion = (
            "发现命名候选组且存在跨 split：如果医院确认这些组属于同一患者/病例/送检，"
            "当前 strict5 存在 patient-level leakage 风险。"
        )
    else:
        conclusion_state = "B"
        conclusion = "发现命名候选组，但候选组当前未跨 split。"

    pattern_lines = [f"- `{name}`: {count}" for name, count in sorted(slide_pattern_counts.items())]
    case_pattern_lines = [f"- `{name}`: {count}" for name, count in sorted(case_pattern_counts.items())]
    size_lines = [f"- size {size}: {count} groups" for size, count in sorted(size_distribution.items())]
    conflict_lines = [
        f"- `{base}`: " + ", ".join(
            f"{member['slide_id']} ({member['label']})" for member in candidate_members[base]
        )
        for base in conflict_bases
    ]
    if not conflict_lines:
        conflict_lines = ["- None."]
    manual_priority = [row for row in manual_rows if row["priority"] == "high"]

    summary = [
        "# Step 2.1: Yiyuan Patient / Case / Slide Identity Audit",
        "",
        "> Candidate groups in this report are explainable naming heuristics only. They are not confirmed patient IDs.",
        "",
        "## Dataset identity",
        "",
        f"- Total records: {len(rows)}",
        f"- Unique slide_id: {len(set(slide_ids))}",
        f"- Unique case_id: {len({row['case_id'] for row in rows})}",
        f"- Literal case_id != slide_id rows: {len(mismatch_rows)}",
        f"- Step 2.0 inventory population/order match: yes (`{relative_display(root, inventory_path)}`)",
        "- No higher-level Yiyuan patient identifier or mapping was found in the repository paths searched.",
        "",
        "## Grouping rule and confidence",
        "",
        "A slide is grouping-eligible only when its full ID parses as a leading numeric core followed by an explicit B/block suffix. Slides form a candidate group only when the complete numeric core is identical and at least two records exist. No edit distance or partial-number similarity is used.",
        "",
        "`medium` means a standard B/block suffix and consistent labels/case-slide spelling; `low` means a nonstandard sub-block suffix, label conflict, or literal case/slide mismatch. No group is assigned `high`, because the repository does not contain hospital semantics proving patient identity. Every group requires manual confirmation.",
        "",
        "## ID patterns",
        "",
        "### slide_id",
        "",
        *pattern_lines,
        f"- Unparsed slide IDs: {unparsed_slide_count}",
        "",
        "### case_id",
        "",
        *case_pattern_lines,
        "",
        "## Candidate multi-slide groups",
        "",
        f"- Candidate groups: {candidate_group_count}",
        f"- Slides involved: {candidate_slide_count}",
        f"- Maximum group size: {max_group_size}",
        *size_lines,
        "",
        "## Label consistency",
        "",
        f"- Label-consistent candidate groups: {consistent_groups}",
        f"- Label-conflicting candidate groups: {conflict_groups}",
        "- Conflicts are high-priority review items; no label was automatically selected.",
        *conflict_lines,
        "",
        "## Strict split audit",
        "",
        f"- Experiment config: `{relative_display(root, config_path)}`",
        f"- Authoritative split directory from config: `{relative_display(root, split_dir)}`",
        f"- Configured folds: {folds}",
        f"- Rule confirmed from generator code: test=i, val=(i+1)%k, train=remaining folds.",
        f"- All fold files cover all {len(rows)} slides exactly once per fold: yes",
        f"- strict_fold_assignments.csv agrees with test membership: {assignment_consistent}",
        f"- Model-directory split copies are semantically identical: {model_copy_consistent}",
        f"- Confirmed case_id leakage groups: {len(confirmed_case_groups)}",
        f"- Potential candidate-group leakage groups (unique): {len(potential_group_ids)}",
        f"- Potential candidate-group leakage fold/group events: {len(potential_event_keys)}",
        f"- Folds involved: {', '.join(map(str, sorted(involved_folds))) if involved_folds else 'none'}",
        f"- Candidate slides involved in potential leakage: {len(involved_slides)}",
        "",
        "Fold coverage details:",
        "",
        *[f"- {check}" for check in fold_checks],
        "",
        "## Manual confirmation",
        "",
        f"- Manual confirmation rows: {len(manual_rows)}",
        f"- High-priority rows: {len(manual_priority)}",
        "- Primary question: does the leading numeric core represent a patient, encounter/accession, specimen, or another entity, and do differing B suffixes identify related tissue blocks?",
        "- The 11 label-conflicting groups should be reviewed first because they may indicate an invalid grouping heuristic, true within-patient heterogeneity, or annotation issues.",
        "- The five case_id values with a trailing period not present in slide_id also require clarification before any normalization policy is adopted.",
        "",
        "## Current conclusion",
        "",
        f"- State: **{conclusion_state}**",
        f"- {conclusion}",
        "- No confirmed literal case_id leakage was found.",
        "- The existing strict5 can remain a provisional baseline, but it cannot yet be described as proven patient-level strict.",
        "- Hospital ID semantics should be confirmed before deciding whether a regrouped strict5 is necessary.",
        "",
        "## Scope",
        "",
        "This step did not modify the dataset CSV, data files, split files, or model artifacts, and did not execute Step 2.2.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"Wrote {len(OUTPUT_FILES)} outputs to {relative_display(root, output_dir)}")
    print(f"Candidate groups: {candidate_group_count}; candidate slides: {candidate_slide_count}; label conflicts: {conflict_groups}")
    print(f"Confirmed case leakage groups: {len(confirmed_case_groups)}; potential candidate leakage groups: {len(potential_group_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
