#!/usr/bin/env python3
"""
从 CSV 的 filename 列读取样本名，在源目录查找对应 SVS 并复制到目标目录。

默认参数：
- CSV: /xiangmu/ViLMIL/ViLa-MIL-main/dataset_csv/广州大学基线表-河源分割任务.csv
- 源目录: /mnt/nas/shared/medical/MedicalData/Adenocarcinoma/省人医数据20260206/河源
- 目标目录: /home/ljh/data/河源

说明：
- filename 可能不带后缀，脚本会自动匹配 .svs/.SVS
- 若同名（不含后缀）在源目录匹配到多个 SVS，会记为冲突并跳过
- 默认不覆盖目标目录中已存在文件；可加 --overwrite 开启覆盖
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VALID_EXTS = (".svs", ".SVS")


def load_csv_rows_with_fallback(csv_path: Path, preferred_encoding: str) -> tuple[list[dict], list[str], str]:
    encodings_to_try = []
    for enc in [preferred_encoding, "utf-8-sig", "utf-8", "gb18030", "gbk"]:
        if enc and enc not in encodings_to_try:
            encodings_to_try.append(enc)

    last_error = None
    for enc in encodings_to_try:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
            return rows, fieldnames, enc
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"无法解码 CSV，尝试过编码: {encodings_to_try}；最后错误: {last_error}",
    )


def resolve_source_file(source_dir: Path, filename_value: str) -> Path:
    raw = str(filename_value).strip()
    if not raw:
        raise FileNotFoundError("filename 为空")

    p = Path(raw)

    if p.suffix:
        exact = source_dir / p.name
        if exact.exists() and exact.suffix.lower() == ".svs":
            return exact

    stem = p.stem if p.suffix else p.name
    for ext in VALID_EXTS:
        candidate = source_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    # 兜底：大小写不敏感同 stem 搜索
    matches = [f for f in source_dir.glob(f"{stem}.*") if f.suffix.lower() == ".svs"]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise FileNotFoundError(f"匹配到多个同名 SVS，无法唯一确定：{names}")

    raise FileNotFoundError(f"未找到对应 SVS：{stem}.svs")


def main() -> int:
    parser = argparse.ArgumentParser(description="按 CSV filename 列复制 SVS 文件")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("/xiangmu/ViLMIL/ViLa-MIL-main/dataset_csv/广州大学基线表-河源分割任务.csv"),
        help="CSV 文件路径（需包含 filename 列）",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/mnt/nas/shared/medical/MedicalData/Adenocarcinoma/省人医数据20260206/河源"),
        help="源 SVS 目录",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("/home/ljh/data/河源"),
        help="目标目录",
    )
    parser.add_argument("--encoding", type=str, default="utf-8-sig", help="CSV 编码")
    parser.add_argument("--overwrite", action="store_true", help="目标已存在时覆盖")
    parser.add_argument("--dry-run", action="store_true", help="仅打印操作，不实际复制")
    parser.add_argument(
        "--fail-log",
        type=Path,
        default=None,
        help="失败记录输出文件（默认：<target-dir>/copy_failed_records.csv）",
    )
    args = parser.parse_args()

    csv_path = args.csv_path.resolve()
    source_dir = args.source_dir.resolve()
    target_dir = args.target_dir.resolve()
    fail_log_path = args.fail_log.resolve() if args.fail_log else (target_dir / "copy_failed_records.csv")

    if not csv_path.exists():
        print(f"[ERROR] CSV 不存在：{csv_path}")
        return 1
    if not source_dir.exists():
        print(f"[ERROR] 源目录不存在：{source_dir}")
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    copied = 0
    skipped_exists = 0
    missing_or_conflict = 0
    failed = 0
    seen_stems = set()
    failed_records = []

    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Source: {source_dir}")
    print(f"[INFO] Target: {target_dir}")
    print(f"[INFO] Dry-run: {args.dry_run}")

    try:
        rows, fieldnames, used_encoding = load_csv_rows_with_fallback(csv_path, args.encoding)
    except UnicodeDecodeError as exc:
        print(f"[ERROR] CSV 解码失败：{exc}")
        return 1

    print(f"[INFO] CSV Encoding used: {used_encoding}")

    if not fieldnames or "filename" not in fieldnames:
        print(f"[ERROR] CSV 缺少 filename 列。当前列：{fieldnames}")
        return 1

    iterator = rows
    if tqdm is not None:
        iterator = tqdm(rows, total=len(rows), ncols=110, desc="Copy SVS")

    for row in iterator:
        total_rows += 1
        filename_value = row.get("filename", "")

        try:
            src = resolve_source_file(source_dir, filename_value)
            stem = src.stem
            if stem in seen_stems:
                # CSV 里重复 filename 时避免重复复制
                continue
            seen_stems.add(stem)

            dst = target_dir / src.name

            if dst.exists() and not args.overwrite:
                print(f"[SKIP] 已存在：{dst.name}")
                skipped_exists += 1
                continue

            if args.dry_run:
                print(f"[PLAN] {src.name} -> {dst}")
                copied += 1
                if tqdm is not None:
                    iterator.set_postfix_str(f"success={copied} skip={skipped_exists} miss={missing_or_conflict} fail={failed}")
                continue

            shutil.copy2(src, dst)
            print(f"[OK] {src.name} -> {dst}")
            copied += 1
            if tqdm is not None:
                iterator.set_postfix_str(f"success={copied} skip={skipped_exists} miss={missing_or_conflict} fail={failed}")

        except FileNotFoundError as exc:
            print(f"[MISS] filename={filename_value} | {exc}")
            missing_or_conflict += 1
            failed_records.append({
                "filename": str(filename_value),
                "reason": str(exc),
                "type": "missing_or_conflict",
            })
            if tqdm is not None:
                iterator.set_postfix_str(f"success={copied} skip={skipped_exists} miss={missing_or_conflict} fail={failed}")
        except Exception as exc:
            print(f"[FAIL] filename={filename_value} | {exc}")
            failed += 1
            failed_records.append({
                "filename": str(filename_value),
                "reason": str(exc),
                "type": "runtime_error",
            })
            if tqdm is not None:
                iterator.set_postfix_str(f"success={copied} skip={skipped_exists} miss={missing_or_conflict} fail={failed}")

    if failed_records:
        fail_log_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_log_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "type", "reason"])
            writer.writeheader()
            writer.writerows(failed_records)
        print(f"[INFO] 已写入失败记录：{fail_log_path}")
    else:
        print("[INFO] 无失败记录。")

    print("\n========== SUMMARY ==========")
    print(f"CSV rows            : {total_rows}")
    print(f"Unique filenames    : {len(seen_stems)}")
    print(f"Copied/Planned      : {copied}")
    print(f"Skipped(exists)     : {skipped_exists}")
    print(f"Missing/Conflicts   : {missing_or_conflict}")
    print(f"Failed              : {failed}")
    print(f"Success count       : {copied}")
    print("=============================")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
