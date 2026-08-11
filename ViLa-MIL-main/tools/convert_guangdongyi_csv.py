#!/usr/bin/env python3
"""
通用 CSV 转换脚本：将任意基线表转换为 ViLa-MIL 所需的三列格式。

输出格式固定为：
- case_id
- slide_id
- label

你可以在命令行中自由指定：
- 哪一列映射到 case_id
- 哪一列映射到 slide_id
- 哪一列映射到 label
- 输入文件路径
- 输出文件路径
- 标签值映射关系

常用示例：
1. 广东医：`WSI filename` 同时作为 case_id 和 slide_id
   python3 ViLa-MIL-main/tools/convert_guangdongyi_csv.py \
       --input 'ViLa-MIL-main/dataset_csv/广州大学基线表-广东医20260602.csv' \
       --output 'ViLa-MIL-main/dataset_csv/all_data_guangdongyi.csv' \
       --id-column 'WSI filename' \
       --label-column 'Adenocarcinoma' \
       --label-map '1=Adenocarcinoma' \
       --label-map '0=NonAdenocarcinoma'

2. 汕头中心医院：`文件名` 同时作为 case_id 和 slide_id
   python3 ViLa-MIL-main/tools/convert_guangdongyi_csv.py \
       --input 'ViLa-MIL-main/dataset_csv/广州大学基线表-汕头中心医院20260603.csv' \
       --output 'ViLa-MIL-main/dataset_csv/all_data_shantou.csv' \
       --id-column '文件名' \
       --label-column 'Adenocarcinoma' \
       --label-map '1=Adenocarcinoma' \
       --label-map '0=NonAdenocarcinoma'

3. case_id 和 slide_id 分别来自不同列
   python3 ViLa-MIL-main/tools/convert_guangdongyi_csv.py \
       --input input.csv \
       --output output.csv \
       --case-id-column '病例号' \
       --slide-id-column '切片号' \
       --label-column '标签列' \
       --label-map '阳性=Adenocarcinoma' \
       --label-map '阴性=NonAdenocarcinoma'

4. 如果原始标签已经是目标标签，可保留原值
   python3 ViLa-MIL-main/tools/convert_guangdongyi_csv.py \
       --input input.csv \
       --output output.csv \
       --id-column 'filename' \
       --label-column 'label' \
       --no-default-label-map \
       --keep-unmapped-label
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_LABEL_MAP_ITEMS = (
    "1=Adenocarcinoma",
    "0=NonAdenocarcinoma",
)
OUTPUT_COLUMNS = ("case_id", "slide_id", "label")


def load_rows_with_fallback(csv_path: Path, preferred_encoding: str) -> tuple[list[dict[str, str]], list[str], str]:
    encodings_to_try: list[str] = []
    for encoding in [preferred_encoding, "utf-8-sig", "utf-8", "gb18030", "gbk"]:
        if encoding and encoding not in encodings_to_try:
            encodings_to_try.append(encoding)

    last_error: Exception | None = None
    for encoding in encodings_to_try:
        try:
            with csv_path.open("r", encoding=encoding, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
            return rows, fieldnames, encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"无法解码 CSV：{csv_path}；尝试编码 {encodings_to_try}；最后错误：{last_error}")


def normalize_value(raw_value: object) -> str:
    value = "" if raw_value is None else str(raw_value).strip()
    if value.endswith(".0"):
        integer_part = value[:-2]
        if integer_part.lstrip("-").isdigit():
            return integer_part
    return value


def parse_label_map(label_map_items: Iterable[str]) -> dict[str, str]:
    label_map: dict[str, str] = {}
    for item in label_map_items:
        if "=" not in item:
            raise ValueError(f"标签映射格式错误：{item!r}，应为 源值=目标值")
        src, dst = item.split("=", 1)
        src_value = normalize_value(src)
        dst_value = dst.strip()
        if not src_value:
            raise ValueError(f"标签映射左侧不能为空：{item!r}")
        if not dst_value:
            raise ValueError(f"标签映射右侧不能为空：{item!r}")
        label_map[src_value] = dst_value
    return label_map


def resolve_id_columns(args: argparse.Namespace) -> tuple[str, str]:
    case_id_column = args.case_id_column
    slide_id_column = args.slide_id_column

    if args.id_column:
        if not case_id_column:
            case_id_column = args.id_column
        if not slide_id_column:
            slide_id_column = args.id_column

    if not case_id_column or not slide_id_column:
        raise ValueError("必须提供 --id-column，或同时提供 --case-id-column 和 --slide-id-column")

    return case_id_column, slide_id_column


def convert_rows(
    rows: list[dict[str, str]],
    case_id_column: str,
    slide_id_column: str,
    label_column: str,
    label_map: dict[str, str],
    keep_unmapped_label: bool,
    default_label: str | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    converted: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    for row_index, row in enumerate(rows, start=2):
        case_id = normalize_value(row.get(case_id_column, ""))
        slide_id = normalize_value(row.get(slide_id_column, ""))
        raw_label = normalize_value(row.get(label_column, ""))

        if not case_id:
            skipped.append({"row": str(row_index), "reason": f"{case_id_column} 为空"})
            continue
        if not slide_id:
            skipped.append({"row": str(row_index), "reason": f"{slide_id_column} 为空"})
            continue

        if not raw_label:
            if default_label is not None:
                label = default_label
            else:
                skipped.append({"row": str(row_index), "reason": f"{label_column} 为空"})
                continue
        elif raw_label in label_map:
            label = label_map[raw_label]
        elif keep_unmapped_label:
            label = raw_label
        else:
            skipped.append({"row": str(row_index), "reason": f"未命中标签映射：{raw_label!r}"})
            continue

        converted.append({
            "case_id": case_id,
            "slide_id": slide_id,
            "label": label,
        })

    return converted, skipped


def write_output(rows: list[dict[str, str]], output_path: Path, output_encoding: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding=output_encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def write_skip_log(skipped_rows: list[dict[str, str]], skip_log_path: Path, output_encoding: str) -> None:
    skip_log_path.parent.mkdir(parents=True, exist_ok=True)
    with skip_log_path.open("w", encoding=output_encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row", "reason"])
        writer.writeheader()
        writer.writerows(skipped_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将任意 CSV 转换为 ViLa-MIL 所需的 case_id, slide_id, label 三列格式",
        epilog=(
            "使用方法摘要:\n"
            "1. 用 --input 指定源 CSV\n"
            "2. 用 --output 指定目标 CSV\n"
            "3. 用 --id-column 指定同一列映射到 case_id/slide_id，或分别传 --case-id-column 和 --slide-id-column\n"
            "4. 用 --label-column 指定标签列\n"
            "5. 用 --label-map '源值=目标值' 配置标签映射，可重复传参\n"
            "\n"
            "示例:\n"
            "python3 ViLa-MIL-main/tools/convert_guangdongyi_csv.py --input input.csv --output output.csv --id-column '文件名' --label-column 'Adenocarcinoma' --label-map '1=Adenocarcinoma' --label-map '0=NonAdenocarcinoma'"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="源 CSV 文件路径")
    parser.add_argument("--output", type=Path, required=True, help="目标 CSV 文件路径")
    parser.add_argument("--encoding", default="utf-8-sig", help="优先尝试的源 CSV 编码，默认 utf-8-sig")
    parser.add_argument("--output-encoding", default="utf-8-sig", help="输出 CSV 编码，默认 utf-8-sig")
    parser.add_argument("--id-column", default=None, help="同一列同时映射到 case_id 和 slide_id")
    parser.add_argument("--case-id-column", default=None, help="映射到 case_id 的列名")
    parser.add_argument("--slide-id-column", default=None, help="映射到 slide_id 的列名")
    parser.add_argument("--label-column", required=True, help="映射到 label 的源列名")
    parser.add_argument(
        "--label-map",
        action="append",
        default=[],
        help="标签映射，格式为 源值=目标值，可重复传参，例如 --label-map '1=Adenocarcinoma'",
    )
    parser.add_argument(
        "--no-default-label-map",
        action="store_true",
        help="禁用默认标签映射（默认内置 1=Adenocarcinoma, 0=NonAdenocarcinoma）",
    )
    parser.add_argument(
        "--keep-unmapped-label",
        action="store_true",
        help="当标签值未命中映射时，直接保留原值而不是跳过",
    )
    parser.add_argument(
        "--default-label",
        default=None,
        help="当标签列为空时使用这个默认标签；不传则空标签记录会被跳过",
    )
    parser.add_argument(
        "--skip-log",
        type=Path,
        default=None,
        help="将跳过记录写入单独的 CSV 日志；默认不写",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        case_id_column, slide_id_column = resolve_id_columns(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在：{input_path}")
        return 1

    try:
        rows, fieldnames, used_encoding = load_rows_with_fallback(input_path, args.encoding)
        label_map_items = [] if args.no_default_label_map else list(DEFAULT_LABEL_MAP_ITEMS)
        label_map_items.extend(args.label_map)
        label_map = parse_label_map(label_map_items)
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    required_columns = [case_id_column, slide_id_column, args.label_column]
    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        print(f"[ERROR] 输入 CSV 缺少列：{missing_columns}；当前列为：{fieldnames}")
        return 1

    converted_rows, skipped_rows = convert_rows(
        rows=rows,
        case_id_column=case_id_column,
        slide_id_column=slide_id_column,
        label_column=args.label_column,
        label_map=label_map,
        keep_unmapped_label=args.keep_unmapped_label,
        default_label=args.default_label,
    )
    write_output(converted_rows, output_path, args.output_encoding)

    if args.skip_log:
        write_skip_log(skipped_rows, args.skip_log.resolve(), args.output_encoding)

    label_counter = Counter(row["label"] for row in converted_rows)
    print(f"[INFO] 输入文件：{input_path}")
    print(f"[INFO] 输出文件：{output_path}")
    print(f"[INFO] 使用输入编码：{used_encoding}")
    print(f"[INFO] 输出编码：{args.output_encoding}")
    print(f"[INFO] case_id 列：{case_id_column}")
    print(f"[INFO] slide_id 列：{slide_id_column}")
    print(f"[INFO] label 列：{args.label_column}")
    print(f"[INFO] 标签映射：{label_map if label_map else '未设置'}")
    print(f"[INFO] 总行数：{len(rows)}")
    print(f"[INFO] 成功转换：{len(converted_rows)}")
    print(f"[INFO] 跳过记录：{len(skipped_rows)}")
    for label_name, count in sorted(label_counter.items()):
        print(f"[INFO] {label_name}: {count}")

    if args.skip_log:
        print(f"[INFO] 跳过日志：{args.skip_log.resolve()}")

    if skipped_rows:
        print("[WARN] 以下记录被跳过：")
        for item in skipped_rows[:20]:
            print(f"  - 行 {item['row']}: {item['reason']}")
        if len(skipped_rows) > 20:
            print(f"  - 其余 {len(skipped_rows) - 20} 条未展开")

    print("[INFO] 使用方法：运行 `python3 ViLa-MIL-main/tools/convert_guangdongyi_csv.py --help` 查看完整参数和示例。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
