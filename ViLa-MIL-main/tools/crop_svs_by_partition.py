#!/usr/bin/env python3
"""\
按 CSV 的“分区”字段对 SVS 做水平等分裁剪，并按【原文件名-L/M/R】命名输出。

输入：
- CSV: ViLa-MIL-main/dataset_csv/广州大学基线表-河源分割任务.csv
- SVS 目录：/home/ljh/data/河源

规则：
- 分区=0001：不裁剪，直接复制到输出目录，文件名不变
- 分区=0002：二等分裁剪（左/右），输出：<stem>-L.svs, <stem>-R.svs
- 分区=0003：三等分裁剪（左/中/右），输出：<stem>-L.svs, <stem>-M.svs, <stem>-R.svs

实现说明（重要）：
- OpenSlide 本身无法写出真正的 Aperio SVS。
- 本脚本使用 libvips/pyvips 写出“金字塔 tiled BigTIFF”，然后将扩展名改为 .svs。
  大多数 WSI 工具会把这类金字塔 TIFF 当作 SVS 来读取，但它不一定包含 Aperio 私有标签。

依赖：
- pyvips + libvips（建议 conda-forge 安装）
- openslide / openslide-python（用于兼容性检测；libvips 的 openslide loader 也会用到）

示例：
conda run -n vila_mil --no-capture-output python ViLa-MIL-main/tools/crop_svs_by_partition.py

默认配置（已写死在代码里，可选覆盖）：
- CSV: ViLa-MIL-main/dataset_csv/广州大学基线表-河源分割任务.csv
- 输入目录: /home/ljh/data/河源
- 输出目录: /home/ljh/data/河源/cropped_svs
- 覆盖: 关闭（默认断点续跑；如需强制重做可用 --overwrite）
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import re

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VALID_EXTS = (".svs", ".SVS")


DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[1] / "dataset_csv" / "广州大学基线表-河源分割任务.csv"
)
DEFAULT_INPUT_DIR = Path("/home/ljh/data/河源")
DEFAULT_OUTPUT_DIR = Path("/mnt/nas/shared/medical/MedicalData/Adenocarcinoma/省人医数据20260206/河源/cropped_svs")
DEFAULT_ENCODING = "gb18030"
DEFAULT_OVERWRITE = False


def load_csv_rows_with_fallback(csv_path: Path, preferred_encoding: str) -> tuple[list[dict], list[str], str]:
    encodings_to_try: list[str] = []
    for enc in [preferred_encoding, "utf-8-sig", "utf-8", "gb18030", "gbk"]:
        if enc and enc not in encodings_to_try:
            encodings_to_try.append(enc)

    last_error: Exception | None = None
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

    matches = [f for f in source_dir.glob(f"{stem}.*") if f.suffix.lower() == ".svs"]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise FileNotFoundError(f"匹配到多个同名 SVS，无法唯一确定：{names}")

    raise FileNotFoundError(f"未找到对应 SVS：{stem}.svs")


@dataclass(frozen=True)
class CropJob:
    src: Path
    dst: Path
    x: int
    y: int
    w: int
    h: int
    xres_ppmm: float | None
    yres_ppmm: float | None
    jpeg_q: int | None
    aperio_description: str | None
    level_dimensions: list[tuple[int, int]] | None


def _require_pyvips() -> "object":
    try:
        import pyvips  # type: ignore

        return pyvips
    except Exception as exc:
        raise RuntimeError(
            "缺少依赖 pyvips/libvips，无法写出金字塔 SVS(TIFF)。\n"
            "建议在 conda 环境中安装：\n"
            "  conda install -n vila_mil -c conda-forge pyvips libvips openslide openslide-python\n"
            f"原始错误: {exc}"
        )


def build_crop_jobs(src: Path, dst_dir: Path, partition_value: str) -> list[CropJob]:
    # 用 OpenSlide 获取 level0 尺寸与 mpp（更贴近原始 SVS 元信息）
    try:
        import openslide  # type: ignore

        s = openslide.OpenSlide(str(src))
        full_w, full_h = map(int, s.dimensions)

        mpp_x = s.properties.get("openslide.mpp-x")
        mpp_y = s.properties.get("openslide.mpp-y")
        xres_ppmm = (1000.0 / float(mpp_x)) if mpp_x else None
        yres_ppmm = (1000.0 / float(mpp_y)) if mpp_y else None

        desc = s.properties.get("tiff.ImageDescription", "")
        m = re.search(r"\bQ=(\d{1,3})\b", desc)
        jpeg_q = int(m.group(1)) if m else None
        aperio_description = desc if desc.startswith("Aperio") else None
        level_dimensions = [(int(w), int(h)) for (w, h) in s.level_dimensions]
        s.close()
    except Exception:
        # 兜底：如果 OpenSlide 读取失败，再用 vips 读取尺寸
        pyvips = _require_pyvips()
        img = pyvips.Image.new_from_file(str(src), access="sequential")
        full_w, full_h = int(img.width), int(img.height)
        xres_ppmm = None
        yres_ppmm = None
        jpeg_q = None
        aperio_description = None
        level_dimensions = None

    part = str(partition_value).strip()

    if part == "0001":
        # 不裁剪：这里返回空，让上层走 copy 分支
        return []

    if part == "0002":
        cut1 = full_w // 2
        widths = [cut1, full_w - cut1]
        xs = [0, cut1]
        suffixes = ["L", "R"]
    elif part == "0003":
        cut1 = full_w // 3
        cut2 = full_w // 3
        cut3 = full_w - cut1 - cut2
        widths = [cut1, cut2, cut3]
        xs = [0, cut1, cut1 + cut2]
        suffixes = ["L", "M", "R"]
    else:
        raise ValueError(f"未知分区值：{part}（期望 0001/0002/0003）")

    jobs: list[CropJob] = []
    for x, w, suf in zip(xs, widths, suffixes, strict=True):
        dst = dst_dir / f"{src.stem}-{suf}.svs"
        jobs.append(
            CropJob(
                src=src,
                dst=dst,
                x=int(x),
                y=0,
                w=int(w),
                h=int(full_h),
                xres_ppmm=xres_ppmm,
                yres_ppmm=yres_ppmm,
                jpeg_q=jpeg_q,
                aperio_description=aperio_description,
                level_dimensions=level_dimensions,
            )
        )
    return jobs


def _which_or_raise(name: str) -> str:
    import shutil as _shutil

    p = _shutil.which(name)
    if not p:
        raise RuntimeError(f"缺少外部工具：{name}（请确认已安装 libtiff-tools 或 conda 中存在该命令）")
    return p


def _build_aperio_description(job: CropJob) -> str:
    # 尽量复用原始描述中的扫描信息，但更新尺寸与 Q。
    q = job.jpeg_q if job.jpeg_q is not None else 91

    if job.aperio_description:
        parts = job.aperio_description.split("|")
        header = parts[0]
        first_line = header.splitlines()[0] if header else "Aperio"
        prefix = (
            f"{first_line} \n"
            f"{job.w}x{job.h} [0,0,{job.w}x{job.h}] (256x256) JPEG/YCC Q={q}"
        )
        if len(parts) > 1:
            return prefix + "|" + "|".join(parts[1:])
        return prefix

    # 兜底：构造一个最简 Aperio 风格描述
    return f"Aperio\n{job.w}x{job.h} [0,0,{job.w}x{job.h}] (256x256) JPEG/YCC Q={q}"


def write_aperio_like_svs(job: CropJob, overwrite: bool, dry_run: bool) -> None:
    """尽量实现“单纯裁剪，其他不变”：
    - 直接裁剪原 SVS 的各个原生层级（不重建金字塔）
    - 用 tiffcp 写成多目录 tiled TIFF，并用 JPEG(YCbCr) 压缩（类似 Aperio JPEG/YCC）
    - 用 tiffset 写入 Aperio 风格 ImageDescription，方便 OpenSlide 识别 vendor=aperio

    说明：这仍然不是厂商原生 SVS 的“无损 tile-copy”，边界非 256 对齐时必然涉及重编码。
    """

    if job.dst.exists() and not overwrite:
        raise FileExistsError(f"已存在且未开启覆盖：{job.dst}")

    job.dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return

    pyvips = _require_pyvips()
    tiffcp = _which_or_raise("tiffcp")
    tiffset = _which_or_raise("tiffset")

    # 读取 level 维度：优先用 OpenSlide 结果；否则至少写单层
    level_dims = job.level_dimensions
    if not level_dims:
        img0 = pyvips.Image.new_from_file(str(job.src), level=0, access="sequential")
        level_dims = [(int(img0.width), int(img0.height))]

    w0, h0 = level_dims[0]
    if w0 <= 0 or h0 <= 0:
        raise RuntimeError(f"无法获取有效尺寸：{job.src}")

    q = job.jpeg_q if job.jpeg_q is not None else 91

    with tempfile.TemporaryDirectory(prefix="crop_svs_") as tmpd:
        tmp_dir = Path(tmpd)

        # 逐层裁剪并保存为临时 TIFF（tiffcp 会重压缩为 YCbCr JPEG）
        level_files: list[Path] = []
        for lvl, (wl, hl) in enumerate(level_dims):
            if wl <= 0 or hl <= 0:
                continue

            sx = wl / w0
            sy = hl / h0
            xl = int(round(job.x * sx))
            yl = int(round(job.y * sy))
            wll = int(round(job.w * sx))
            hll = int(round(job.h * sy))

            # clamp
            xl = max(0, min(xl, wl - 1))
            yl = max(0, min(yl, hl - 1))
            wll = max(1, min(wll, wl - xl))
            hll = max(1, min(hll, hl - yl))

            img = pyvips.Image.new_from_file(str(job.src), level=int(lvl), access="sequential")
            crop = img.crop(xl, yl, wll, hll)

            # 分辨率：像素/mm；低层级按缩放比例降低像素密度
            save_kwargs: dict = {
                "tile": True,
                "pyramid": False,
                "compression": "jpeg",
                "Q": int(q),
                "tile_width": 256,
                "tile_height": 256,
                "properties": False,
                "bigtiff": True,
            }
            if job.xres_ppmm is not None and job.yres_ppmm is not None:
                save_kwargs["xres"] = float(job.xres_ppmm * sx)
                save_kwargs["yres"] = float(job.yres_ppmm * sy)
                save_kwargs["resunit"] = "cm"

            lvl_tif = tmp_dir / f"lvl{lvl}.tif"
            crop.tiffsave(str(lvl_tif), **save_kwargs)
            level_files.append(lvl_tif)

        if not level_files:
            raise RuntimeError("未生成任何层级临时文件")

        tmp_out_tif = tmp_dir / "out.tif"
        if tmp_out_tif.exists():
            tmp_out_tif.unlink()

        # 用 tiffcp 写最终多目录 TIFF：
        # -m 0: 取消内存限制（否则大图直接 MemoryLimitError）
        # -M : 禁用 mmap，避免某些系统上资源问题
        # -t/-w/-l: tiled 256
        # -c jpeg:Q: 默认输出 YCbCr（不加 r），更接近 Aperio JPEG/YCC
        cmd = [
            tiffcp,
            "-m",
            "0",
            "-M",
            "-8",
            "-t",
            "-w",
            "256",
            "-l",
            "256",
            "-c",
            f"jpeg:{q}",
            *[str(p) for p in level_files],
            str(tmp_out_tif),
        ]
        subprocess.run(cmd, check=True)

        # 写入 Aperio 风格 ImageDescription，帮助 OpenSlide 识别为 aperio
        desc = _build_aperio_description(job)
        subprocess.run([tiffset, "-s", "270", desc, str(tmp_out_tif)], check=True)

        # 写到最终 .svs（/tmp 可能与输出目录跨文件系统，需用 move 而非 rename）
        if job.dst.exists() and overwrite:
            job.dst.unlink()
        shutil.move(str(tmp_out_tif), str(job.dst))


def write_pyramidal_tiff_as_svs(job: CropJob, overwrite: bool, dry_run: bool) -> None:
    # 旧实现：vips pyramid(2x) 会生成过多层级且体积显著增大；保留函数名以兼容调用点，
    # 实际走更接近 Aperio 的多层级裁剪写法。
    write_aperio_like_svs(job, overwrite=overwrite, dry_run=dry_run)


def copy_as_is(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> None:
    if dst.exists() and not overwrite:
        raise FileExistsError(f"已存在且未开启覆盖：{dst}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return

    if dst.exists() and overwrite:
        dst.unlink()
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="按 CSV 分区裁剪 SVS，并按 L/M/R 命名输出")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="CSV 文件路径（需包含 filename 与 分区 列）",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="输入 SVS 目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录（会创建）",
    )
    parser.add_argument("--encoding", type=str, default=DEFAULT_ENCODING, help="CSV 编码")
    parser.add_argument("--filename-col", type=str, default="filename", help="SVS 文件名列")
    parser.add_argument("--partition-col", type=str, default="分区", help="分区列")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERWRITE,
        help="输出已存在时覆盖（默认开启，可用 --no-overwrite 关闭）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际裁剪/复制")
    parser.add_argument("--max-rows", type=int, default=0, help="最多处理前 N 行（0=不限制，用于测试）")
    parser.add_argument(
        "--fail-log",
        type=Path,
        default=None,
        help="失败记录输出文件（默认：<output-dir>/crop_failed_records.csv）",
    )
    args = parser.parse_args()

    csv_path = args.csv_path.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    fail_log_path = args.fail_log.resolve() if args.fail_log else (output_dir / "crop_failed_records.csv")

    if not csv_path.exists():
        print(f"[ERROR] CSV 不存在：{csv_path}")
        return 1
    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在：{input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows, fieldnames, used_encoding = load_csv_rows_with_fallback(csv_path, args.encoding)
    except UnicodeDecodeError as exc:
        print(f"[ERROR] CSV 解码失败：{exc}")
        return 1

    if not fieldnames or args.filename_col not in fieldnames or args.partition_col not in fieldnames:
        print(
            "[ERROR] CSV 缺少必要列。"
            f"需要：{args.filename_col!r}, {args.partition_col!r}；当前列：{fieldnames}"
        )
        return 1

    print(f"[INFO] CSV: {csv_path}")
    print(f"[INFO] Input: {input_dir}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] CSV Encoding used: {used_encoding}")
    print(f"[INFO] Dry-run: {args.dry_run}")

    work_rows = rows
    if args.max_rows and args.max_rows > 0:
        work_rows = rows[: args.max_rows]

    iterator = work_rows
    if tqdm is not None:
        iterator = tqdm(work_rows, total=len(work_rows), ncols=110, desc="Crop SVS")

    total = 0
    ok = 0
    skipped_exists = 0
    missing = 0
    failed = 0
    failed_records: list[dict[str, str]] = []

    for row in iterator:
        total += 1
        filename_value = row.get(args.filename_col, "")
        partition_value = row.get(args.partition_col, "")

        try:
            src = resolve_source_file(input_dir, str(filename_value))
            part = str(partition_value).strip()

            if part == "0001":
                dst = output_dir / src.name
                try:
                    copy_as_is(src, dst, overwrite=args.overwrite, dry_run=args.dry_run)
                    print(f"[OK] COPY {src.name} -> {dst.name}")
                    ok += 1
                except FileExistsError:
                    skipped_exists += 1
                    print(f"[SKIP] 已存在：{dst.name}")
                continue

            jobs = build_crop_jobs(src, output_dir, part)
            for job in jobs:
                try:
                    write_pyramidal_tiff_as_svs(job, overwrite=args.overwrite, dry_run=args.dry_run)
                    print(f"[OK] CROP {src.name} -> {job.dst.name} (x={job.x} w={job.w})")
                    ok += 1
                except FileExistsError:
                    skipped_exists += 1
                    print(f"[SKIP] 已存在：{job.dst.name}")

        except FileNotFoundError as exc:
            missing += 1
            print(f"[MISS] filename={filename_value} | {exc}")
            failed_records.append(
                {
                    "filename": str(filename_value),
                    "partition": str(partition_value),
                    "type": "missing",
                    "reason": str(exc),
                }
            )
        except Exception as exc:
            failed += 1
            print(f"[FAIL] filename={filename_value} partition={partition_value} | {exc}")
            failed_records.append(
                {
                    "filename": str(filename_value),
                    "partition": str(partition_value),
                    "type": "runtime_error",
                    "reason": str(exc),
                }
            )

        if tqdm is not None:
            iterator.set_postfix_str(f"ok={ok} skip={skipped_exists} miss={missing} fail={failed}")

    if failed_records:
        fail_log_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_log_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "partition", "type", "reason"])
            writer.writeheader()
            writer.writerows(failed_records)
        print(f"[INFO] 已写入失败记录：{fail_log_path}")

    print("\n========== SUMMARY ==========")
    print(f"Rows processed       : {total}")
    print(f"Outputs OK/Planned   : {ok}")
    print(f"Skipped (exists)     : {skipped_exists}")
    print(f"Missing              : {missing}")
    print(f"Failed               : {failed}")
    print(f"Output dir           : {output_dir}")
    print("=============================")

    # 缺失文件不一定算“运行失败”，但为了可控，若有 runtime_error 才返回 2
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
