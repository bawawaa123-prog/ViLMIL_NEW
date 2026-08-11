#!/usr/bin/env python3
"""\
按传入的横向起始坐标，对单个 SVS 做自定义水平裁剪，并写出为 .svs。

设计目标：
- 复用 crop_svs_by_partition.py 中已经验证过的裁剪与写出方式
- 不再依赖 CSV，每次只处理一个 SVS
- 不再按等分裁剪，而是根据传入的裁剪块数和各块起始横坐标进行裁剪

裁剪规则：
- 使用 --crop-count 指定输出块数
- 使用 --x-starts 指定每一块在 level0 上的起始横坐标，数量必须与块数一致
- 第 i 块宽度 = 第 i+1 块起始 x - 当前块起始 x
- 最后一块宽度 = 全图宽度 - 最后一块起始 x
- 为避免遗漏区域，默认要求第一块起始 x 必须为 0

输出命名：
- 默认 2 块时：<stem>-L.svs, <stem>-R.svs
- 默认 3 块时：<stem>-L.svs, <stem>-M.svs, <stem>-R.svs
- 其他块数时：<stem>-P1.svs, <stem>-P2.svs, ...
- 可通过 --suffixes 自定义后缀

依赖：
- pyvips + libvips
- openslide / openslide-python
- tiffcp / tiffset

示例：
conda run -n vila_mil --no-capture-output python ViLa-MIL-main/tools/crop_single_svs_by_xcoords.py \
  --input-dir /home/ljh/data/河源 \
  --filename B2400949-Y1 \
  --output-dir /home/ljh/data/河源/custom_crops \
  --crop-count 3 \
  --x-starts 0,28500,62100

如果原图宽度为 90000，则会生成：
- 第 1 块: [0, 28500)
- 第 2 块: [28500, 62100)
- 第 3 块: [62100, 90000)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
import re


VALID_EXTS = (".svs", ".SVS")
DEFAULT_INPUT_DIR = Path("/home/ljh/data/河源")
DEFAULT_OUTPUT_DIR = Path("/mnt/nas/shared/medical/MedicalData/Adenocarcinoma/省人医数据20260206/河源/custom_crops")
DEFAULT_OVERWRITE = False


def resolve_source_file(source_dir: Path, filename_value: str) -> Path:
    raw = str(filename_value).strip()
    if not raw:
        raise FileNotFoundError("filename 为空")

    p = Path(raw)

    if p.is_absolute():
        if p.exists() and p.suffix.lower() == ".svs":
            return p
        raise FileNotFoundError(f"指定的绝对路径不存在或不是 SVS：{p}")

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


def _which_or_raise(name: str) -> str:
    import shutil as _shutil

    p = _shutil.which(name)
    if not p:
        raise RuntimeError(f"缺少外部工具：{name}（请确认已安装 libtiff-tools 或 conda 中存在该命令）")
    return p


def parse_int_list(raw_value: str, arg_name: str) -> list[int]:
    parts = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not parts:
        raise ValueError(f"{arg_name} 不能为空")

    values: list[int] = []
    for item in parts:
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{arg_name} 中包含非整数值：{item}") from exc
    return values


def parse_suffixes(raw_value: str | None, crop_count: int) -> list[str]:
    if raw_value:
        suffixes = [item.strip() for item in raw_value.split(",") if item.strip()]
        if len(suffixes) != crop_count:
            raise ValueError(f"--suffixes 数量必须等于 --crop-count，当前为 {len(suffixes)} != {crop_count}")
        return suffixes

    if crop_count == 1:
        return ["P1"]
    if crop_count == 2:
        return ["L", "R"]
    if crop_count == 3:
        return ["L", "M", "R"]
    return [f"P{i}" for i in range(1, crop_count + 1)]


def read_slide_metadata(src: Path) -> tuple[int, int, float | None, float | None, int | None, str | None, list[tuple[int, int]] | None]:
    try:
        import openslide  # type: ignore

        slide = openslide.OpenSlide(str(src))
        full_w, full_h = map(int, slide.dimensions)

        mpp_x = slide.properties.get("openslide.mpp-x")
        mpp_y = slide.properties.get("openslide.mpp-y")
        xres_ppmm = (1000.0 / float(mpp_x)) if mpp_x else None
        yres_ppmm = (1000.0 / float(mpp_y)) if mpp_y else None

        desc = slide.properties.get("tiff.ImageDescription", "")
        match = re.search(r"\bQ=(\d{1,3})\b", desc)
        jpeg_q = int(match.group(1)) if match else None
        aperio_description = desc if desc.startswith("Aperio") else None
        level_dimensions = [(int(w), int(h)) for (w, h) in slide.level_dimensions]
        slide.close()
        return full_w, full_h, xres_ppmm, yres_ppmm, jpeg_q, aperio_description, level_dimensions
    except Exception:
        pyvips = _require_pyvips()
        img = pyvips.Image.new_from_file(str(src), access="sequential")
        return int(img.width), int(img.height), None, None, None, None, None


def validate_x_starts(x_starts: list[int], crop_count: int, full_w: int, require_zero_start: bool) -> None:
    if crop_count <= 0:
        raise ValueError("--crop-count 必须大于 0")
    if len(x_starts) != crop_count:
        raise ValueError(f"--x-starts 数量必须等于 --crop-count，当前为 {len(x_starts)} != {crop_count}")
    if require_zero_start and x_starts[0] != 0:
        raise ValueError("默认要求第一块起始横坐标为 0；如确实需要跳过左侧区域，请使用 --allow-nonzero-first-x")

    prev = None
    for idx, x_value in enumerate(x_starts, start=1):
        if x_value < 0 or x_value >= full_w:
            raise ValueError(f"第 {idx} 个起始横坐标越界：{x_value}，合法范围为 [0, {full_w - 1}]")
        if prev is not None and x_value <= prev:
            raise ValueError(f"--x-starts 必须严格递增，发现 {prev} 后面是 {x_value}")
        prev = x_value


def build_crop_jobs_from_x_starts(
    src: Path,
    dst_dir: Path,
    x_starts: list[int],
    suffixes: list[str],
) -> list[CropJob]:
    full_w, full_h, xres_ppmm, yres_ppmm, jpeg_q, aperio_description, level_dimensions = read_slide_metadata(src)

    jobs: list[CropJob] = []
    for index, (start_x, suffix) in enumerate(zip(x_starts, suffixes, strict=True)):
        next_x = x_starts[index + 1] if index + 1 < len(x_starts) else full_w
        width = next_x - start_x
        if width <= 0:
            raise ValueError(f"第 {index + 1} 块宽度无效：start={start_x}, next={next_x}")

        dst = dst_dir / f"{src.stem}-{suffix}.svs"
        jobs.append(
            CropJob(
                src=src,
                dst=dst,
                x=int(start_x),
                y=0,
                w=int(width),
                h=int(full_h),
                xres_ppmm=xres_ppmm,
                yres_ppmm=yres_ppmm,
                jpeg_q=jpeg_q,
                aperio_description=aperio_description,
                level_dimensions=level_dimensions,
            )
        )
    return jobs


def _build_aperio_description(job: CropJob) -> str:
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

    return f"Aperio\n{job.w}x{job.h} [0,0,{job.w}x{job.h}] (256x256) JPEG/YCC Q={q}"


def write_aperio_like_svs(job: CropJob, overwrite: bool, dry_run: bool) -> None:
    if job.dst.exists() and not overwrite:
        raise FileExistsError(f"已存在且未开启覆盖：{job.dst}")

    job.dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return

    pyvips = _require_pyvips()
    tiffcp = _which_or_raise("tiffcp")
    tiffset = _which_or_raise("tiffset")

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

            xl = max(0, min(xl, wl - 1))
            yl = max(0, min(yl, hl - 1))
            wll = max(1, min(wll, wl - xl))
            hll = max(1, min(hll, hl - yl))

            img = pyvips.Image.new_from_file(str(job.src), level=int(lvl), access="sequential")
            crop = img.crop(xl, yl, wll, hll)

            save_kwargs: dict[str, object] = {
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

        desc = _build_aperio_description(job)
        subprocess.run([tiffset, "-s", "270", desc, str(tmp_out_tif)], check=True)

        if job.dst.exists() and overwrite:
            job.dst.unlink()
        shutil.move(str(tmp_out_tif), str(job.dst))


def main() -> int:
    parser = argparse.ArgumentParser(description="按自定义横坐标裁剪单个 SVS")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="输入 SVS 所在目录；若 --filename 传绝对路径，则忽略该参数",
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="待裁剪的 SVS 文件名，可不带后缀；也支持直接传入绝对路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    parser.add_argument(
        "--crop-count",
        type=int,
        required=True,
        help="裁剪块数",
    )
    parser.add_argument(
        "--x-starts",
        type=str,
        required=True,
        help="每一块在 level0 上的起始横坐标，逗号分隔，例如 0,30000,65000",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        default=None,
        help="输出文件后缀，逗号分隔；例如 L,M,R 或 A,B,C",
    )
    parser.add_argument(
        "--allow-nonzero-first-x",
        action="store_true",
        help="允许第一块起始横坐标不为 0；启用后，最左侧未覆盖区域会被跳过",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERWRITE,
        help="输出已存在时覆盖（默认关闭，可用 --overwrite 开启）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际写文件")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists() and not Path(args.filename).is_absolute():
        print(f"[ERROR] 输入目录不存在：{input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        src = resolve_source_file(input_dir, args.filename)
        x_starts = parse_int_list(args.x_starts, "--x-starts")
        suffixes = parse_suffixes(args.suffixes, args.crop_count)

        full_w, full_h, _, _, _, _, _ = read_slide_metadata(src)
        validate_x_starts(
            x_starts=x_starts,
            crop_count=args.crop_count,
            full_w=full_w,
            require_zero_start=not args.allow_nonzero_first_x,
        )

        jobs = build_crop_jobs_from_x_starts(
            src=src,
            dst_dir=output_dir,
            x_starts=x_starts,
            suffixes=suffixes,
        )
    except Exception as exc:
        print(f"[ERROR] 参数或输入校验失败：{exc}")
        return 1

    print(f"[INFO] Source: {src}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Slide size(level0): {full_w}x{full_h}")
    print(f"[INFO] Crop count: {args.crop_count}")
    print(f"[INFO] X starts: {x_starts}")
    print(f"[INFO] Suffixes: {suffixes}")
    print(f"[INFO] Dry-run: {args.dry_run}")

    planned = 0
    skipped_exists = 0
    failed = 0

    for index, job in enumerate(jobs, start=1):
        try:
            write_aperio_like_svs(job, overwrite=args.overwrite, dry_run=args.dry_run)
            planned += 1
            print(
                f"[OK] PART {index}: {job.dst.name} | x={job.x}, w={job.w}, y={job.y}, h={job.h}"
            )
        except FileExistsError:
            skipped_exists += 1
            print(f"[SKIP] 已存在：{job.dst.name}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] PART {index}: {job.dst.name} | {exc}")

    print("\n========== SUMMARY ==========")
    print(f"Outputs OK/Planned   : {planned}")
    print(f"Skipped (exists)     : {skipped_exists}")
    print(f"Failed               : {failed}")
    print(f"Output dir           : {output_dir}")
    print("=============================")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())