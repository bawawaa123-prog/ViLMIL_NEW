#!/usr/bin/env python3
"""\
为一批 SVS（含裁剪后的 pseudo-SVS：多目录 tiled BigTIFF 改扩展名）导出低分辨率预览图，便于传输。

功能：
- 扫描输入目录中的 .svs/.SVS 文件
- 使用 OpenSlide 生成缩略图（按最大边长限制缩放）
- 以 .jpg 或 .png 保存到输出目录

依赖：
- openslide-python（并正确安装 openslide 动态库）
- pillow

示例：
  conda run -n vila_mil --no-capture-output \
    python ViLa-MIL-main/tools/export_svs_preview_images.py \
      --input-dir /mnt/nas/shared/medical/MedicalData/Adenocarcinoma/省人医数据20260206/河源/cropped_svs \
      --output-dir /mnt/nas/shared/medical/MedicalData/Adenocarcinoma/省人医数据20260206/河源/cropped_svs/previews \
      --max-edge 1024 --ext .jpg

说明：
- 默认不覆盖已存在的预览图，便于断点续跑；需要重做请加 --overwrite
- 若 OpenSlide 无法打开某些文件，会记录到失败日志（默认：<output-dir>/preview_failed_records.csv）
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


VALID_EXTS = (".svs", ".SVS")


DEFAULT_INPUT_DIR = Path(
    "/mnt/nas/shared/medical/MedicalData/Adenocarcinoma/省人医数据20260206/河源/cropped_svs"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "previews"
DEFAULT_MAX_EDGE = 1024
DEFAULT_EXT = ".jpg"
DEFAULT_OVERWRITE = False


@dataclass(frozen=True)
class PreviewJob:
    src: Path
    dst: Path


def _require_pyvips() -> "object":
    try:
        import pyvips  # type: ignore

        return pyvips
    except Exception as exc:
        raise RuntimeError(
            "缺少依赖 pyvips/libvips（仅在 OpenSlide 失败时作为兜底使用）。\n"
            "（conda-forge 常用安装方式：conda install -c conda-forge pyvips libvips）\n"
            f"原始错误: {exc}"
        )


def _require_openslide_and_pillow() -> tuple[object, object]:
    try:
        import openslide  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "缺少依赖 openslide-python 或系统未安装 openslide 动态库。\n"
            "（conda-forge 常用安装方式：conda install -c conda-forge openslide openslide-python）\n"
            f"原始错误: {exc}"
        )

    try:
        from PIL import Image  # type: ignore

        return openslide, Image
    except Exception as exc:
        raise RuntimeError(
            "缺少依赖 pillow。\n"
            "（conda-forge 常用安装方式：conda install -c conda-forge pillow）\n"
            f"原始错误: {exc}"
        )


def iter_svs_files(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix in VALID_EXTS]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix in VALID_EXTS]
    files.sort(key=lambda p: p.name)
    return files


def build_jobs(svs_files: list[Path], output_dir: Path, ext: str) -> list[PreviewJob]:
    norm_ext = ext.lower().strip()
    if not norm_ext.startswith("."):
        norm_ext = "." + norm_ext
    if norm_ext not in (".jpg", ".jpeg", ".png"):
        raise ValueError(f"不支持的输出格式：{ext}（仅支持 .jpg/.jpeg/.png）")

    jobs: list[PreviewJob] = []
    for src in svs_files:
        dst = output_dir / f"{src.stem}{norm_ext}"
        jobs.append(PreviewJob(src=src, dst=dst))
    return jobs


def export_preview(job: PreviewJob, max_edge: int, overwrite: bool, dry_run: bool) -> None:
    if job.dst.exists() and not overwrite:
        raise FileExistsError(f"已存在且未开启覆盖：{job.dst}")

    job.dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return

    suffix = job.dst.suffix.lower()

    # 优先 OpenSlide（对带金字塔的 WSI 最稳、也最快），失败则用 vips 缩略图兜底。
    openslide_error: Exception | None = None
    try:
        openslide, _ = _require_openslide_and_pillow()
        slide = openslide.OpenSlide(str(job.src))
        try:
            img = slide.get_thumbnail((int(max_edge), int(max_edge)))
        finally:
            slide.close()

        if suffix in (".jpg", ".jpeg"):
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(str(job.dst), format="JPEG", quality=90, optimize=True)
        elif suffix == ".png":
            img.save(str(job.dst), format="PNG", optimize=True)
        else:
            raise ValueError(f"不支持的输出后缀：{job.dst.suffix}")
        return
    except Exception as exc:
        openslide_error = exc

    # 兜底：pyvips 直接做 thumbnail 并保存（不依赖 pillow）。
    try:
        pyvips = _require_pyvips()
        thumb = pyvips.Image.thumbnail(
            str(job.src),
            int(max_edge),
            height=int(max_edge),
            size="down",
        )
        if suffix in (".jpg", ".jpeg"):
            thumb.jpegsave(str(job.dst), Q=90, optimize_coding=True, strip=True)
        elif suffix == ".png":
            thumb.pngsave(str(job.dst), compression=6, strip=True)
        else:
            raise ValueError(f"不支持的输出后缀：{job.dst.suffix}")
    except Exception as exc:
        raise RuntimeError(f"OpenSlide 失败：{openslide_error}；pyvips 兜底也失败：{exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="为 SVS 批量导出低分辨率预览图（jpg/png）")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="输入 SVS 目录（默认指向 cropped_svs）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出预览图目录（会创建）",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=DEFAULT_MAX_EDGE,
        help="预览图最大边长（像素），OpenSlide 会按比例缩放",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=DEFAULT_EXT,
        help="输出格式后缀：.jpg/.png（默认 .jpg）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERWRITE,
        help="输出已存在时覆盖（默认关闭，可用 --overwrite 开启）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划，不实际导出")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="最多处理前 N 个文件（0=不限制，用于测试）",
    )
    parser.add_argument(
        "--fail-log",
        type=Path,
        default=None,
        help="失败记录输出文件（默认：<output-dir>/preview_failed_records.csv）",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    fail_log_path = args.fail_log.resolve() if args.fail_log else (output_dir / "preview_failed_records.csv")

    if args.max_edge <= 0:
        print(f"[ERROR] --max-edge 必须为正整数，当前：{args.max_edge}")
        return 1

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在：{input_dir}")
        return 1

    files = iter_svs_files(input_dir, recursive=args.recursive)
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    try:
        jobs = build_jobs(files, output_dir=output_dir, ext=args.ext)
    except Exception as exc:
        print(f"[ERROR] 参数错误：{exc}")
        return 1

    print(f"[INFO] Input: {input_dir}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Files: {len(files)}")
    print(f"[INFO] Max edge: {args.max_edge}")
    print(f"[INFO] Ext: {args.ext}")
    print(f"[INFO] Overwrite: {args.overwrite}")
    print(f"[INFO] Dry-run: {args.dry_run}")

    iterator = jobs
    if tqdm is not None:
        iterator = tqdm(jobs, total=len(jobs), ncols=110, desc="Export previews")

    total = 0
    ok = 0
    skipped_exists = 0
    failed = 0
    failed_records: list[dict[str, str]] = []

    for job in iterator:
        total += 1
        try:
            export_preview(job, max_edge=args.max_edge, overwrite=args.overwrite, dry_run=args.dry_run)
            ok += 1
            print(f"[OK] {job.src.name} -> {job.dst.name}")
        except FileExistsError:
            skipped_exists += 1
            print(f"[SKIP] 已存在：{job.dst.name}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {job.src.name} | {exc}")
            failed_records.append(
                {
                    "svs": str(job.src),
                    "output": str(job.dst),
                    "type": "runtime_error",
                    "reason": str(exc),
                }
            )

        if tqdm is not None:
            iterator.set_postfix_str(f"ok={ok} skip={skipped_exists} fail={failed}")

    if failed_records:
        fail_log_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_log_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["svs", "output", "type", "reason"])
            writer.writeheader()
            writer.writerows(failed_records)
        print(f"[INFO] 已写入失败记录：{fail_log_path}")

    print("\n========== SUMMARY ==========")
    print(f"Files processed      : {total}")
    print(f"Outputs OK/Planned   : {ok}")
    print(f"Skipped (exists)     : {skipped_exists}")
    print(f"Failed               : {failed}")
    print(f"Output dir           : {output_dir}")
    print("=============================")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
