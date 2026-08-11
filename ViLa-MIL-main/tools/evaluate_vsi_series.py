#!/usr/bin/env python3

import argparse
import csv
import datetime as dt
import os
import re
import subprocess
import sys

import numpy as np


def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc.stdout


def parse_series_ids(showinf_text):
    # Example line: "Series #14"
    ids = sorted({int(x) for x in re.findall(r"Series\s*#(\d+)", showinf_text)})
    return ids


def read_thumb_with_pyvips(image_path, thumb_max_side):
    import pyvips

    thumb = pyvips.Image.thumbnail(image_path, int(thumb_max_side))
    arr = np.frombuffer(thumb.write_to_memory(), dtype=np.uint8)
    arr = arr.reshape(thumb.height, thumb.width, thumb.bands)
    return arr, int(thumb.width), int(thumb.height)


def read_thumb_with_tifffile(image_path, thumb_max_side):
    import tifffile

    with tifffile.TiffFile(image_path) as tf:
        page = tf.pages[0]
        data = page.asarray()

    if data.ndim == 2:
        h, w = data.shape
    else:
        h, w = data.shape[0], data.shape[1]

    scale = max(h, w) / float(max(1, thumb_max_side))
    step = max(1, int(scale))
    thumb = data[::step, ::step]

    if thumb.ndim == 2:
        thumb = np.expand_dims(thumb, axis=-1)

    th, tw = thumb.shape[0], thumb.shape[1]
    return thumb.astype(np.uint8, copy=False), int(tw), int(th)


def compute_tissue_ratio(image_path, thumb_max_side, threshold):
    # Prefer pyvips for speed/memory safety; fallback to tifffile if unavailable.
    try:
        arr, tw, th = read_thumb_with_pyvips(image_path, thumb_max_side)
    except Exception:
        arr, tw, th = read_thumb_with_tifffile(image_path, thumb_max_side)

    if arr.ndim == 2:
        nonblack = arr > threshold
    else:
        nonblack = np.any(arr > threshold, axis=-1)

    total = int(nonblack.size)
    tissue = int(nonblack.sum())
    ratio = float(tissue / total) if total > 0 else 0.0
    return ratio, tissue, total, tw, th


def write_table(rows, csv_path, tsv_path):
    fieldnames = [
        "series_id",
        "status",
        "ome_path",
        "tissue_ratio",
        "tissue_pixels",
        "total_pixels",
        "thumb_w",
        "thumb_h",
        "error",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="扫描VSI全部series，逐个转换并统计组织占比"
    )
    parser.add_argument("--vsi", required=True, help="输入VSI路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--showinf", required=True, help="showinf可执行文件路径")
    parser.add_argument("--bfconvert", required=True, help="bfconvert可执行文件路径")
    parser.add_argument("--thumb-max-side", type=int, default=1536, help="缩略图最长边")
    parser.add_argument("--threshold", type=int, default=8, help="组织判定阈值(0-255)")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有series转换结果")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    vsi_abs = os.path.abspath(args.vsi)
    base = os.path.splitext(os.path.basename(vsi_abs))[0]
    showinf_txt = os.path.join(args.output_dir, "showinf.txt")

    print(f"[1/4] 读取showinf: {vsi_abs}")
    showinf_out = run_cmd([args.showinf, "-nopix", vsi_abs])
    with open(showinf_txt, "w", encoding="utf-8") as f:
        f.write(showinf_out)

    series_ids = parse_series_ids(showinf_out)
    if not series_ids:
        raise RuntimeError("未在showinf输出中解析到任何series")
    print(f"发现 series 数量: {len(series_ids)} -> {series_ids}")

    rows = []
    total_n = len(series_ids)
    print("[2/4] 逐个转换并评估组织占比")
    for idx, sid in enumerate(series_ids, start=1):
        ome_name = f"{base}_series{sid}.ome.tif"
        ome_path = os.path.join(args.output_dir, ome_name)
        row = {
            "series_id": sid,
            "status": "",
            "ome_path": ome_path,
            "tissue_ratio": "",
            "tissue_pixels": "",
            "total_pixels": "",
            "thumb_w": "",
            "thumb_h": "",
            "error": "",
        }

        print(f"  [{idx}/{total_n}] series {sid}")
        try:
            need_convert = args.overwrite or (not os.path.isfile(ome_path))
            if need_convert:
                cmd = [args.bfconvert, "-overwrite", "-series", str(sid), vsi_abs, ome_path]
                run_cmd(cmd)

            ratio, tissue, total, tw, th = compute_tissue_ratio(
                ome_path, args.thumb_max_side, args.threshold
            )
            row["status"] = "ok"
            row["tissue_ratio"] = f"{ratio:.6f}"
            row["tissue_pixels"] = tissue
            row["total_pixels"] = total
            row["thumb_w"] = tw
            row["thumb_h"] = th
        except Exception as e:
            row["status"] = "failed"
            row["error"] = str(e).strip().replace("\n", " | ")

        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: float(r["tissue_ratio"]) if r["status"] == "ok" and r["tissue_ratio"] != "" else -1.0,
        reverse=True,
    )

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"series_tissue_ratio_{ts}.csv")
    tsv_path = os.path.join(args.output_dir, f"series_tissue_ratio_{ts}.tsv")

    print("[3/4] 写出结果表")
    write_table(rows_sorted, csv_path, tsv_path)

    ok_rows = [r for r in rows_sorted if r["status"] == "ok"]
    print("[4/4] 完成")
    print(f"成功: {len(ok_rows)}/{len(rows_sorted)}")
    print(f"CSV: {csv_path}")
    print(f"TSV: {tsv_path}")
    if ok_rows:
        top = ok_rows[0]
        print(
            "最佳series(按组织占比): "
            f"{top['series_id']} ratio={top['tissue_ratio']} path={top['ome_path']}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
