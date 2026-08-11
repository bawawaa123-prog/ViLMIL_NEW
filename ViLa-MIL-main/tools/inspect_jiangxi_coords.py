#!/usr/bin/env python3
"""Inspect generated Jiangxi coords h5 files and print coordinate shapes/samples."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import h5py
import pandas as pd


def resolve_patches_dir(root: str, size: int) -> str:
    """Accept either coords root or patches_<size> directory."""
    root = os.path.abspath(root)
    base = os.path.basename(root)
    if base == f"patches_{size}":
        return root
    candidate = os.path.join(root, f"patches_{size}")
    if os.path.isdir(candidate):
        return candidate
    return root


def collect_h5_files(patches_dir: str) -> Dict[str, str]:
    if not os.path.isdir(patches_dir):
        return {}
    out: Dict[str, str] = {}
    for name in os.listdir(patches_dir):
        if not name.lower().endswith(".h5"):
            continue
        slide_id = os.path.splitext(name)[0]
        out[slide_id] = os.path.join(patches_dir, name)
    return out


def inspect_one_h5(h5_path: str, head: int) -> dict:
    try:
        with h5py.File(h5_path, "r") as f:
            if "coords" in f:
                dset = f["coords"]
            elif len(f.keys()) > 0:
                first_key = list(f.keys())[0]
                dset = f[first_key]
            else:
                return {"ok": False, "error": "empty_h5"}

            shape = tuple(dset.shape)
            dtype = str(dset.dtype)
            n = int(dset.shape[0]) if len(dset.shape) >= 1 else 0
            show_n = max(0, min(head, n))
            sample = dset[:show_n].tolist() if show_n > 0 else []

            patch_size = dset.attrs.get("patch_size", None)
            patch_level = dset.attrs.get("patch_level", None)
            name_attr = dset.attrs.get("name", None)

            return {
                "ok": True,
                "shape": shape,
                "dtype": dtype,
                "n_coords": n,
                "patch_size": patch_size,
                "patch_level": patch_level,
                "name": name_attr,
                "sample": sample,
            }
    except Exception as e:  # file may still be written
        return {"ok": False, "error": str(e)}


def fmt_info(title: str, info: Optional[dict]) -> str:
    if info is None:
        return f"{title}: missing"
    if not info.get("ok", False):
        return f"{title}: error={info.get('error')}"
    return (
        f"{title}: shape={info['shape']} n={info['n_coords']} "
        f"patch_size={info.get('patch_size')} patch_level={info.get('patch_level')} "
        f"sample={info.get('sample', [])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect coords_512 and coords_2048 outputs (shape + sample coords)"
    )
    parser.add_argument(
        "--coords-512-root",
        default="data/jiangxi_2048_512/coords_512",
        help="coords_512 root or patches_512 path",
    )
    parser.add_argument(
        "--coords-2048-root",
        default="data/jiangxi_2048_512/coords_2048",
        help="coords_2048 root or patches_2048 path",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=3,
        help="number of sample coordinates to print from each h5",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only inspect first N slide_ids",
    )
    parser.add_argument(
        "--slide-id",
        type=str,
        default=None,
        help="optional single slide_id (or .h5 filename) to inspect",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="optional path to save summary csv",
    )
    args = parser.parse_args()

    patches_512 = resolve_patches_dir(args.coords_512_root, 512)
    patches_2048 = resolve_patches_dir(args.coords_2048_root, 2048)

    files_512 = collect_h5_files(patches_512)
    files_2048 = collect_h5_files(patches_2048)

    all_slide_ids = sorted(set(files_512.keys()) | set(files_2048.keys()))
    if args.slide_id is not None:
        target = os.path.splitext(os.path.basename(args.slide_id.strip()))[0]
        all_slide_ids = [target]
    if args.limit is not None:
        all_slide_ids = all_slide_ids[: args.limit]

    print(f"patches_512 : {patches_512} ({len(files_512)} files)")
    print(f"patches_2048: {patches_2048} ({len(files_2048)} files)")
    print(f"inspect_slide_count: {len(all_slide_ids)}")
    print("-" * 120)

    rows = []
    for slide_id in all_slide_ids:
        info_512 = inspect_one_h5(files_512[slide_id], args.head) if slide_id in files_512 else None
        info_2048 = inspect_one_h5(files_2048[slide_id], args.head) if slide_id in files_2048 else None

        print(f"[{slide_id}]")
        print("  " + fmt_info("512 ", info_512))
        print("  " + fmt_info("2048", info_2048))

        rows.append(
            {
                "slide_id": slide_id,
                "has_512": slide_id in files_512,
                "has_2048": slide_id in files_2048,
                "ok_512": (info_512 or {}).get("ok", False) if info_512 is not None else False,
                "ok_2048": (info_2048 or {}).get("ok", False) if info_2048 is not None else False,
                "n_coords_512": (info_512 or {}).get("n_coords", None),
                "n_coords_2048": (info_2048 or {}).get("n_coords", None),
                "shape_512": str((info_512 or {}).get("shape", None)),
                "shape_2048": str((info_2048 or {}).get("shape", None)),
                "patch_size_512": (info_512 or {}).get("patch_size", None),
                "patch_size_2048": (info_2048 or {}).get("patch_size", None),
                "patch_level_512": (info_512 or {}).get("patch_level", None),
                "patch_level_2048": (info_2048 or {}).get("patch_level", None),
                "error_512": (info_512 or {}).get("error", None),
                "error_2048": (info_2048 or {}).get("error", None),
            }
        )
        print("-" * 120)

    df = pd.DataFrame(rows)
    if not df.empty:
        both_ok = ((df["ok_512"] == True) & (df["ok_2048"] == True)).sum()  # noqa: E712
        only_512 = ((df["has_512"] == True) & (df["has_2048"] == False)).sum()  # noqa: E712
        only_2048 = ((df["has_512"] == False) & (df["has_2048"] == True)).sum()  # noqa: E712
        any_error = ((df["ok_512"] == False) | (df["ok_2048"] == False)).sum()  # noqa: E712

        print("SUMMARY")
        print(f"  both_ok    : {int(both_ok)}")
        print(f"  only_512   : {int(only_512)}")
        print(f"  only_2048  : {int(only_2048)}")
        print(f"  any_error  : {int(any_error)}")

    if args.save_csv:
        out_csv = os.path.abspath(args.save_csv)
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"saved_csv: {out_csv}")


if __name__ == "__main__":
    main()
