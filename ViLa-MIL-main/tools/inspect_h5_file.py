#!/usr/bin/env python3
"""Inspect a single h5 file and print its shape and contents.

Example:
  python tools/inspect_h5_file.py \
    --h5-file data/jiangxi/reseg_v2_full/patches_coords_5x_eq4096/patches_4096/20251222-21.h5
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import h5py
import numpy as np


def _to_python(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_value(value: Any) -> str:
    value = _to_python(value)
    if isinstance(value, str):
        return value
    return repr(value)


def _print_attrs(obj: h5py.Dataset | h5py.Group, indent: str) -> None:
    if len(obj.attrs) == 0:
        print(f"{indent}attrs: <none>")
        return

    print(f"{indent}attrs:")
    for key in obj.attrs.keys():
        print(f"{indent}  - {key}: {_format_value(obj.attrs[key])}")


def _print_dataset(name: str, dset: h5py.Dataset, indent: str, preview_rows: int, full: bool) -> None:
    print(f"{indent}dataset: {name}")
    print(f"{indent}  shape: {dset.shape}")
    print(f"{indent}  dtype: {dset.dtype}")
    _print_attrs(dset, indent + "  ")

    try:
        data = dset[()]
    except Exception as exc:
        print(f"{indent}  content: <failed to read: {exc}>")
        return

    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            print(f"{indent}  content: {_format_value(data.item())}")
            return

        if full or preview_rows <= 0 or data.shape[0] <= preview_rows:
            print(f"{indent}  content:")
            print(data)
        else:
            print(f"{indent}  content preview (first {preview_rows} rows):")
            print(data[:preview_rows])
        return

    print(f"{indent}  content: {_format_value(data)}")


def _walk_group(name: str, obj: h5py.Group | h5py.Dataset, indent: str, preview_rows: int, full: bool) -> None:
    if isinstance(obj, h5py.Dataset):
        _print_dataset(name, obj, indent, preview_rows, full)
        return

    if name:
        print(f"{indent}group: {name}")
    _print_attrs(obj, indent + ("  " if name else ""))

    for child_name, child in obj.items():
        child_path = f"{name}/{child_name}" if name else child_name
        _walk_group(child_path, child, indent + "  ", preview_rows, full)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print h5 file structure, shapes, and contents")
    parser.add_argument("--h5-file", required=True, help="Path to the target .h5 file")
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of leading rows to print for array-like datasets when --full is not set",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full contents of each dataset instead of a preview",
    )
    args = parser.parse_args()

    h5_path = os.path.abspath(args.h5_file)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"h5 file not found: {h5_path}")

    print(f"h5_file: {h5_path}")
    print("=" * 120)

    with h5py.File(h5_path, "r") as h5_file:
        _print_attrs(h5_file, "")
        print("-" * 120)

        if len(h5_file.keys()) == 0:
            print("<empty h5 file>")
            return

        for key, value in h5_file.items():
            _walk_group(key, value, "", args.preview_rows, args.full)
            print("-" * 120)


if __name__ == "__main__":
    main()