#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
严格交叉验证划分脚本（与本项目 main.py / dataset_generic.py 兼容）

为什么需要这个脚本：
- 现有 create_splits_seq.py 采用“每折独立随机抽样”策略，
  不保证每个样本恰好一次出现在 test 集，因此不是严格 K 折交叉验证。
- 本脚本生成“严格 K 折”划分：
  1) test 折两两互斥；
  2) 所有样本在 K 个 fold 中恰好一次进入 test；
  3) 支持按 case_id 分组，避免同一患者泄漏到不同集合。

========================
输入 CSV 格式（必须包含）
========================
列名要求（默认）：
- case_id   : 患者/病例ID（用于分组）
- slide_id  : 切片ID（用于训练时索引 h5）
- label     : 类别标签（字符串或整数均可）

示例：
case_id,slide_id,label
P001,2460239-B2,Adenocarcinoma
P001,2460242-B2,Adenocarcinoma
P002,2460399-B2,NonAdenocarcinoma

========================
输出文件格式
========================
输出目录会生成：
- splits_0.csv ... splits_{k-1}.csv
  每个文件包含三列：train,val,test
  每列是 slide_id 列表，不等长部分用空值（NaN）填充。
  该格式可被 datasets/dataset_generic.py::return_splits(from_id=False, csv_path=...) 直接读取。

- splits_0_bool.csv ... splits_{k-1}_bool.csv
  布尔one-hot风格划分（可选调试）

- splits_0_descriptor.csv ...
  每个类别在 train/val/test 的样本数统计

- strict_fold_assignments.csv
  每个 slide 对应的 fold_id（其作为 test 出现的fold）

严格三划分策略（默认）：
- test fold = i
- val  fold = (i + 1) % k
- train = 其余 folds
因此：test 与 val 在每个 fold 内严格不重叠，且 test 全局严格 K 折。
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _majority_vote(values: pd.Series):
    vc = values.value_counts()
    return vc.index[0]


def _build_group_table(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    strict_single_label_per_group: bool,
) -> pd.DataFrame:
    """构建 group -> label 的映射表。

    若同一 group 内存在多个标签：
    - strict_single_label_per_group=True: 直接报错
    - 否则: 使用多数投票标签，并给出告警
    """
    rows = []
    conflicted = 0

    for gid, gdf in df.groupby(group_col):
        labels = gdf[label_col].dropna().unique().tolist()
        if len(labels) == 0:
            raise ValueError(f"group={gid} 没有有效标签")
        if len(labels) > 1:
            conflicted += 1
            if strict_single_label_per_group:
                raise ValueError(
                    f"group={gid} 出现多个标签 {labels}，请先清洗数据，或关闭 --strict_single_label_per_group"
                )
            label = _majority_vote(gdf[label_col])
        else:
            label = labels[0]

        rows.append({group_col: gid, label_col: label})

    if conflicted > 0:
        print(
            f"[Warn] 检测到 {conflicted} 个 {group_col} 存在多标签，已按多数投票处理。"
        )

    gtable = pd.DataFrame(rows)
    gtable = gtable.reset_index(drop=True)
    return gtable


def _pad_columns_to_df(data: Dict[str, List[str]]) -> pd.DataFrame:
    max_len = max(len(v) for v in data.values()) if data else 0
    out = {}
    for k, v in data.items():
        if len(v) < max_len:
            out[k] = v + [np.nan] * (max_len - len(v))
        else:
            out[k] = v
    return pd.DataFrame(out)


def _save_bool_split(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    out_path: str,
):
    all_ids = train_ids + val_ids + test_ids
    labels = (
        ["train"] * len(train_ids)
        + ["val"] * len(val_ids)
        + ["test"] * len(test_ids)
    )
    df = pd.DataFrame(index=all_ids, columns=["train", "val", "test"], data=False)
    for sid, part in zip(all_ids, labels):
        df.loc[sid, part] = True
    df.to_csv(out_path)


def _save_descriptor(
    df_all: pd.DataFrame,
    slide_col: str,
    label_col: str,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    out_path: str,
):
    classes = sorted(df_all[label_col].dropna().unique().tolist())
    descriptor = pd.DataFrame(0, index=classes, columns=["train", "val", "test"])

    id_to_label = dict(zip(df_all[slide_col].tolist(), df_all[label_col].tolist()))

    for sid in train_ids:
        descriptor.loc[id_to_label[sid], "train"] += 1
    for sid in val_ids:
        descriptor.loc[id_to_label[sid], "val"] += 1
    for sid in test_ids:
        descriptor.loc[id_to_label[sid], "test"] += 1

    descriptor.to_csv(out_path)


def _check_no_overlap(a: List[str], b: List[str], c: List[str]):
    sa, sb, sc = set(a), set(b), set(c)
    assert sa.isdisjoint(sb), "train 与 val 有重叠"
    assert sa.isdisjoint(sc), "train 与 test 有重叠"
    assert sb.isdisjoint(sc), "val 与 test 有重叠"


def main():
    parser = argparse.ArgumentParser(description="Create strict K-fold CSV splits for ViLa-MIL")
    parser.add_argument("--csv_path", type=str, required=True, help="输入数据CSV路径")
    parser.add_argument("--save_dir", type=str, required=True, help="输出目录（保存splits_i.csv）")
    parser.add_argument("--k", type=int, default=5, help="K折数量（建议>=3）")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")

    parser.add_argument("--group_col", type=str, default="case_id", help="分组列名，默认case_id")
    parser.add_argument("--slide_col", type=str, default="slide_id", help="slide列名，默认slide_id")
    parser.add_argument("--label_col", type=str, default="label", help="标签列名，默认label")

    parser.add_argument(
        "--strict_single_label_per_group",
        action="store_true",
        default=False,
        help="若同一group存在多标签时是否直接报错（默认False，采用多数投票）",
    )

    args = parser.parse_args()

    if args.k < 3:
        raise ValueError("严格 train/val/test 三划分建议 k>=3。")

    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)

    required = [args.group_col, args.slide_col, args.label_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"输入CSV缺少必要列: {col}")

    # 去重（防止重复slide行）
    df = df.drop_duplicates(subset=[args.slide_col]).reset_index(drop=True)

    # group-level label表
    gtable = _build_group_table(
        df=df,
        group_col=args.group_col,
        label_col=args.label_col,
        strict_single_label_per_group=args.strict_single_label_per_group,
    )

    # 分层K折在 group 层完成，保证 case 不泄漏
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)

    groups = gtable[args.group_col].to_numpy()
    labels = gtable[args.label_col].to_numpy()

    group_fold: Dict[str, int] = {}
    for fold_id, (_, test_idx) in enumerate(skf.split(groups, labels)):
        for idx in test_idx:
            group_fold[groups[idx]] = fold_id

    # slide 映射到其 group 的fold
    df["fold_id"] = df[args.group_col].map(group_fold)

    # 保存每个slide属于哪个fold（作为test出现的fold）
    assignment_path = os.path.join(args.save_dir, "strict_fold_assignments.csv")
    df[[args.slide_col, args.group_col, args.label_col, "fold_id"]].to_csv(assignment_path, index=False)

    # 输出每折 train/val/test CSV
    all_test_union = set()
    all_slides = set(df[args.slide_col].tolist())

    for i in range(args.k):
        test_fold = i
        val_fold = (i + 1) % args.k

        test_ids = df.loc[df["fold_id"] == test_fold, args.slide_col].tolist()
        val_ids = df.loc[df["fold_id"] == val_fold, args.slide_col].tolist()
        train_ids = df.loc[(df["fold_id"] != test_fold) & (df["fold_id"] != val_fold), args.slide_col].tolist()

        _check_no_overlap(train_ids, val_ids, test_ids)

        # 兼容项目读取格式：三列不等长列表
        split_df = _pad_columns_to_df({"train": train_ids, "val": val_ids, "test": test_ids})
        split_path = os.path.join(args.save_dir, f"splits_{i}.csv")
        split_df.to_csv(split_path, index=False)

        # bool风格和descriptor，便于检查
        bool_path = os.path.join(args.save_dir, f"splits_{i}_bool.csv")
        _save_bool_split(train_ids, val_ids, test_ids, bool_path)

        desc_path = os.path.join(args.save_dir, f"splits_{i}_descriptor.csv")
        _save_descriptor(df, args.slide_col, args.label_col, train_ids, val_ids, test_ids, desc_path)

        all_test_union.update(test_ids)

    # 严格K折检查：每个样本恰好一次出现在test全集
    if all_test_union != all_slides:
        missing = len(all_slides - all_test_union)
        extra = len(all_test_union - all_slides)
        raise RuntimeError(
            f"严格K折检查失败: missing_in_test_union={missing}, extra_in_test_union={extra}"
        )

    print("=" * 80)
    print("Strict K-fold splits generated successfully.")
    print(f"Input CSV: {args.csv_path}")
    print(f"Output Dir: {args.save_dir}")
    print(f"K: {args.k}, Seed: {args.seed}")
    print("Rule: fold i -> test=i, val=(i+1)%k, train=others")
    print("Files: splits_i.csv, splits_i_bool.csv, splits_i_descriptor.csv, strict_fold_assignments.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
