#!/usr/bin/env python3
"""Collect completed Step 2.6 fold summaries without touching training outputs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def read_group(root: Path, name: str) -> pd.DataFrame:
    path = root / name / f"adenocarcinoma_biomedclip_{name}_strict5_s1" / "fold_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} fold summary: {path}")
    df = pd.read_csv(path)
    required = {"fold", "test_auc", "test_acc", "test_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if len(df) != 5 or sorted(df["fold"].astype(int).tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError(f"{path} must contain exactly folds 1..5")
    df.insert(0, "scale", name)
    return df[["scale", "fold", "test_auc", "test_acc", "test_f1"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("analysis/stage2_yiyuan_data_audit/06_scale_ablation"))
    args = ap.parse_args()
    groups = pd.concat([read_group(args.root, n) for n in ("low_only", "high_only", "dual_scale")], ignore_index=True)
    rows = []
    for metric in ("test_auc", "test_acc", "test_f1"):
        wide = groups.pivot(index="fold", columns="scale", values=metric)
        for fold, row in wide.iterrows():
            rows.append({"metric": metric, "fold": int(fold), **{k: float(row[k]) for k in wide.columns},
                         "dual_minus_low": float(row["dual_scale"] - row["low_only"]),
                         "dual_minus_high": float(row["dual_scale"] - row["high_only"]),
                         "high_minus_low": float(row["high_only"] - row["low_only"])})
        rows.append({"metric": metric, "fold": "mean", **{k: float(groups.loc[groups.scale == k, metric].mean()) for k in ("low_only", "high_only", "dual_scale")},
                     "dual_minus_low": float(groups.loc[groups.scale == "dual_scale", metric].mean() - groups.loc[groups.scale == "low_only", metric].mean()),
                     "dual_minus_high": float(groups.loc[groups.scale == "dual_scale", metric].mean() - groups.loc[groups.scale == "high_only", metric].mean()),
                     "high_minus_low": float(groups.loc[groups.scale == "high_only", metric].mean() - groups.loc[groups.scale == "low_only", metric].mean())})
        rows.append({"metric": metric, "fold": "std", **{k: float(groups.loc[groups.scale == k, metric].std(ddof=0)) for k in ("low_only", "high_only", "dual_scale")},
                     "dual_minus_low": np.nan, "dual_minus_high": np.nan, "high_minus_low": np.nan})
    out = args.root / "comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    summary = args.root / "summary.md"
    lines = ["# Step 2.6: Low-only / High-only / Dual-scale 消融", "", "三组使用相同 strict5 case-level split、seed=1、80 epochs、Adam、lr=1e-4、prompt 与现有 BiomedCLIP features。`dual_scale` 的结果来自独立输出目录；不覆盖 Stage 1 baseline。", "", "## Fold results", ""]
    for metric in ("test_auc", "test_acc", "test_f1"):
        lines.append(f"### {metric}")
        lines.append("")
        lines.append(pd.DataFrame(rows).query("metric == @metric and fold != 'std'").to_markdown(index=False))
        lines.append("")
    lines += ["## Interpretation", "", "请结合 comparison.csv 的 mean/std 与逐 fold 差值判断 dual 是否稳定优于 single-scale；若 dual 在某些 fold 下降，应报告该事实。", "", "## Decision", "", "完整结果生成后再判定是否值得进入 spatially aligned cross-scale reasoning。"]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
