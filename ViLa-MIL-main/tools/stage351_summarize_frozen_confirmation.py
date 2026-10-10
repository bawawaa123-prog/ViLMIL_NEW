#!/usr/bin/env python3
"""Summarize Stage 3.5.0/3.5.1 validation checkpoints without inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("auc", "accuracy", "macro_f1")


def _selected_epoch(epoch_path: Path, fold: int, arm: str) -> dict:
    frame = pd.read_csv(epoch_path)
    required = {"fold", "epoch", "val_acc", "val_auc", "val_f1", "val_loss"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"{arm} fold {fold}: epoch_details missing {missing}")
    if frame.empty:
        raise AssertionError(f"{arm} fold {fold}: epoch_details is empty")
    if frame["fold"].nunique() != 1 or int(frame["fold"].iloc[0]) != fold + 1:
        raise AssertionError(f"{arm} fold {fold}: unexpected one-based fold values {frame['fold'].unique().tolist()}")
    # EarlyStopping receives val_error=1-val_acc and only saves on strict
    # improvement. idxmax therefore reproduces selected checkpoint epoch,
    # choosing the earliest epoch when validation accuracy ties.
    selected = frame.loc[frame["val_acc"].idxmax()]
    return {
        "selected_epoch": int(selected["epoch"]),
        "auc": float(selected["val_auc"]),
        "accuracy": float(selected["val_acc"]),
        "macro_f1": float(selected["val_f1"]),
        "selected_val_error": float(1.0 - selected["val_acc"]),
        "selected_val_loss": float(selected["val_loss"]),
        "epochs_run": int(frame["epoch"].max()),
    }


def _arm_fold(root: Path, arm: str, fold: int) -> tuple[dict, Path]:
    code = f"stage350_{arm}_folds01" if fold in (0, 1) else f"stage351_{arm}_fold{fold}"
    if fold in (0, 1):
        path = root / "00_clean_high_baseline" / arm / f"{code}_s1"
    else:
        path = root / "01_frozen_confirmation" / arm / f"fold{fold}" / f"{code}_s1"
    if not path.is_dir():
        raise FileNotFoundError(f"missing {arm} fold {fold} result directory: {path}")
    summary_path = path / "fold_summary.csv"
    epoch_path = path / "epoch_details.csv"
    if not summary_path.is_file() or not epoch_path.is_file():
        raise FileNotFoundError(f"missing validation result files for {arm} fold {fold}: {summary_path}, {epoch_path}")
    summary = pd.read_csv(summary_path)
    if summary.empty or len(summary) != 1:
        raise AssertionError(f"{arm} fold {fold}: expected one row in fold_summary.csv, got {len(summary)}")
    if "test_auc" in summary and not summary[[c for c in ("test_auc", "test_acc", "test_f1") if c in summary]].isna().all().all():
        raise AssertionError(f"{arm} fold {fold}: test metrics are populated; validation-only summary expected")
    selected = _selected_epoch(epoch_path, fold, arm)
    summary_val_acc = float(summary["val_acc"].iloc[0])
    summary_val_auc = float(summary["val_auc"].iloc[0])
    if not np.isclose(summary_val_acc, selected["accuracy"], rtol=0, atol=1e-12):
        raise AssertionError(f"{arm} fold {fold}: fold_summary val_acc disagrees with selected epoch")
    if not np.isclose(summary_val_auc, selected["auc"], rtol=0, atol=1e-12):
        raise AssertionError(f"{arm} fold {fold}: fold_summary val_auc disagrees with selected epoch")
    return selected, path


def _collect(repo_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows, manifest = [], []
    for fold in range(5):
        routed, routed_path = _arm_fold(repo_root, "routed_control", fold)
        clean, clean_path = _arm_fold(repo_root, "clean_high", fold)
        row = {
            "fold": fold,
            "routed_auc": routed["auc"], "routed_accuracy": routed["accuracy"], "routed_macro_f1": routed["macro_f1"],
            "clean_auc": clean["auc"], "clean_accuracy": clean["accuracy"], "clean_macro_f1": clean["macro_f1"],
            "delta_auc": clean["auc"] - routed["auc"],
            "delta_accuracy": clean["accuracy"] - routed["accuracy"],
            "delta_macro_f1": clean["macro_f1"] - routed["macro_f1"],
            "routed_selected_epoch": routed["selected_epoch"],
            "clean_selected_epoch": clean["selected_epoch"],
            "routed_epochs_run": routed["epochs_run"],
            "clean_epochs_run": clean["epochs_run"],
        }
        rows.append(row)
        manifest.extend([
            {"fold": fold, "arm": "routed_control", "selected_epoch": routed["selected_epoch"], "result_dir": str(routed_path)},
            {"fold": fold, "arm": "clean_high", "selected_epoch": clean["selected_epoch"], "result_dir": str(clean_path)},
        ])
    return pd.DataFrame(rows), manifest


def _summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ("routed", "clean"):
        for metric in METRICS:
            values = metrics[f"{arm}_{metric}"].to_numpy(dtype=float)
            rows.append({
                "arm": "routed_control" if arm == "routed" else "clean_high",
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std()),
            })
    return pd.DataFrame(rows)


def _write_md(output: Path, metrics: pd.DataFrame, summary: pd.DataFrame, manifest: list[dict]):
    lines = [
        "# Stage 3.5.1 frozen Clean High confirmation",
        "",
        "Validation-only summary of Stage 3.5.0 development folds 0/1 and Stage 3.5.1 confirmation folds 2/3/4. No model inference was performed by this tool.",
        "",
        "## Checkpoint selection",
        "",
        "Macro-F1 is taken from the selected checkpoint epoch in `epoch_details.csv`. The repository passes `val_error` to `EarlyStopping`; selection therefore minimizes validation error / maximizes validation accuracy, with the first epoch winning ties. AUC is not used for selection.",
        "",
        "## Fold comparison",
        "",
        "| Fold | Routed AUC | Clean AUC | Delta AUC | Routed Acc | Clean Acc | Delta Acc | Routed Macro-F1 | Clean Macro-F1 | Delta Macro-F1 | Routed Epoch | Clean Epoch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {int(row.fold)} | {row.routed_auc:.4f} | {row.clean_auc:.4f} | {row.delta_auc:+.4f} | "
            f"{row.routed_accuracy:.4f} | {row.clean_accuracy:.4f} | {row.delta_accuracy:+.4f} | "
            f"{row.routed_macro_f1:.4f} | {row.clean_macro_f1:.4f} | {row.delta_macro_f1:+.4f} | "
            f"{int(row.routed_selected_epoch)} | {int(row.clean_selected_epoch)} |"
        )
    lines += ["", "## Five-fold validation summary", "", "| Arm | Metric | Mean | Std |", "|---|---|---:|---:|"]
    for _, row in summary.iterrows():
        lines.append(f"| {row.arm} | {row.metric} | {row.mean:.4f} | {row.std:.4f} |")
    lines += [
        "",
        "Stage 3.5.1 folds 2/3/4 are frozen confirmation folds. This report does not modify architecture, training protocol, checkpoints, or test evaluation.",
        "",
        "Selected checkpoint manifest is stored in `stage351_selected_epochs.csv`.",
    ]
    (output / "stage351_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", type=Path, default=Path("."), help="ViLa-MIL-main repository root")
    parser.add_argument("--output_dir", type=Path, default=Path("analysis/stage3_model_design/05_finalization/01_frozen_confirmation"))
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = repo_root / output
    output.mkdir(parents=True, exist_ok=True)
    metrics, manifest = _collect(repo_root)
    confirmation = metrics[metrics.fold.isin([2, 3, 4])].copy()
    confirmation.to_csv(output / "stage351_confirmation_folds234.csv", index=False)
    metrics.to_csv(output / "stage351_full5fold_metrics.csv", index=False)
    summary = _summary(metrics)
    summary.to_csv(output / "stage351_full5fold_summary.csv", index=False)
    pd.DataFrame(manifest).to_csv(output / "stage351_selected_epochs.csv", index=False)
    _write_md(output, metrics, summary, manifest)
    print(metrics.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
