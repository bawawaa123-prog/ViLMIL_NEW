#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class RocBootstrapResult:
    fpr_grid: np.ndarray
    tpr_mean: np.ndarray
    tpr_lower: np.ndarray
    tpr_upper: np.ndarray
    auc_lower: float
    auc_upper: float
    n_valid_bootstrap: int


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


PLOT_METRIC_SPECS = [
    ("Sensitivity", "sensitivity"),
    ("Specificity", "specificity"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("Accuracy", "accuracy"),
    # Match eval.py's test_f1 (macro-F1 for binary classification).
    ("F1 score (macro)", "f1_score_macro"),
]


CLINICAL_BAR_COLORS = [
    "#6aaedf",  # blue
    "#69bf6f",  # green
    "#ffd25f",  # yellow
    "#5dbdb4",  # teal
    "#3f9998",  # dark teal
    "#f5a5aa",  # pink
]


def compute_clinical_metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred, average="binary", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1_score_pos": float(f1_pos),
        "f1_score_macro": float(f1_macro),
    }


def bootstrap_clinical_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if n_bootstrap <= 0:
        return {}

    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: dict[str, list[float]] = {key: [] for _, key in PLOT_METRIC_SPECS}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        metrics = compute_clinical_metric_dict(y_true[idx], y_pred[idx])
        for _, key in PLOT_METRIC_SPECS:
            value = metrics.get(key, float("nan"))
            if np.isfinite(value):
                samples[key].append(float(value))

    ci: dict[str, tuple[float, float]] = {}
    for _, key in PLOT_METRIC_SPECS:
        vals = np.asarray(samples[key], dtype=float)
        if vals.size:
            ci[key] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        else:
            ci[key] = (float("nan"), float("nan"))
    return ci


def bootstrap_roc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int,
    seed: int,
    fpr_points: int = 201,
) -> RocBootstrapResult:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    fpr_grid = np.linspace(0.0, 1.0, fpr_points)

    tpr_samples = []
    auc_samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = y_score[idx]
        if np.unique(y_b).size < 2:
            continue

        fpr_b, tpr_b, _ = roc_curve(y_b, s_b)
        tpr_interp = np.interp(fpr_grid, fpr_b, tpr_b)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        tpr_samples.append(tpr_interp)
        auc_samples.append(roc_auc_score(y_b, s_b))

    if not tpr_samples:
        nan_arr = np.full_like(fpr_grid, np.nan, dtype=float)
        return RocBootstrapResult(
            fpr_grid=fpr_grid,
            tpr_mean=nan_arr,
            tpr_lower=nan_arr,
            tpr_upper=nan_arr,
            auc_lower=float("nan"),
            auc_upper=float("nan"),
            n_valid_bootstrap=0,
        )

    tpr_arr = np.asarray(tpr_samples)
    auc_arr = np.asarray(auc_samples)
    return RocBootstrapResult(
        fpr_grid=fpr_grid,
        tpr_mean=np.mean(tpr_arr, axis=0),
        tpr_lower=np.percentile(tpr_arr, 2.5, axis=0),
        tpr_upper=np.percentile(tpr_arr, 97.5, axis=0),
        auc_lower=float(np.percentile(auc_arr, 2.5)),
        auc_upper=float(np.percentile(auc_arr, 97.5)),
        n_valid_bootstrap=tpr_arr.shape[0],
    )


def plot_roc_with_ci(
    title: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    auc_val: float,
    roc_boot: RocBootstrapResult,
    output_path: str,
    dpi: int,
    plot_name: str = "Heyuan",
) -> None:
    """Plot ROC in the style of the provided target figure."""
    fpr_emp, tpr_emp, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    if roc_boot.n_valid_bootstrap > 0:
        ax.fill_between(
            roc_boot.fpr_grid,
            roc_boot.tpr_lower,
            roc_boot.tpr_upper,
            color="#64b5f6",
            alpha=0.13,
            zorder=1,
        )
        auc_text = f"{auc_val:.2f} ({roc_boot.auc_lower:.2f} - {roc_boot.auc_upper:.2f})"
    else:
        auc_text = f"{auc_val:.2f} (N/A)"

    ax.step(fpr_emp, tpr_emp, where="post", lw=3.0, color="#1e88e5", label=plot_name, zorder=3)
    ax.plot([0, 1], [0, 1], ls="--", lw=2.0, color="#8a8a8a", zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1-Specificity", fontsize=20)
    ax.set_ylabel("Sensitivity", fontsize=20)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="both", labelsize=17, width=1.4, length=6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_color("black")

    ax.text(0.57, 0.44, plot_name, fontsize=19, transform=ax.transAxes)
    ax.text(0.57, 0.30, f"AUC (95% CI)\n{auc_text}", fontsize=18, transform=ax.transAxes)
    ax.legend(loc="lower right", frameon=False, fontsize=18, handlelength=2.0)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_fold_metric_bars(
    metric_row: pd.Series,
    output_path: str,
    dpi: int,
    title: str | None = None,
    metric_ci: dict[str, tuple[float, float]] | None = None,
) -> None:
    labels = [label for label, _ in PLOT_METRIC_SPECS]
    keys = [key for _, key in PLOT_METRIC_SPECS]
    values = [float(metric_row[key]) for key in keys]

    yerr = None
    if metric_ci:
        lower_err = []
        upper_err = []
        for key, value in zip(keys, values):
            lo, hi = metric_ci.get(key, (float("nan"), float("nan")))
            lower_err.append(0.0 if not np.isfinite(lo) else max(0.0, value - lo))
            upper_err.append(0.0 if not np.isfinite(hi) else max(0.0, hi - value))
        yerr = np.vstack([lower_err, upper_err])

    fig, ax = plt.subplots(figsize=(12.0, 6.3))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        yerr=yerr,
        color=CLINICAL_BAR_COLORS,
        alpha=0.95,
        capsize=6,
        error_kw={"elinewidth": 1.8, "capthick": 1.8, "ecolor": "black"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="y", labelsize=12)
    if title is None:
        title = f"Clinical Metrics (Fold {int(metric_row['fold'])})"
    ax.set_title(title, fontsize=20, pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            min(v + 0.018, 1.025),
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    output_path: str,
    dpi: int,
    normalize: bool = False,
    class_names: tuple[str, str] = ("Pre-IAC", "IAC"),
) -> None:
    """Plot confusion matrix in the style of the provided target figure."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        data = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        fmt = ".2f"
    else:
        data = cm.astype(float)
        fmt = ".0f"

    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=16)
    ax.set_yticklabels(class_names, fontsize=16)
    ax.set_xlabel("Predicted label", fontsize=18)
    ax.set_ylabel("True label", fontsize=18)
    ax.set_title(title, fontsize=24, pad=10)
    ax.tick_params(axis="both", length=0)

    thresh = np.nanmax(data) / 2.0 if np.isfinite(np.nanmax(data)) else 0.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                format(data[i, j], fmt),
                ha="center",
                va="center",
                fontsize=22,
                color="white" if data[i, j] > thresh else "#222222",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def find_fold_csvs(external_dir: str) -> list[tuple[int, str]]:
    pairs = []
    for name in os.listdir(external_dir):
        m = re.fullmatch(r"fold_(\d+)\.csv", name)
        if m:
            pairs.append((int(m.group(1)), os.path.join(external_dir, name)))
    return sorted(pairs, key=lambda x: x[0])


def drop_fold_column(df: pd.DataFrame) -> pd.DataFrame:
    if "fold" in df.columns:
        return df.drop(columns=["fold"])
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-fold external report from original no-threshold outputs.")
    parser.add_argument("--external_dir", required=True, help="Directory with fold_*.csv and timing_details.csv")
    parser.add_argument("--output_dir", default=None, help="Output report directory")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations for ROC 95% CI")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI")
    parser.add_argument("--plot_name", default="Heyuan", help="Name displayed in figure titles/legends")
    parser.add_argument("--negative_label", default="Pre-IAC", help="Class name for label 0 in the confusion matrix")
    parser.add_argument("--positive_label", default="IAC", help="Class name for label 1 in the confusion matrix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    external_dir = os.path.abspath(args.external_dir)
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(external_dir, "no_threshold_external_report")
    )
    os.makedirs(output_dir, exist_ok=True)

    fold_csvs = find_fold_csvs(external_dir)
    if not fold_csvs:
        raise FileNotFoundError(f"No fold_*.csv found in {external_dir}")

    timing_path = os.path.join(external_dir, "timing_details.csv")
    timing_df = pd.read_csv(timing_path) if os.path.isfile(timing_path) else None
    if timing_df is not None:
        timing_df["fold"] = timing_df["fold"].astype(int)

    metrics_rows = []
    auc_rows = []
    timing_rows = []
    fold_details: dict[int, dict[str, object]] = {}

    for fold, fold_csv in fold_csvs:
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        df = pd.read_csv(fold_csv)
        required = {"slide_id", "Y", "p_1"}
        if not required.issubset(df.columns):
            miss = sorted(required - set(df.columns))
            raise ValueError(f"{fold_csv} missing columns: {miss}")

        y_true = df["Y"].astype(int).to_numpy()
        if "Y_hat" in df.columns:
            y_pred = df["Y_hat"].astype(int).to_numpy()
        else:
            y_pred = (df["p_1"].astype(float).to_numpy() >= 0.5).astype(int)
        p_1 = df["p_1"].astype(float).to_numpy()

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cm = np.array([[tn, fp], [fn, tp]], dtype=int)

        sensitivity = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_pos = f1_score(y_true, y_pred, average="binary", zero_division=0)
        auc = roc_auc_score(y_true, p_1)

        metric_row = {
            "fold": int(fold),
            "num_samples": int(len(df)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "f1_score": float(f1_macro),
            "f1_score_macro": float(f1_macro),
            "f1_score_pos": float(f1_pos),
            "auc": float(auc),
        }
        metrics_rows.append(metric_row)

        metrics_df_fold = pd.DataFrame([metric_row])
        metrics_df_fold.to_csv(os.path.join(fold_dir, f"fold_{fold}_metrics_table.csv"), index=False)
        metric_ci = bootstrap_clinical_metric_ci(y_true, y_pred, n_bootstrap=args.bootstrap, seed=args.seed + 10000 + fold)
        pd.DataFrame(
            [
                {
                    "metric": key,
                    "ci_lower_95": vals[0],
                    "ci_upper_95": vals[1],
                }
                for key, vals in metric_ci.items()
            ]
        ).to_csv(os.path.join(fold_dir, f"fold_{fold}_clinical_metrics_ci.csv"), index=False)
        plot_fold_metric_bars(
            metrics_df_fold.iloc[0],
            os.path.join(fold_dir, f"fold_{fold}_metrics_bar.png"),
            args.dpi,
            title=f"Clinical Metrics ({args.plot_name} Fold {fold})",
            metric_ci=metric_ci,
        )

        roc_boot = bootstrap_roc_ci(y_true, p_1, n_bootstrap=args.bootstrap, seed=args.seed + fold)
        plot_roc_with_ci(
            title=f"External ROC - Fold {fold}",
            y_true=y_true,
            y_score=p_1,
            auc_val=float(auc),
            roc_boot=roc_boot,
            output_path=os.path.join(fold_dir, f"fold_{fold}_roc_with_95ci.png"),
            dpi=args.dpi,
            plot_name=f"{args.plot_name} Fold {fold}",
        )

        auc_row = {
            "fold": int(fold),
            "auc": float(auc),
            "auc_ci_lower_95": float(roc_boot.auc_lower),
            "auc_ci_upper_95": float(roc_boot.auc_upper),
            "bootstrap_valid_samples": int(roc_boot.n_valid_bootstrap),
        }
        auc_rows.append(auc_row)
        pd.DataFrame([auc_row]).to_csv(os.path.join(fold_dir, f"fold_{fold}_auc_ci.csv"), index=False)

        pd.DataFrame(
            [
                {"true_label": 0, "pred_0": int(tn), "pred_1": int(fp)},
                {"true_label": 1, "pred_0": int(fn), "pred_1": int(tp)},
            ]
        ).to_csv(os.path.join(fold_dir, f"fold_{fold}_confusion_matrix.csv"), index=False)
        plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix ({args.plot_name} Fold {fold})",
            output_path=os.path.join(fold_dir, f"fold_{fold}_confusion_matrix.png"),
            dpi=args.dpi,
            normalize=False,
            class_names=(args.negative_label, args.positive_label),
        )
        plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix ({args.plot_name} Fold {fold}, Normalized)",
            output_path=os.path.join(fold_dir, f"fold_{fold}_confusion_matrix_normalized.png"),
            dpi=args.dpi,
            normalize=True,
            class_names=(args.negative_label, args.positive_label),
        )

        inf_cols = ["slide_id", "Y"]
        if "Y_hat" in df.columns:
            inf_cols.append("Y_hat")
        if "p_0" in df.columns:
            inf_cols.append("p_0")
        inf_cols.append("p_1")
        inf_df = df[inf_cols].copy()
        inf_df.insert(0, "fold", int(fold))
        inf_df["y_pred_used"] = y_pred
        inf_df["is_correct"] = (y_pred == y_true).astype(int)
        inf_df.to_csv(os.path.join(fold_dir, f"fold_{fold}_inference_results.csv"), index=False)

        if timing_df is not None:
            t = timing_df[timing_df["fold"] == int(fold)].copy()
            if len(t) == 1:
                t["slides_per_second"] = t["num_wsi"] / t["eval_seconds"]
                t["milliseconds_per_slide"] = t["avg_wsi_seconds"] * 1000.0
                t["accuracy"] = float(accuracy)
                t["f1_score"] = float(f1_macro)
                t["auc"] = float(auc)
                t.to_csv(os.path.join(fold_dir, f"fold_{fold}_inference_timing_table.csv"), index=False)
                timing_rows.append(t.iloc[0].to_dict())

        fold_details[int(fold)] = {
            "df": df,
            "y_true": y_true,
            "p_1": p_1,
            "cm": cm,
            "metric_row": metric_row,
            "auc_row": auc_row,
            "roc_boot": roc_boot,
            "metric_ci": metric_ci,
            "timing_row": timing_rows[-1] if timing_rows and int(timing_rows[-1].get("fold", -1)) == int(fold) else None,
        }

    all_metrics_df = pd.DataFrame(metrics_rows).sort_values("fold")
    all_auc_df = pd.DataFrame(auc_rows).sort_values("fold")
    all_metrics_df.to_csv(os.path.join(output_dir, "all_folds_metrics_table.csv"), index=False)
    all_auc_df.to_csv(os.path.join(output_dir, "all_folds_auc_ci_table.csv"), index=False)

    if timing_rows:
        pd.DataFrame(timing_rows).sort_values("fold").to_csv(
            os.path.join(output_dir, "all_folds_inference_timing_table.csv"),
            index=False,
        )

    best_acc_idx = int(all_metrics_df["accuracy"].idxmax())
    best_acc_fold = int(all_metrics_df.loc[best_acc_idx, "fold"])

    best_dir = os.path.join(output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    best_detail = fold_details[best_acc_fold]

    best_metrics_df = pd.DataFrame([best_detail["metric_row"]])
    best_metrics_df = drop_fold_column(best_metrics_df)
    best_metrics_df.to_csv(os.path.join(best_dir, "best_metrics_table.csv"), index=False)
    plot_fold_metric_bars(
        pd.Series(best_detail["metric_row"]),
        os.path.join(best_dir, "best_metrics_bar.png"),
        args.dpi,
        title=f"Clinical Metrics ({args.plot_name})",
        metric_ci=best_detail.get("metric_ci"),
    )

    best_auc_df = pd.DataFrame([best_detail["auc_row"]])
    best_auc_df = drop_fold_column(best_auc_df)
    best_auc_df.to_csv(os.path.join(best_dir, "best_auc_ci.csv"), index=False)

    cm = np.asarray(best_detail["cm"])
    pd.DataFrame(
        [
            {"true_label": 0, "pred_0": int(cm[0, 0]), "pred_1": int(cm[0, 1])},
            {"true_label": 1, "pred_0": int(cm[1, 0]), "pred_1": int(cm[1, 1])},
        ]
    ).to_csv(os.path.join(best_dir, "best_confusion_matrix.csv"), index=False)
    plot_confusion_matrix(
        cm,
        title=f"Confusion Matrix ({args.plot_name})",
        output_path=os.path.join(best_dir, "best_confusion_matrix.png"),
        dpi=args.dpi,
        normalize=False,
        class_names=(args.negative_label, args.positive_label),
    )
    plot_confusion_matrix(
        cm,
        title=f"Confusion Matrix ({args.plot_name}, Normalized)",
        output_path=os.path.join(best_dir, "best_confusion_matrix_normalized.png"),
        dpi=args.dpi,
        normalize=True,
        class_names=(args.negative_label, args.positive_label),
    )

    best_df = pd.DataFrame(best_detail["df"]).copy()
    inf_cols = ["slide_id", "Y"]
    if "Y_hat" in best_df.columns:
        inf_cols.append("Y_hat")
    if "p_0" in best_df.columns:
        inf_cols.append("p_0")
    inf_cols.append("p_1")
    best_inf_df = best_df[inf_cols].copy()
    if "Y_hat" in best_df.columns:
        y_pred_best = best_df["Y_hat"].astype(int).to_numpy()
    else:
        y_pred_best = (best_df["p_1"].astype(float).to_numpy() >= 0.5).astype(int)
    y_true_best = best_df["Y"].astype(int).to_numpy()
    best_inf_df["y_pred_used"] = y_pred_best
    best_inf_df["is_correct"] = (y_pred_best == y_true_best).astype(int)
    best_inf_df.to_csv(os.path.join(best_dir, "best_inference_results.csv"), index=False)

    best_misclassified = best_inf_df[best_inf_df["is_correct"] == 0].copy()
    if not best_misclassified.empty:
        if "y_pred_used" in best_misclassified.columns:
            best_misclassified = best_misclassified.rename(columns={"y_pred_used": "pred_label"})
        if "Y" in best_misclassified.columns:
            best_misclassified = best_misclassified.rename(columns={"Y": "true_label"})
        if "p_0" not in best_misclassified.columns and "p_1" in best_misclassified.columns:
            best_misclassified["p_0"] = 1.0 - best_misclassified["p_1"].astype(float)
        if "true_label" in best_misclassified.columns and "pred_label" in best_misclassified.columns:
            best_misclassified["true_type"] = best_misclassified["true_label"].map({0: args.negative_label, 1: args.positive_label})
            best_misclassified["pred_type"] = best_misclassified["pred_label"].map({0: args.negative_label, 1: args.positive_label})
        best_misclassified = best_misclassified[
            [c for c in ["slide_id", "true_label", "pred_label", "true_type", "pred_type", "p_0", "p_1"] if c in best_misclassified.columns]
        ]
    else:
        best_misclassified = pd.DataFrame(columns=["slide_id", "true_label", "pred_label", "true_type", "pred_type", "p_0", "p_1"])
    best_misclassified.to_csv(os.path.join(best_dir, "best_misclassified_slides.csv"), index=False)

    timing_row = best_detail["timing_row"]
    if timing_row is not None:
        best_timing_df = pd.DataFrame([timing_row])
        best_timing_df = drop_fold_column(best_timing_df)
        best_timing_df.to_csv(os.path.join(best_dir, "best_inference_timing_table.csv"), index=False)

    plot_roc_with_ci(
        title="Best (ACC) ROC",
        y_true=np.asarray(best_detail["y_true"]),
        y_score=np.asarray(best_detail["p_1"]),
        auc_val=float(best_detail["metric_row"]["auc"]),
        roc_boot=best_detail["roc_boot"],
        output_path=os.path.join(best_dir, "best_roc_with_95ci.png"),
        dpi=args.dpi,
        plot_name=args.plot_name,
    )

    best_info = pd.DataFrame(
        [
            {
                "type": "best_acc",
                "f1_score": float(all_metrics_df.loc[best_acc_idx, "f1_score"]),
                "accuracy": float(all_metrics_df.loc[best_acc_idx, "accuracy"]),
                "auc": float(all_metrics_df.loc[best_acc_idx, "auc"]),
            },
        ]
    )
    best_info.to_csv(os.path.join(best_dir, "best_selection.csv"), index=False)

    config = {
        "external_dir": external_dir,
        "output_dir": output_dir,
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "best_metric": "accuracy",
        "best_acc_fold": int(best_acc_fold),
        "plot_name": args.plot_name,
        "negative_label": args.negative_label,
        "positive_label": args.positive_label,
    }
    with open(os.path.join(output_dir, "report_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("=== Completed ===")
    print(f"Saved report directory: {output_dir}")
    print(f"Best by ACC: fold {best_acc_fold}")


if __name__ == "__main__":
    main()
