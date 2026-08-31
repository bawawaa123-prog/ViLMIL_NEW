#!/usr/bin/env python3
"""Validation-only complementarity diagnosis for independently trained Low/High models."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import stage335_routing_diagnostics as stage335

DEVICE = stage335.DEVICE


def _checkpoint_for_fold(template: str, fold: int) -> str:
    pattern = template.format(fold=fold)
    matches = sorted(glob.glob(os.path.expanduser(pattern)))
    if len(matches) != 1:
        raise RuntimeError(f"expected one checkpoint for fold {fold}, got {matches} from {pattern}")
    return matches[0]


def _metrics(labels: np.ndarray, logits: np.ndarray, n_classes: int) -> dict:
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    try:
        if len(np.unique(labels)) <= 1:
            auc = float("nan")
        elif n_classes == 2:
            auc = float(roc_auc_score(labels, probs[:, 1]))
        else:
            auc = float(roc_auc_score(labels, probs, multi_class="ovr"))
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "predictions": preds,
    }


def _collect(args, dataset, n_classes: int, fold: int, prompts, checkpoint: str, scale_mode: str) -> pd.DataFrame:
    from utils.utils import get_simple_loader

    split_path = Path(args.splits_dir) / f"splits_{fold}.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"missing validation split file: {split_path}")
    split_dataset = dataset.return_splits(from_id=False, csv_path=str(split_path))[1]
    loader = get_simple_loader(split_dataset, batch_size=1, num_workers=0, mode="transformer")
    args.checkpoint = checkpoint
    args.scale_mode = scale_mode
    model = stage335._model(args, n_classes, prompts, "normal")
    rows = []
    with torch.no_grad():
        for batch in loader:
            data_s, coord_s, data_l, coord_l, label = batch[:5]
            slide_id = str(batch[5][0]) if len(batch) >= 6 else str(len(rows))
            mapping = batch[6][0] if len(batch) >= 7 else None
            data_s, coord_s = data_s.to(DEVICE), coord_s.to(DEVICE)
            data_l, coord_l = data_l.to(DEVICE), coord_l.to(DEVICE)
            label = label.to(DEVICE)
            _, _, _, branches = model(
                data_s, coord_s, data_l, coord_l, label,
                mapping=mapping, return_branch_logits=True,
            )
            # The selected branch is taken from its independently trained model.
            key = "logits_low" if scale_mode == "low" else "logits_high"
            logits = branches[key][0].detach().cpu().numpy().astype(float)
            rows.append({
                "slide_id": slide_id,
                "label": int(label.item()),
                **{f"logit_{c}": float(value) for c, value in enumerate(logits)},
                "prediction": int(np.argmax(logits)),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise AssertionError(f"fold {fold} validation produced no samples")
    if frame["slide_id"].duplicated().any():
        duplicate_ids = frame.loc[frame["slide_id"].duplicated(), "slide_id"].tolist()
        raise AssertionError(f"duplicate slide IDs in {scale_mode} fold {fold}: {duplicate_ids[:5]}")
    return frame


def _logit_stats(frame: pd.DataFrame, n_classes: int, branch: str, fold: int) -> dict:
    logits = frame[[f"{branch}_logit_{c}" for c in range(n_classes)]].to_numpy(dtype=float)
    if n_classes == 2:
        margins = np.abs(logits[:, 1] - logits[:, 0])
    else:
        ordered = np.sort(logits, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
    norms = np.linalg.norm(logits, axis=1)
    return {
        "fold": fold,
        "branch": branch,
        "n_slides": len(frame),
        "logit_norm_mean": float(norms.mean()),
        "logit_norm_std": float(norms.std()),
        "logit_norm_min": float(norms.min()),
        "logit_norm_max": float(norms.max()),
        "prediction_margin_mean": float(margins.mean()),
        "prediction_margin_std": float(margins.std()),
        "prediction_margin_min": float(margins.min()),
        "prediction_margin_max": float(margins.max()),
    }


def _run_fold(args, dataset, n_classes: int, fold: int, prompts, output: Path):
    low_ckpt = _checkpoint_for_fold(args.low_checkpoint_template, fold)
    high_ckpt = _checkpoint_for_fold(args.high_checkpoint_template, fold)
    low = _collect(args, dataset, n_classes, fold, prompts, low_ckpt, "low").rename(
        columns={"label": "label_low", "prediction": "prediction_low", **{f"logit_{c}": f"low_logit_{c}" for c in range(n_classes)}}
    )
    high = _collect(args, dataset, n_classes, fold, prompts, high_ckpt, "high").rename(
        columns={"label": "label_high", "prediction": "prediction_high", **{f"logit_{c}": f"high_logit_{c}" for c in range(n_classes)}}
    )
    if set(low.slide_id) != set(high.slide_id):
        raise AssertionError(f"fold {fold} Low/High validation slide IDs differ")
    merged = low.merge(high, on="slide_id", how="inner", validate="one_to_one")
    if len(merged) != len(low) or len(merged) != len(high):
        raise AssertionError(f"fold {fold} Low/High ID alignment dropped samples")
    if not np.array_equal(merged.label_low.to_numpy(), merged.label_high.to_numpy()):
        raise AssertionError(f"fold {fold} Low/High labels disagree after ID alignment")
    merged = merged.sort_values("slide_id").reset_index(drop=True)
    merged["label"] = merged["label_low"].astype(int)
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / f"fold_{fold}_branch_predictions.csv", index=False)

    labels = merged.label.to_numpy(dtype=int)
    low_logits = merged[[f"low_logit_{c}" for c in range(n_classes)]].to_numpy(dtype=float)
    high_logits = merged[[f"high_logit_{c}" for c in range(n_classes)]].to_numpy(dtype=float)
    low_m, high_m = _metrics(labels, low_logits, n_classes), _metrics(labels, high_logits, n_classes)
    low_pred, high_pred = low_m["predictions"], high_m["predictions"]
    low_correct, high_correct = low_pred == labels, high_pred == labels
    high_wrong_low_correct = ~high_correct & low_correct
    low_wrong_high_correct = ~low_correct & high_correct
    both_correct, both_wrong = low_correct & high_correct, ~low_correct & ~high_correct
    comp = {
        "fold": fold,
        "n_slides": len(labels),
        "high_wrong_low_correct": int(high_wrong_low_correct.sum()),
        "low_wrong_high_correct": int(low_wrong_high_correct.sum()),
        "both_correct": int(both_correct.sum()),
        "both_wrong": int(both_wrong.sum()),
        "disagreement_count": int((low_pred != high_pred).sum()),
        "disagreement_rate": float(np.mean(low_pred != high_pred)),
        "conditional_high_error_rescue_count": int(high_wrong_low_correct.sum()),
        "conditional_high_error_rescue_rate": float(high_wrong_low_correct.sum() / max(1, (~high_correct).sum())),
        "oracle_correct": int((low_correct | high_correct).sum()),
        "oracle_accuracy": float(np.mean(low_correct | high_correct)),
        # Names retained for continuity with Stage 3.4.0 summaries.
        "high_wrong_count": int((~high_correct).sum()),
        "high_wrong_low_correct_conditional_rate": float(high_wrong_low_correct.sum() / max(1, (~high_correct).sum())),
    }
    pd.DataFrame([comp]).to_csv(output / f"fold_{fold}_complementarity_summary.csv", index=False)

    alpha_rows = []
    for alpha in np.linspace(0.0, 1.0, 11):
        metrics = _metrics(labels, alpha * high_logits + (1.0 - alpha) * low_logits, n_classes)
        alpha_rows.append({"fold": fold, "alpha": float(alpha), "auc": metrics["auc"], "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "diagnostic_only": True})
    alpha_frame = pd.DataFrame(alpha_rows)
    alpha_frame["is_best_auc"] = alpha_frame.auc == alpha_frame.auc.max()
    alpha_frame["is_best_accuracy"] = alpha_frame.accuracy == alpha_frame.accuracy.max()
    alpha_frame["is_best_macro_f1"] = alpha_frame.macro_f1 == alpha_frame.macro_f1.max()
    alpha_frame.to_csv(output / f"fold_{fold}_alpha_sweep.csv", index=False)

    metric_rows = [
        {"fold": fold, "mode": "low_independent", "auc": low_m["auc"], "accuracy": low_m["accuracy"], "macro_f1": low_m["macro_f1"]},
        {"fold": fold, "mode": "high_original", "auc": high_m["auc"], "accuracy": high_m["accuracy"], "macro_f1": high_m["macro_f1"]},
    ]
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output / f"fold_{fold}_branch_metrics.csv", index=False)
    stats_frame = pd.DataFrame([
        _logit_stats(merged, n_classes, "low", fold),
        _logit_stats(merged, n_classes, "high", fold),
    ])
    stats_frame.to_csv(output / f"fold_{fold}_logit_statistics.csv", index=False)
    return comp, alpha_frame, metrics_frame, stats_frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low_checkpoint_template", required=True, help="fold-formatted Low-only checkpoint path")
    parser.add_argument("--high_checkpoint_template", required=True, help="fold-formatted original High-trained checkpoint path")
    parser.add_argument("--data_root_dir", required=True)
    parser.add_argument("--data_folder_s", required=True)
    parser.add_argument("--data_folder_l", required=True)
    parser.add_argument("--mapping_path", required=True)
    parser.add_argument("--splits_dir", required=True)
    parser.add_argument("--task", required=True, choices=["task_tcga_rcc_subtyping", "task_tcga_lung_subtyping", "task_adenocarcinoma"])
    parser.add_argument("--csv_path", default="dataset_csv/all_data.csv")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--text_prompt_path")
    parser.add_argument("--text_prompt", nargs="+")
    parser.add_argument("--prototype_number", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", default="analysis/stage3_model_design/04_scale_complementarity/stage342_independent_complementarity")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prompts = stage335._prompts(args)
    dataset, n_classes = stage335._dataset(args, prompts)
    all_comp, all_alpha, all_metrics, all_stats = [], [], [], []
    manifest = []
    for fold in args.folds:
        low_ckpt = _checkpoint_for_fold(args.low_checkpoint_template, int(fold))
        high_ckpt = _checkpoint_for_fold(args.high_checkpoint_template, int(fold))
        manifest.append({"fold": int(fold), "low_checkpoint": low_ckpt, "high_checkpoint": high_ckpt, "split": "val"})
        print(f"[Stage 3.4.2] fold={fold} split=val; ID-aligned independent Low/High checkpoints")
        comp, alpha, metrics, stats = _run_fold(args, dataset, n_classes, int(fold), prompts, output)
        all_comp.append(pd.DataFrame([comp])); all_alpha.append(alpha); all_metrics.append(metrics); all_stats.append(stats)
    pd.DataFrame(manifest).to_csv(output / "checkpoint_manifest.csv", index=False)
    pd.concat(all_comp, ignore_index=True).to_csv(output / "complementarity_summary.csv", index=False)
    pd.concat(all_alpha, ignore_index=True).to_csv(output / "alpha_sweep.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(output / "branch_metrics.csv", index=False)
    pd.concat(all_stats, ignore_index=True).to_csv(output / "logit_statistics.csv", index=False)
    print(pd.concat(all_metrics, ignore_index=True).to_string(index=False))
    print(pd.concat(all_comp, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
