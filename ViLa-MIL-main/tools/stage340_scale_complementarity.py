#!/usr/bin/env python3
"""Stage 3.4.0 validation-only Low/High predictive complementarity analysis."""

from __future__ import annotations

import argparse
import glob
import json
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


def _checkpoint_for_fold(args, fold: int) -> str:
    pattern = args.checkpoint_template.format(fold=fold)
    matches = sorted(glob.glob(os.path.expanduser(pattern)))
    if len(matches) != 1:
        raise RuntimeError(f"expected one checkpoint for fold {fold}, got {matches} from {pattern}")
    return matches[0]


def _model(args, n_classes, prompts, fold):
    args.checkpoint = _checkpoint_for_fold(args, fold)
    args.seed = int(args.seed)
    # Reuse the Stage 3.3.5 model builder/checkpoint loader. Normal routing is
    # retained because these are the deployed Stage 3.3.4 checkpoints.
    return stage335._model(args, n_classes, prompts, "normal")


def _metrics(labels, logits):
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    try:
        auc = float(roc_auc_score(labels, probs[:, 1])) if len(np.unique(labels)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "probabilities": probs,
        "predictions": preds,
    }


def _collect(args, dataset, n_classes, fold, prompts):
    from utils.utils import get_simple_loader

    split_path = Path(args.splits_dir) / f"splits_{fold}.csv"
    split_dataset = dataset.return_splits(from_id=False, csv_path=str(split_path))[1]
    loader = get_simple_loader(split_dataset, batch_size=1, num_workers=0, mode="transformer")
    model = _model(args, n_classes, prompts, fold)
    rows = []
    with torch.no_grad():
        for batch in loader:
            data_s, coord_s, data_l, coord_l, label = batch[:5]
            mapping = batch[6][0] if len(batch) >= 7 else None
            slide_id = str(batch[5][0]) if len(batch) >= 6 else str(len(rows))
            data_s, coord_s = data_s.to(DEVICE), coord_s.to(DEVICE)
            data_l, coord_l, label = data_l.to(DEVICE), coord_l.to(DEVICE), label.to(DEVICE)
            _, _, _, branches = model(
                data_s, coord_s, data_l, coord_l, label,
                mapping=mapping, return_branch_logits=True,
            )
            low = branches["logits_low"][0].detach().cpu().numpy()
            high = branches["logits_high"][0].detach().cpu().numpy()
            dual = branches["logits_dual"][0].detach().cpu().numpy()
            row = {"slide_id": slide_id, "label": int(label.item())}
            for name, values in (("logits_low", low), ("logits_high", high)):
                for c, value in enumerate(values):
                    row[f"{name}_{c}"] = float(value)
            for name, values in (("probability_low", _metrics([label.item()], [low])["probabilities"][0]),
                                 ("probability_high", _metrics([label.item()], [high])["probabilities"][0]),
                                 ("probability_dual", _metrics([label.item()], [dual])["probabilities"][0])):
                for c, value in enumerate(values):
                    row[f"{name}_{c}"] = float(value)
            row["prediction_low"] = int(np.argmax(low))
            row["prediction_high"] = int(np.argmax(high))
            row["prediction_dual"] = int(np.argmax(dual))
            rows.append(row)
    return pd.DataFrame(rows)


def _run_fold(args, dataset, n_classes, fold, prompts, output):
    frame = _collect(args, dataset, n_classes, fold, prompts)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"fold_{fold}_branch_predictions.csv", index=False)
    labels = frame.label.to_numpy(dtype=int)
    low = frame[[f"logits_low_{c}" for c in range(n_classes)]].to_numpy()
    high = frame[[f"logits_high_{c}" for c in range(n_classes)]].to_numpy()
    dual = low + high
    low_m, high_m, dual_m = _metrics(labels, low), _metrics(labels, high), _metrics(labels, dual)
    low_pred, high_pred = low_m["predictions"], high_m["predictions"]
    correct_low, correct_high = low_pred == labels, high_pred == labels
    comp = pd.DataFrame([{
        "fold": fold, "n_slides": len(labels),
        "both_correct": int(np.sum(correct_low & correct_high)),
        "high_correct_low_wrong": int(np.sum(correct_high & ~correct_low)),
        "high_wrong_low_correct": int(np.sum(~correct_high & correct_low)),
        "both_wrong": int(np.sum(~correct_high & ~correct_low)),
        "prediction_disagreement_count": int(np.sum(low_pred != high_pred)),
        "prediction_disagreement_rate": float(np.mean(low_pred != high_pred)),
        "oracle_correct": int(np.sum(correct_low | correct_high)),
        "oracle_accuracy": float(np.mean(correct_low | correct_high)),
        "high_wrong_low_correct_rate": float(np.mean(~correct_high & correct_low)),
        "high_wrong_count": int(np.sum(~correct_high)),
        "high_wrong_low_correct_conditional_rate": float(
            np.sum(~correct_high & correct_low) / max(1, np.sum(~correct_high))
        ),
    }])
    comp.to_csv(output / f"fold_{fold}_complementarity_summary.csv", index=False)
    alpha_rows = []
    for alpha in np.linspace(0.0, 1.0, 11):
        m = _metrics(labels, alpha * high + (1.0 - alpha) * low)
        alpha_rows.append({"fold": fold, "alpha": float(alpha), "auc": m["auc"],
                           "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    alpha_frame = pd.DataFrame(alpha_rows)
    alpha_frame["is_best_auc"] = alpha_frame.auc == alpha_frame.auc.max()
    alpha_frame["is_best_accuracy"] = alpha_frame.accuracy == alpha_frame.accuracy.max()
    alpha_frame["is_best_macro_f1"] = alpha_frame.macro_f1 == alpha_frame.macro_f1.max()
    alpha_frame.to_csv(output / f"fold_{fold}_alpha_sweep.csv", index=False)
    metric_rows = []
    for name, m in (("low", low_m), ("high", high_m), ("dual", dual_m)):
        metric_rows.append({"fold": fold, "mode": name, "auc": m["auc"],
                            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    pd.DataFrame(metric_rows).to_csv(output / f"fold_{fold}_branch_metrics.csv", index=False)
    return frame, comp, alpha_frame, pd.DataFrame(metric_rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint_template", required=True)
    p.add_argument("--data_root_dir", required=True)
    p.add_argument("--data_folder_s", required=True)
    p.add_argument("--data_folder_l", required=True)
    p.add_argument("--mapping_path", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--task", required=True, choices=["task_tcga_rcc_subtyping", "task_tcga_lung_subtyping", "task_adenocarcinoma"])
    p.add_argument("--csv_path", default="dataset_csv/all_data.csv")
    p.add_argument("--split", choices=["val"], default="val")
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--text_prompt_path")
    p.add_argument("--text_prompt", nargs="+")
    p.add_argument("--prototype_number", type=int, default=16)
    p.add_argument("--scale_mode", choices=["dual", "low", "high"], default="high")
    p.add_argument("--seed", type=int, default=340)
    p.add_argument("--output_dir", default="analysis/stage3_model_design/04_scale_complementarity/stage340_scale_complementarity")
    return p.parse_args()


def main():
    args = parse_args()
    # Create the destination before dataset/model initialization so callers
    # can safely tee logs into the same directory.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    prompts = stage335._prompts(args)
    dataset, n_classes = stage335._dataset(args, prompts)
    out = Path(args.output_dir)
    all_comp, all_alpha, all_metrics = [], [], []
    for fold in args.folds:
        print(f"[Stage 3.4.0] fold={fold} split=val")
        _, comp, alpha, metrics = _run_fold(args, dataset, n_classes, int(fold), prompts, out)
        all_comp.append(comp); all_alpha.append(alpha); all_metrics.append(metrics)
    pd.concat(all_comp, ignore_index=True).to_csv(out / "complementarity_summary.csv", index=False)
    pd.concat(all_alpha, ignore_index=True).to_csv(out / "alpha_sweep.csv", index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(out / "branch_metrics.csv", index=False)
    print(pd.concat(all_metrics, ignore_index=True).to_string(index=False))
    print(pd.concat(all_comp, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
