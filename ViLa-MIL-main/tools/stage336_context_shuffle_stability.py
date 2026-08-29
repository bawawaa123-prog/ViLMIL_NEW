#!/usr/bin/env python3
"""Stage 3.3.6: multi-seed stability test for spatial context shuffling.

The normal prediction is computed once per fold.  The same checkpoint/model
is then evaluated with context_shuffle for each requested seed.  This is an
inference-only script and never writes checkpoints or training artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

# Support direct invocation from the repository root:
# ``python tools/stage336_context_shuffle_stability.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import stage335_routing_diagnostics as stage335


DEVICE = stage335.DEVICE


def _seed_everything(seed: int) -> None:
    """Make model construction and inference RNG state reproducible."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _checkpoint_for_fold(args, fold: int) -> str:
    if args.checkpoint_template:
        return args.checkpoint_template.format(fold=fold)
    if len(args.folds) != 1:
        raise ValueError("--checkpoint is only valid with one fold; use --checkpoint_template for multiple folds")
    return args.checkpoint


def _prediction_metrics(frame: pd.DataFrame, n_classes: int) -> dict[str, float | int]:
    labels = frame["Y"].to_numpy(dtype=int)
    preds = frame["Y_hat"].to_numpy(dtype=int)
    probs = frame[[f"p_{c}" for c in range(n_classes)]].to_numpy(dtype=float)
    return {
        "n_slides": int(len(frame)),
        "auc": stage335._auc(labels, probs, n_classes),
        "accuracy": float((labels == preds).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def _run_model(model, loader, fold: int, mode: str, seed: int | None, n_classes: int):
    """Evaluate one fold and return per-slide predictions and diagnostics."""
    prediction_rows = []
    diagnostics = []
    if seed is not None:
        model.routing_diagnostic_mode = "context_shuffle"
    else:
        model.routing_diagnostic_mode = "normal"

    with torch.no_grad():
        for index, batch in enumerate(loader):
            data_s, coord_s, data_l, coord_l, label = batch[:5]
            mapping = batch[6][0]
            slide_id = str(batch[5][0])
            data_s, coord_s = data_s.to(DEVICE), coord_s.to(DEVICE)
            data_l, coord_l, label = data_l.to(DEVICE), coord_l.to(DEVICE), label.to(DEVICE)

            # Existing Stage 3.3.5 uses a slide-specific deterministic seed.
            # Keep that convention, with the explicit Stage 3.3.6 seed as the
            # base.  The model's CPU Generator controls the actual shuffle.
            if seed is not None:
                slide_seed = int(seed) + index
                _seed_everything(slide_seed)
                model.routing_diagnostic_seed = slide_seed
            else:
                _seed_everything(0)

            prob, pred, _ = model(data_s, coord_s, data_l, coord_l, label, mapping=mapping)
            probs = prob[0].detach().cpu().numpy()
            row = {"slide_id": slide_id, "Y": int(label.item()), "Y_hat": int(pred.item())}
            row.update({f"p_{c}": float(probs[c]) for c in range(n_classes)})
            prediction_rows.append(row)
            diag = dict(getattr(model, "last_routing_diagnostics", {}) or {})
            diag.update({"slide_id": slide_id, "fold": int(fold), "mode": mode})
            if seed is not None:
                diag["shuffle_seed"] = int(seed)
                diag["slide_shuffle_seed"] = int(slide_seed)
            diagnostics.append(diag)

    return pd.DataFrame(prediction_rows), diagnostics


def _paired_metrics(normal: pd.DataFrame, shuffled: pd.DataFrame, n_classes: int) -> dict[str, float | int]:
    normal = normal.sort_values("slide_id").reset_index(drop=True)
    shuffled = shuffled.sort_values("slide_id").reset_index(drop=True)
    if not normal["slide_id"].equals(shuffled["slide_id"]):
        raise ValueError("normal and context_shuffle slide IDs do not match")
    probability_columns = [f"p_{c}" for c in range(n_classes)]
    probability_delta = shuffled[probability_columns].to_numpy(dtype=float) - normal[probability_columns].to_numpy(dtype=float)
    shuffled_metrics = _prediction_metrics(shuffled, n_classes)
    normal_metrics = _prediction_metrics(normal, n_classes)
    shuffled_metrics.update({
        "delta_auc": float(shuffled_metrics["auc"] - normal_metrics["auc"]),
        "delta_accuracy": float(shuffled_metrics["accuracy"] - normal_metrics["accuracy"]),
        "delta_macro_f1": float(shuffled_metrics["macro_f1"] - normal_metrics["macro_f1"]),
        "mean_absolute_probability_change": float(np.abs(probability_delta).mean()),
        "changed_prediction_count": int((shuffled["Y_hat"].to_numpy() != normal["Y_hat"].to_numpy()).sum()),
    })
    return shuffled_metrics


def _write_diagnostics(path: Path, diagnostics: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in diagnostics:
            handle.write(json.dumps(item, sort_keys=True) + "\n")


def evaluate_fold(args, dataset, n_classes: int, fold: int, prompts: list[str]) -> list[dict]:
    from utils.utils import get_simple_loader

    split_path = Path(args.splits_dir) / f"splits_{fold}.csv"
    if args.split == "all":
        split_dataset = dataset
    else:
        split_dataset = dataset.return_splits(
            from_id=False,
            csv_path=str(split_path),
        )[ {"train": 0, "val": 1, "test": 2}[args.split] ]
    loader = get_simple_loader(split_dataset, batch_size=1, num_workers=0, mode="transformer")

    checkpoint = _checkpoint_for_fold(args, fold)
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"missing checkpoint for fold {fold}: {checkpoint}")
    args.checkpoint = checkpoint
    args.seed = int(args.baseline_seed)
    _seed_everything(args.baseline_seed)
    model = stage335._model(args, n_classes, prompts, "normal")

    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    normal_frame, normal_diagnostics = _run_model(model, loader, fold, "normal", None, n_classes)
    normal_frame.to_csv(fold_dir / "normal_predictions.csv", index=False)
    _write_diagnostics(fold_dir / "normal_diagnostics.jsonl", normal_diagnostics)
    normal_metrics = _prediction_metrics(normal_frame, n_classes)
    rows = [{
        "fold": int(fold), "run": "normal", "shuffle_seed": "",
        **normal_metrics, "delta_auc": 0.0, "delta_accuracy": 0.0,
        "delta_macro_f1": 0.0, "mean_absolute_probability_change": 0.0,
        "changed_prediction_count": 0,
    }]

    for seed in args.seeds:
        seed = int(seed)
        shuffled_frame, diagnostics = _run_model(model, loader, fold, "context_shuffle", seed, n_classes)
        shuffled_frame.to_csv(fold_dir / f"context_shuffle_seed_{seed}_predictions.csv", index=False)
        _write_diagnostics(fold_dir / f"context_shuffle_seed_{seed}_diagnostics.jsonl", diagnostics)
        metrics = _paired_metrics(normal_frame, shuffled_frame, n_classes)
        rows.append({"fold": int(fold), "run": "context_shuffle", "shuffle_seed": seed, **metrics})

    frame = pd.DataFrame(rows)
    frame.to_csv(fold_dir / "metrics.csv", index=False)
    numeric = ["auc", "accuracy", "macro_f1", "delta_auc", "delta_accuracy", "delta_macro_f1", "mean_absolute_probability_change", "changed_prediction_count"]
    shuffle_frame = frame[frame["run"] == "context_shuffle"]
    summary_rows = [
        {"fold": int(fold), "summary": "normal", **{key: float(frame.iloc[0][key]) for key in numeric}},
        {"fold": int(fold), "summary": "shuffle_mean", **{key: float(shuffle_frame[key].mean()) for key in numeric}},
        {"fold": int(fold), "summary": "shuffle_std", **{key: float(shuffle_frame[key].std(ddof=0)) for key in numeric}},
    ]
    pd.DataFrame(summary_rows).to_csv(fold_dir / "summary.csv", index=False)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint")
    checkpoint_group.add_argument("--checkpoint_template", help="format string containing {fold}")
    parser.add_argument("--data_root_dir", required=True)
    parser.add_argument("--data_folder_s", required=True)
    parser.add_argument("--data_folder_l", required=True)
    parser.add_argument("--mapping_path", required=True)
    parser.add_argument("--splits_dir", required=True)
    parser.add_argument("--task", required=True, choices=["task_tcga_rcc_subtyping", "task_tcga_lung_subtyping", "task_adenocarcinoma"])
    parser.add_argument("--csv_path", default="dataset_csv/all_data.csv")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output_dir", default="analysis/stage3_model_design/03_low_context_routing/06_stabilization/stage336_spatial_mapping_robustness")
    parser.add_argument("--text_prompt_path")
    parser.add_argument("--text_prompt", nargs="+")
    parser.add_argument("--prototype_number", type=int, default=16)
    parser.add_argument("--scale_mode", choices=["dual", "low", "high"], default="high")
    parser.add_argument("--baseline_seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.split != "all":
        for fold in args.folds:
            split_path = Path(args.splits_dir) / f"splits_{fold}.csv"
            if not split_path.is_file():
                raise FileNotFoundError(f"missing split file: {split_path}")
    prompts = stage335._prompts(args)
    dataset, n_classes = stage335._dataset(args, prompts)
    all_rows = []
    for fold in args.folds:
        print(f"[Stage 3.3.6] fold={fold} split={args.split} seeds={args.seeds}")
        all_rows.extend(evaluate_fold(args, dataset, n_classes, int(fold), prompts))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(output / "per_seed_metrics.csv", index=False)
    summary_rows = []
    numeric = ["auc", "accuracy", "macro_f1", "delta_auc", "delta_accuracy", "delta_macro_f1", "mean_absolute_probability_change", "changed_prediction_count"]
    for fold in args.folds:
        fold_frame = per_seed[per_seed["fold"] == int(fold)]
        normal = fold_frame[fold_frame["run"] == "normal"].iloc[0]
        shuffled = fold_frame[fold_frame["run"] == "context_shuffle"]
        summary_rows.extend([
            {"fold": int(fold), "summary": "normal", **{key: float(normal[key]) for key in numeric}},
            {"fold": int(fold), "summary": "shuffle_mean", **{key: float(shuffled[key].mean()) for key in numeric}},
            {"fold": int(fold), "summary": "shuffle_std", **{key: float(shuffled[key].std(ddof=0)) for key in numeric}},
        ])
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, allow_nan=True)
    print(per_seed.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
