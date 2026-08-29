#!/usr/bin/env python3
"""Stage 3.3.7 inference diagnostics for low-context spatial structure.

This script evaluates normal, context_mean, and residual_off on validation
slides and samples within-slide Low-parent cosine pairs. It does not train or
modify checkpoints.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import stage335_routing_diagnostics as stage335


DEVICE = stage335.DEVICE


def _seed_everything(seed: int) -> None:
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
        pattern = args.checkpoint_template.format(fold=fold)
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if len(matches) != 1:
            if not matches:
                raise FileNotFoundError(f"no checkpoint matches fold {fold}: {pattern}")
            raise RuntimeError(f"multiple checkpoints match fold {fold}: {matches}")
        return matches[0]
    if len(args.folds) != 1:
        raise ValueError("--checkpoint requires exactly one fold; use --checkpoint_template for multiple folds")
    return args.checkpoint


def _valid_low_parent_mask(mapping: dict, device: torch.device) -> torch.Tensor:
    low_valid = torch.as_tensor(mapping["low_valid_mask"], device=device, dtype=torch.bool)
    ptr = torch.as_tensor(mapping["parent_ptr"], device=device, dtype=torch.long)
    return low_valid & ((ptr[1:] - ptr[:-1]) > 0)


def low_spatial_statistics(
    low_features: torch.Tensor,
    mapping: dict,
    *,
    seed: int,
    pair_samples: int,
) -> dict[str, float | int | str]:
    """Compute reproducible sampled within-slide cosine statistics."""
    features = low_features.float()
    mask = _valid_low_parent_mask(mapping, features.device)
    valid = features[mask]
    count = int(valid.shape[0])
    result: dict[str, float | int | str] = {
        "valid_low_parent_count": count,
        "pair_sample_count": 0,
        "pair_sampling_seed": int(seed),
        "within_slide_cosine_mean": float("nan"),
        "within_slide_cosine_std": float("nan"),
        "within_slide_cosine_min": float("nan"),
        "within_slide_cosine_max": float("nan"),
        "low_feature_variance": float("nan"),
        "low_feature_dispersion": float("nan"),
    }
    if count == 0:
        return result

    mean_feature = valid.mean(dim=0)
    centered = valid - mean_feature
    result["low_feature_variance"] = float(centered.pow(2).mean().detach().cpu())
    result["low_feature_dispersion"] = float(centered.pow(2).sum(dim=1).mean().detach().cpu())
    if count < 2 or pair_samples <= 0:
        return result

    n_pairs = min(int(pair_samples), count * (count - 1) // 2)
    rng = np.random.default_rng(int(seed))
    left_parts, right_parts = [], []
    while sum(len(part) for part in left_parts) < n_pairs:
        remaining = n_pairs - sum(len(part) for part in left_parts)
        left_chunk = rng.integers(0, count, size=max(remaining * 2, 2))
        right_chunk = rng.integers(0, count, size=max(remaining * 2, 2))
        valid_pair = left_chunk != right_chunk
        left_parts.append(left_chunk[valid_pair])
        right_parts.append(right_chunk[valid_pair])
    left = np.concatenate(left_parts)[:n_pairs]
    right = np.concatenate(right_parts)[:n_pairs]
    if len(left) == 0:
        return result
    left_t = torch.as_tensor(left, device=features.device, dtype=torch.long)
    right_t = torch.as_tensor(right, device=features.device, dtype=torch.long)
    normalized = torch.nn.functional.normalize(valid, p=2, dim=1, eps=1e-8)
    cosine = (normalized[left_t] * normalized[right_t]).sum(dim=1)
    result.update({
        "pair_sample_count": int(cosine.numel()),
        "within_slide_cosine_mean": float(cosine.mean().detach().cpu()),
        "within_slide_cosine_std": float(cosine.std(unbiased=False).detach().cpu()),
        "within_slide_cosine_min": float(cosine.min().detach().cpu()),
        "within_slide_cosine_max": float(cosine.max().detach().cpu()),
    })
    return result


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


def _run_model(model, loader, fold: int, mode: str, n_classes: int):
    model.routing_diagnostic_mode = mode
    prediction_rows, diagnostics = [], []
    with torch.no_grad():
        for batch in loader:
            data_s, coord_s, data_l, coord_l, label = batch[:5]
            mapping = batch[6][0]
            slide_id = str(batch[5][0])
            data_s, coord_s = data_s.to(DEVICE), coord_s.to(DEVICE)
            data_l, coord_l, label = data_l.to(DEVICE), coord_l.to(DEVICE), label.to(DEVICE)
            probability, prediction, _ = model(
                data_s, coord_s, data_l, coord_l, label, mapping=mapping
            )
            probs = probability[0].detach().cpu().numpy()
            row = {"slide_id": slide_id, "Y": int(label.item()), "Y_hat": int(prediction.item())}
            row.update({f"p_{c}": float(probs[c]) for c in range(n_classes)})
            prediction_rows.append(row)
            diag = dict(getattr(model, "last_routing_diagnostics", {}) or {})
            diag.update({"slide_id": slide_id, "fold": int(fold), "mode": mode})
            diagnostics.append(diag)
    return pd.DataFrame(prediction_rows), diagnostics


def _paired_metrics(normal: pd.DataFrame, variant: pd.DataFrame, n_classes: int) -> dict[str, float | int]:
    normal = normal.sort_values("slide_id").reset_index(drop=True)
    variant = variant.sort_values("slide_id").reset_index(drop=True)
    if not normal["slide_id"].equals(variant["slide_id"]):
        raise ValueError("normal and diagnostic slide IDs do not match")
    probability_columns = [f"p_{c}" for c in range(n_classes)]
    delta = variant[probability_columns].to_numpy(dtype=float) - normal[probability_columns].to_numpy(dtype=float)
    normal_metrics = _prediction_metrics(normal, n_classes)
    variant_metrics = _prediction_metrics(variant, n_classes)
    variant_metrics.update({
        "delta_auc": float(variant_metrics["auc"] - normal_metrics["auc"]),
        "delta_accuracy": float(variant_metrics["accuracy"] - normal_metrics["accuracy"]),
        "delta_macro_f1": float(variant_metrics["macro_f1"] - normal_metrics["macro_f1"]),
        "mean_absolute_probability_change": float(np.abs(delta).mean()),
        "changed_prediction_count": int((variant["Y_hat"].to_numpy() != normal["Y_hat"].to_numpy()).sum()),
    })
    return variant_metrics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")


def evaluate_fold(args, dataset, n_classes: int, fold: int, prompts: list[str]) -> list[dict]:
    from utils.utils import get_simple_loader

    split_path = Path(args.splits_dir) / f"splits_{fold}.csv"
    split_dataset = dataset if args.split == "all" else dataset.return_splits(
        from_id=False, csv_path=str(split_path)
    )[ {"train": 0, "val": 1, "test": 2}[args.split] ]
    loader = get_simple_loader(split_dataset, batch_size=1, num_workers=0, mode="transformer")
    checkpoint = _checkpoint_for_fold(args, fold)
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"missing checkpoint for fold {fold}: {checkpoint}")
    args.checkpoint = checkpoint
    # Reuse Stage 3.3.5 _model(), which expects args.seed for router config.
    args.seed = int(args.baseline_seed)
    _seed_everything(args.baseline_seed)
    model = stage335._model(args, n_classes, prompts, "normal")
    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Low statistics are derived directly from each slide's low feature rows,
    # before context replacement; they are independent of inference mode.
    low_stats = []
    for batch in loader:
        data_s, _, _, _, _, slide_ids, mappings = batch
        slide_id = str(slide_ids[0])
        low_tensor = data_s[0] if data_s.ndim == 3 and data_s.shape[0] == 1 else data_s
        low_stats.append({
            "fold": int(fold), "slide_id": slide_id,
            **low_spatial_statistics(
                low_tensor, mappings[0], seed=args.pair_seed + len(low_stats), pair_samples=args.pair_samples
            ),
        })
    pd.DataFrame(low_stats).to_csv(fold_dir / "low_context_spatial_stats.csv", index=False)
    _write_jsonl(fold_dir / "low_context_spatial_stats.jsonl", low_stats)
    low_keys = (
        "valid_low_parent_count", "pair_sample_count", "within_slide_cosine_mean",
        "within_slide_cosine_std", "within_slide_cosine_min", "within_slide_cosine_max",
        "low_feature_variance", "low_feature_dispersion",
    )
    low_summary = []
    for summary_name, ddof in (("mean", 0), ("std", 0)):
        low_summary.append({
            "fold": int(fold), "summary": summary_name,
            **{key: float(np.nanstd([row[key] for row in low_stats], ddof=ddof))
               if summary_name == "std" else float(np.nanmean([row[key] for row in low_stats]))
               for key in low_keys},
        })
    pd.DataFrame(low_summary).to_csv(fold_dir / "low_context_spatial_summary.csv", index=False)

    normal_frame, normal_diag = _run_model(model, loader, fold, "normal", n_classes)
    normal_frame.to_csv(fold_dir / "normal_predictions.csv", index=False)
    _write_jsonl(fold_dir / "normal_diagnostics.jsonl", normal_diag)
    normal_metrics = _prediction_metrics(normal_frame, n_classes)
    rows = [{"fold": int(fold), "mode": "normal", **normal_metrics,
             "delta_auc": 0.0, "delta_accuracy": 0.0, "delta_macro_f1": 0.0,
             "mean_absolute_probability_change": 0.0, "changed_prediction_count": 0}]

    for mode in ("context_mean", "residual_off"):
        frame, diagnostics = _run_model(model, loader, fold, mode, n_classes)
        frame.to_csv(fold_dir / f"{mode}_predictions.csv", index=False)
        _write_jsonl(fold_dir / f"{mode}_diagnostics.jsonl", diagnostics)
        rows.append({"fold": int(fold), "mode": mode,
                     **_paired_metrics(normal_frame, frame, n_classes)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(fold_dir / "metrics.csv", index=False)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--checkpoint_template", help="format string containing {fold}")
    parser.add_argument("--data_root_dir", required=True)
    parser.add_argument("--data_folder_s", required=True)
    parser.add_argument("--data_folder_l", required=True)
    parser.add_argument("--mapping_path", required=True)
    parser.add_argument("--splits_dir", required=True)
    parser.add_argument("--task", required=True, choices=["task_tcga_rcc_subtyping", "task_tcga_lung_subtyping", "task_adenocarcinoma"])
    parser.add_argument("--csv_path", default="dataset_csv/all_data.csv")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--output_dir", default="analysis/stage3_model_design/03_low_context_routing/06_stabilization/stage337_low_context_spatial_discriminability")
    parser.add_argument("--text_prompt_path")
    parser.add_argument("--text_prompt", nargs="+")
    parser.add_argument("--prototype_number", type=int, default=16)
    parser.add_argument("--scale_mode", choices=["dual", "low", "high"], default="high")
    parser.add_argument("--baseline_seed", type=int, default=0)
    parser.add_argument("--pair_seed", type=int, default=337)
    parser.add_argument("--pair_samples", type=int, default=1024)
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
        print(f"[Stage 3.3.7] fold={fold} split={args.split} pair_seed={args.pair_seed}")
        all_rows.extend(evaluate_fold(args, dataset, n_classes, int(fold), prompts))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    summary_rows = []
    for fold in args.folds:
        fold_metrics = metrics[metrics["fold"] == int(fold)]
        normal = fold_metrics[fold_metrics["mode"] == "normal"].iloc[0]
        for mode in ("normal", "context_mean", "residual_off"):
            row = fold_metrics[fold_metrics["mode"] == mode].iloc[0]
            summary_rows.append({"fold": int(fold), "mode": mode,
                                 **{key: float(row[key]) for key in (
                                     "auc", "accuracy", "macro_f1", "delta_auc",
                                     "delta_accuracy", "delta_macro_f1",
                                     "mean_absolute_probability_change",
                                     "changed_prediction_count")}})
    pd.DataFrame(summary_rows).to_csv(output / "summary.csv", index=False)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, allow_nan=True)
    low_summary_frames = []
    for fold in args.folds:
        path = output / f"fold_{int(fold)}" / "low_context_spatial_summary.csv"
        if path.is_file():
            low_summary_frames.append(pd.read_csv(path))
    if low_summary_frames:
        pd.concat(low_summary_frames, ignore_index=True).to_csv(
            output / "low_context_spatial_summary.csv", index=False
        )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
