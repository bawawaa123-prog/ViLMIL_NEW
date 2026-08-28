#!/usr/bin/env python3
"""Inference-only Stage 3.3.5 routing diagnostics.

Each mode loads the same Stage 3.3.4 checkpoint and evaluates the requested
split. No optimizer, checkpoint, or training artifact is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# When invoked as ``python tools/stage335_routing_diagnostics.py``, Python
# places ``tools/`` (rather than the repository root) on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODES = ("normal", "residual_off", "route_one", "route_mean", "route_shuffle", "context_shuffle")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prompts(args):
    if args.text_prompt:
        return [str(item) for item in args.text_prompt]
    if not args.text_prompt_path:
        raise ValueError("provide --text_prompt_path or --text_prompt")
    frame = pd.read_csv(args.text_prompt_path)
    cols = [str(c).strip().lower() for c in frame.columns]
    if "low_resolution_description" in cols and "high_resolution_description" in cols:
        low = frame.iloc[:, cols.index("low_resolution_description")].astype(str).tolist()
        high = frame.iloc[:, cols.index("high_resolution_description")].astype(str).tolist()
        return low + high
    if len(frame.columns) >= 2:
        return frame.iloc[:, -2].astype(str).tolist() + frame.iloc[:, -1].astype(str).tolist()
    return frame.iloc[:, 0].astype(str).tolist()


def _dataset(args, prompts):
    try:
        from datasets.dataset_generic import Generic_MIL_Dataset
    except ModuleNotFoundError as exc:
        if exc.name == "h5py":
            raise RuntimeError(
                "Stage 3.3.5 requires h5py to read feature H5 files. "
                "Install the project environment dependency (h5py 3.11.x) and rerun."
            ) from exc
        raise

    if args.task == "task_tcga_rcc_subtyping":
        csv_path, labels, n_classes = "dataset_csv/TCGA_RCC_subtyping.csv", {"CCRCC": 0, "PRCC": 1, "CRCC": 2}, 3
    elif args.task == "task_tcga_lung_subtyping":
        csv_path, labels, n_classes = "dataset_csv/TCGA_Lung_subtyping.csv", {"LUAD": 0, "LUSC": 1}, 2
    elif args.task == "task_adenocarcinoma":
        csv_path, labels, n_classes = args.csv_path, {"Adenocarcinoma": 0, "NonAdenocarcinoma": 1}, 2
    else:
        raise ValueError(f"unsupported task: {args.task}")
    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        mode="transformer",
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict=labels,
        patient_strat=False,
        ignore=[],
        mapping_path=args.mapping_path,
        return_mapping=True,
    )
    return dataset, n_classes


def _model(args, n_classes, prompts, mode):
    import ml_collections
    from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP

    config = ml_collections.ConfigDict()
    config.input_size = 512
    config.hidden_size = 192
    config.text_prompt = prompts
    config.prototype_number = args.prototype_number
    # Stage 3.3.4 checkpoints used for this diagnostic were trained with the
    # high-only scale ablation; keep that behavior explicit and configurable.
    config.scale_mode = args.scale_mode
    config.finetune_text_encoder = False
    config.use_low_context_routing = True
    config.use_routing_stabilization = True
    config.routing_diagnostic_mode = mode
    config.routing_diagnostic_seed = args.seed
    model = ViLa_MIL_BiomedCLIP(config=config, num_classes=n_classes)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = {key.replace(".module", ""): value for key, value in checkpoint.items() if "instance_loss_fn" not in key}
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model


def _auc(labels, probs, n_classes):
    if len(np.unique(labels)) <= 1:
        return float("nan")
    try:
        if n_classes == 2:
            return float(roc_auc_score(labels, probs[:, 1]))
        return float(roc_auc_score(labels, probs, multi_class="ovr"))
    except ValueError:
        return float("nan")


def evaluate(args, dataset, n_classes, fold, mode, prompts):
    from utils.utils import get_simple_loader

    split_path = Path(args.splits_dir) / f"splits_{fold}.csv"
    if args.split == "all":
        split_dataset = dataset
    else:
        split_dataset = dataset.return_splits(from_id=False, csv_path=str(split_path))[{"train": 0, "val": 1, "test": 2}[args.split]]
    loader = get_simple_loader(split_dataset, batch_size=1, num_workers=0, mode="transformer")
    model = _model(args, n_classes, prompts, mode)
    out = Path(args.output_dir) / mode
    out.mkdir(parents=True, exist_ok=True)
    prediction_rows, diagnostics = [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            data_s, coord_s, data_l, coord_l, label = batch[:5]
            mapping = batch[6][0]
            slide_id = str(batch[5][0])
            data_s, coord_s = data_s.to(DEVICE), coord_s.to(DEVICE)
            data_l, coord_l, label = data_l.to(DEVICE), coord_l.to(DEVICE), label.to(DEVICE)
            # Keep route_shuffle reproducible without changing normal inference.
            torch.manual_seed(args.seed + index)
            model.routing_diagnostic_seed = args.seed + index
            prob, pred, _ = model(data_s, coord_s, data_l, coord_l, label, mapping=mapping)
            probs = prob[0].detach().cpu().numpy()
            row = {"slide_id": slide_id, "Y": int(label.item()), "Y_hat": int(pred.item())}
            row.update({f"p_{c}": float(probs[c]) for c in range(n_classes)})
            prediction_rows.append(row)
            diag = dict(getattr(model, "last_routing_diagnostics", {}) or {})
            diag.update({"slide_id": slide_id, "fold": int(fold), "mode": mode})
            diagnostics.append(diag)

    pred_frame = pd.DataFrame(prediction_rows)
    labels = pred_frame["Y"].to_numpy(dtype=int)
    preds = pred_frame["Y_hat"].to_numpy(dtype=int)
    probs = pred_frame[[f"p_{c}" for c in range(n_classes)]].to_numpy(dtype=float)
    metrics = {
        "fold": int(fold), "mode": mode, "split": args.split,
        "n_slides": int(len(pred_frame)), "auc": _auc(labels, probs, n_classes),
        "acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
    # Fold-level means make mode comparisons possible without parsing JSONL.
    diagnostic_keys = (
        "mapped_route_mean", "mapped_route_std", "mapped_route_min", "mapped_route_max",
        "same_low_parent_route_std_mean", "same_low_parent_route_std_max",
        "residual_norm", "high_feature_norm", "residual_high_ratio",
        "residual_cap_trigger_rate",
        "residual_cap_trigger_rate_mapped",
    )
    for key in diagnostic_keys:
        values = [float(item[key]) for item in diagnostics if key in item]
        metrics[key] = float(np.mean(values)) if values else float("nan")
    pred_frame.to_csv(out / f"fold_{fold}_predictions.csv", index=False)
    with (out / f"fold_{fold}_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for diag in diagnostics:
            handle.write(json.dumps(diag, sort_keys=True) + "\n")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--models_exp_code", default=None, help="optional results subdirectory; checkpoint remains authoritative")
    parser.add_argument("--data_root_dir", required=True)
    parser.add_argument("--data_folder_s", required=True)
    parser.add_argument("--data_folder_l", required=True)
    parser.add_argument("--mapping_path", required=True)
    parser.add_argument("--splits_dir", required=True)
    parser.add_argument("--task", required=True, choices=["task_tcga_rcc_subtyping", "task_tcga_lung_subtyping", "task_adenocarcinoma"])
    parser.add_argument("--csv_path", default="dataset_csv/all_data.csv")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--output_dir", default="./stage335_diagnostics")
    parser.add_argument("--text_prompt_path")
    parser.add_argument("--text_prompt", nargs="+")
    parser.add_argument("--prototype_number", type=int, default=16)
    parser.add_argument("--scale_mode", choices=["dual", "low", "high"], default="high")
    parser.add_argument("--seed", type=int, default=335)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.mapping_path:
        raise ValueError("--mapping_path is required for Stage 3.3.5 diagnostics")
    prompts = _prompts(args)
    dataset, n_classes = _dataset(args, prompts)
    metrics = []
    for mode in args.modes:
        for fold in args.folds:
            split_path = Path(args.splits_dir) / f"splits_{fold}.csv"
            if args.split != "all" and not split_path.is_file():
                raise FileNotFoundError(f"missing split file: {split_path}")
            print(f"[Stage 3.3.5] mode={mode} fold={fold} split={args.split}")
            metrics.append(evaluate(args, dataset, n_classes, fold, mode, prompts))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
