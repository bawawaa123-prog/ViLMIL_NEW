#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate attention heatmaps for BiomedCLIP-based ViLa-MIL checkpoints.

This script is designed for external cohorts such as Heyuan/Jiangxi where
features_biomedclip_5x, features_biomedclip_20x and WSI files already exist.
It loads a single checkpoint, runs forward_with_attention() for each slide,
and exports:

- attention_h5/<slide_id>_5x_attention.h5
- attention_h5/<slide_id>_20x_attention.h5
- heatmaps/<slide_id>_5x_heatmap.png
- heatmaps/<slide_id>_20x_heatmap.png
- predictions.csv

Recommended usage: generate one set of heatmaps for one selected fold.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
from utils.heatmap_utils import (
    create_attention_heatmap,
    create_attention_heatmap_scatter,
    save_attention_to_h5,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import ml_collections  # type: ignore
except ImportError:  # pragma: no cover - lightweight fallback for stricter environments
    from types import SimpleNamespace

    class _FallbackMLCollections:
        @staticmethod
        def ConfigDict():
            return SimpleNamespace()

    ml_collections = _FallbackMLCollections()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate attention heatmaps for BiomedCLIP ViLa-MIL")
    parser.add_argument("--csv_path", type=str, required=True, help="Dataset CSV containing case_id, slide_id, label")
    parser.add_argument("--data_root_dir", type=str, required=True, help="Root dir containing features folders")
    parser.add_argument("--data_folder_s", type=str, required=True, help="5x features folder under data_root_dir")
    parser.add_argument("--data_folder_l", type=str, required=True, help="20x features folder under data_root_dir")
    parser.add_argument("--wsi_root", type=str, required=True, help="Directory containing WSI files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fold checkpoint")
    parser.add_argument("--results_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--text_prompt_path", type=str, required=True, help="Prompt CSV")
    parser.add_argument("--prototype_number", type=int, default=16)
    parser.add_argument("--task", type=str, default="task_adenocarcinoma")
    parser.add_argument("--heatmap_downsample", type=int, default=64)
    parser.add_argument("--heatmap_alpha", type=float, default=0.78)
    parser.add_argument("--heatmap_style", type=str, default="paper", choices=["scatter", "region", "paper"])
    parser.add_argument("--point_size", type=int, default=18)
    parser.add_argument("--heatmap_cmap", type=str, default="paper", help="Colormap name, e.g. paper or jet")
    parser.add_argument("--max_slides", type=int, default=None, help="Optional limit for quick tests")
    parser.add_argument("--skip_heatmaps", action="store_true", default=False, help="Only export attention_h5 and predictions")
    parser.add_argument("--wsi_suffix", type=str, default=".svs", help="WSI file suffix")
    return parser.parse_args()


def load_text_prompts(prompt_csv: str) -> list[str]:
    df_tp = pd.read_csv(prompt_csv)
    cols = [c.strip().lower() for c in df_tp.columns]

    if "low_resolution_description" in cols and "high_resolution_description" in cols:
        low_idx = cols.index("low_resolution_description")
        high_idx = cols.index("high_resolution_description")
        low_prompts = df_tp.iloc[:, low_idx].astype(str).fillna("").tolist()
        high_prompts = df_tp.iloc[:, high_idx].astype(str).fillna("").tolist()
        return list(map(str, low_prompts)) + list(map(str, high_prompts))

    if len(df_tp.columns) >= 2:
        low_prompts = df_tp.iloc[:, -2].astype(str).fillna("").tolist()
        high_prompts = df_tp.iloc[:, -1].astype(str).fillna("").tolist()
        return list(map(str, low_prompts)) + list(map(str, high_prompts))

    arr = pd.read_csv(prompt_csv, header=None).values
    return [str(x) for x in arr.reshape(-1).tolist()]


def infer_label_dict(task: str) -> tuple[dict[str, int], int]:
    if task == "task_adenocarcinoma":
        return {"Adenocarcinoma": 0, "NonAdenocarcinoma": 1}, 2
    if task == "task_tcga_lung_subtyping":
        return {"LUAD": 0, "LUSC": 1}, 2
    if task == "task_tcga_rcc_subtyping":
        return {"CCRCC": 0, "PRCC": 1, "CRCC": 2}, 3
    raise NotImplementedError(f"Unsupported task: {task}")


def build_model(n_classes: int, text_prompt: list[str], prototype_number: int, checkpoint: str) -> ViLa_MIL_BiomedCLIP:
    config = ml_collections.ConfigDict()
    config.input_size = 512
    config.hidden_size = 192
    config.text_prompt = text_prompt
    config.prototype_number = prototype_number
    config.finetune_text_encoder = False

    model = ViLa_MIL_BiomedCLIP(config=config, num_classes=n_classes)
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_clean = {k.replace(".module", ""): v for k, v in ckpt.items() if "instance_loss_fn" not in k}
    model.load_state_dict(ckpt_clean, strict=True)
    model = model.to(device)
    model.eval()
    return model


def resolve_wsi_path(wsi_root: str, slide_id: str, suffix: str) -> str | None:
    direct = os.path.join(wsi_root, slide_id + suffix)
    if os.path.isfile(direct):
        return direct

    for candidate in Path(wsi_root).glob(f"{slide_id}.*"):
        if candidate.is_file():
            return str(candidate)
    return None


def main() -> int:
    args = parse_args()

    label_dict, n_classes = infer_label_dict(args.task)
    text_prompt = load_text_prompts(args.text_prompt_path)

    dataset = Generic_MIL_Dataset(
        csv_path=args.csv_path,
        mode="transformer",
        data_dir_s=os.path.join(args.data_root_dir, args.data_folder_s),
        data_dir_l=os.path.join(args.data_root_dir, args.data_folder_l),
        shuffle=False,
        print_info=True,
        label_dict=label_dict,
        patient_strat=False,
        ignore=[],
    )

    model = build_model(
        n_classes=n_classes,
        text_prompt=text_prompt,
        prototype_number=args.prototype_number,
        checkpoint=args.checkpoint,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    heatmap_dir = os.path.join(args.results_dir, "heatmaps")
    attention_h5_dir = os.path.join(args.results_dir, "attention_h5")
    os.makedirs(attention_h5_dir, exist_ok=True)
    if not args.skip_heatmaps:
        os.makedirs(heatmap_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    prediction_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, str]] = []

    with torch.no_grad():
        iterator = tqdm(loader, total=len(dataset), desc="Generating heatmaps", ncols=100)
        for idx, batch in enumerate(iterator):
            if args.max_slides is not None and idx >= args.max_slides:
                break

            features_s, coords_s, features_l, coords_l, label, slide_id = batch
            if isinstance(slide_id, (tuple, list)):
                slide_id = slide_id[0]
            if hasattr(slide_id, 'numel') and slide_id.numel() == 1:
                slide_id = str(slide_id.item())
            else:
                slide_id = str(slide_id)

            features_s = features_s.to(device)
            coords_s = coords_s.to(device)
            features_l = features_l.to(device)
            coords_l = coords_l.to(device)
            label = label.to(device)

            try:
                logits, y_prob, y_hat, attn_s, attn_l = model.forward_with_attention(
                    features_s, coords_s, features_l, coords_l, label
                )
            except Exception as exc:
                failed_rows.append({"slide_id": str(slide_id), "reason": f"forward_failed:{exc}"})
                continue

            coords_s_np = coords_s.cpu().numpy()
            coords_l_np = coords_l.cpu().numpy()
            if coords_s_np.ndim == 3:
                coords_s_np = coords_s_np[0]
            if coords_l_np.ndim == 3:
                coords_l_np = coords_l_np[0]

            attn_s_np = attn_s.cpu().numpy().reshape(-1)
            attn_l_np = attn_l.cpu().numpy().reshape(-1)

            if len(attn_s_np) != len(coords_s_np):
                failed_rows.append(
                    {"slide_id": str(slide_id), "reason": f"5x_length_mismatch:{len(attn_s_np)}!={len(coords_s_np)}"}
                )
                continue
            if len(attn_l_np) != len(coords_l_np):
                failed_rows.append(
                    {"slide_id": str(slide_id), "reason": f"20x_length_mismatch:{len(attn_l_np)}!={len(coords_l_np)}"}
                )
                continue

            wsi_path = resolve_wsi_path(args.wsi_root, str(slide_id), args.wsi_suffix)
            if wsi_path is None:
                failed_rows.append({"slide_id": str(slide_id), "reason": "wsi_missing"})
                continue

            save_attention_to_h5(attn_s_np, coords_s_np, os.path.join(attention_h5_dir, f"{slide_id}_5x_attention.h5"))
            save_attention_to_h5(attn_l_np, coords_l_np, os.path.join(attention_h5_dir, f"{slide_id}_20x_attention.h5"))

            if not args.skip_heatmaps:
                heatmap_path_5x = os.path.join(heatmap_dir, f"{slide_id}_5x_heatmap.png")
                heatmap_path_20x = os.path.join(heatmap_dir, f"{slide_id}_20x_heatmap.png")

                if args.heatmap_style == "scatter":
                    ok_5x = create_attention_heatmap_scatter(
                        attention_scores=attn_s_np,
                        coords=coords_s_np,
                        wsi_path=wsi_path,
                        output_path=heatmap_path_5x,
                        downsample=args.heatmap_downsample,
                        point_size=args.point_size,
                        alpha=args.heatmap_alpha,
                    )
                    ok_20x = create_attention_heatmap_scatter(
                        attention_scores=attn_l_np,
                        coords=coords_l_np,
                        wsi_path=wsi_path,
                        output_path=heatmap_path_20x,
                        downsample=args.heatmap_downsample,
                        point_size=args.point_size,
                        alpha=args.heatmap_alpha,
                    )
                elif args.heatmap_style == "region":
                    ok_5x = create_attention_heatmap(
                        attention_scores=attn_s_np,
                        coords=coords_s_np,
                        wsi_path=wsi_path,
                        output_path=heatmap_path_5x,
                        downsample=args.heatmap_downsample,
                        alpha=args.heatmap_alpha,
                        cmap=args.heatmap_cmap,
                        level=2,
                    )
                    ok_20x = create_attention_heatmap(
                        attention_scores=attn_l_np,
                        coords=coords_l_np,
                        wsi_path=wsi_path,
                        output_path=heatmap_path_20x,
                        downsample=args.heatmap_downsample,
                        alpha=args.heatmap_alpha,
                        cmap=args.heatmap_cmap,
                        level=1,
                    )
                else:
                    ok_5x = create_attention_heatmap(
                        attention_scores=attn_s_np,
                        coords=coords_s_np,
                        wsi_path=wsi_path,
                        output_path=heatmap_path_5x,
                        downsample=args.heatmap_downsample,
                        alpha=args.heatmap_alpha,
                        cmap=args.heatmap_cmap,
                        level=2,
                        blank_canvas=True,
                        canvas_color=(255, 255, 255),
                        keep_background_white=True,
                    )
                    ok_20x = create_attention_heatmap(
                        attention_scores=attn_l_np,
                        coords=coords_l_np,
                        wsi_path=wsi_path,
                        output_path=heatmap_path_20x,
                        downsample=args.heatmap_downsample,
                        alpha=args.heatmap_alpha,
                        cmap=args.heatmap_cmap,
                        level=1,
                        blank_canvas=True,
                        canvas_color=(255, 255, 255),
                        keep_background_white=True,
                    )

                if not (ok_5x or ok_20x):
                    failed_rows.append({"slide_id": str(slide_id), "reason": "heatmap_failed"})

            probs = y_prob.cpu().numpy()[0]
            prediction_rows.append(
                {
                    "slide_id": str(slide_id),
                    "true_label": int(label.item()),
                    "pred_label": int(y_hat.item()),
                    **{f"p_{c}": float(probs[c]) for c in range(n_classes)},
                    "n_5x_patches": int(len(attn_s_np)),
                    "n_20x_patches": int(len(attn_l_np)),
                }
            )

    pd.DataFrame(prediction_rows).to_csv(os.path.join(args.results_dir, "predictions.csv"), index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(os.path.join(args.results_dir, "failed_slides.csv"), index=False)

    print(f"Saved outputs to: {args.results_dir}")
    print(f"Successful slides: {len(prediction_rows)}")
    print(f"Failed slides: {len(failed_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
