#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Jiangxi-specific overlay attention heatmaps and optionally export top-k patches.

This script is adapted for the Jiangxi rerun pipeline where:
- WSI files are single-level TIFF / OME-TIFF
- 5x branch uses level-0 coords generated with patch_size=4096
- 20x branch uses level-0 coords generated with patch_size=1024
- Patch filenames may be either:
  - <slide_id>_<x>_<y>.png
  - <slide_id>_256_<x>_<y>.png

Outputs:
- attention_h5/<slide_id>_5x_attention.h5
- attention_h5/<slide_id>_20x_attention.h5
- overlay_heatmaps/5x/<slide_id>_5x_overlay_heatmap.png
- overlay_heatmaps/20x/<slide_id>_20x_overlay_heatmap.png
- predictions.csv

Optional outputs:
- top_patches/5x/top/<slide_id>/...
- top_patches/20x/top/<slide_id>/...
- top_patches/top_patches_manifest.csv
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import openslide
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_generic import Generic_MIL_Dataset
from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
from utils.heatmap_utils import get_attention_colormap, save_attention_to_h5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import ml_collections  # type: ignore
except ImportError:  # pragma: no cover
    from types import SimpleNamespace

    class _FallbackMLCollections:
        @staticmethod
        def ConfigDict():
            return SimpleNamespace()

    ml_collections = _FallbackMLCollections()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jiangxi overlay attention heatmaps and top-k patches")
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
    parser.add_argument("--heatmap_alpha", type=float, default=0.72)
    parser.add_argument("--heatmap_cmap", type=str, default="soft_paper")
    parser.add_argument("--saturation_boost", type=float, default=1.0)
    parser.add_argument("--value_boost", type=float, default=1.0)
    parser.add_argument("--max_slides", type=int, default=None)
    parser.add_argument("--skip_heatmaps", action="store_true", default=False, help="Only export attention_h5 and predictions")
    parser.add_argument("--wsi_suffix", type=str, default=".tif", help="WSI file suffix for direct lookup")
    parser.add_argument("--patch_size_5x_level0", type=int, default=4096, help="Jiangxi 5x effective level-0 patch size")
    parser.add_argument("--patch_size_20x_level0", type=int, default=1024, help="Jiangxi 20x effective level-0 patch size")
    parser.add_argument("--show_colorbar", action="store_true", default=True, help="Show colorbar legend")

    parser.add_argument("--export_top_k", type=int, default=0, help="If > 0, export top-k patches after attention generation")
    parser.add_argument("--patches_5x_dir", type=str, default=None, help="Root patches_5x directory for top-k export")
    parser.add_argument("--patches_20x_dir", type=str, default=None, help="Root patches_20x directory for top-k export")
    parser.add_argument("--copy_mode", type=str, default="copy", choices=["copy", "symlink"])
    parser.add_argument("--export_bottom_k", type=int, default=0)
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


def patch_filename_candidates(slide_id: str, x: int, y: int) -> list[str]:
    x = int(x)
    y = int(y)
    return [
        f"{slide_id}_{x}_{y}.png",
        f"{slide_id}_256_{x}_{y}.png",
    ]


def build_attention_overlay(*, slide: openslide.OpenSlide, coords: np.ndarray, scores: np.ndarray, patch_extent: int,
                            downsample: int, alpha: float, cmap_name: str,
                            saturation_boost: float, value_boost: float) -> tuple[np.ndarray, np.ndarray]:
    w, h = slide.dimensions
    thumb_w = max(1, int(np.ceil(w / downsample)))
    thumb_h = max(1, int(np.ceil(h / downsample)))
    background = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
    thumb_h, thumb_w = background.shape[:2]

    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    heatmap = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    count_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)

    patch_w = max(1, int(np.ceil(patch_extent / downsample)))
    patch_h = max(1, int(np.ceil(patch_extent / downsample)))

    for (x0, y0), score in zip(coords, scores_norm):
        x0 = int(x0)
        y0 = int(y0)
        x_thumb = int(np.floor(x0 / downsample))
        y_thumb = int(np.floor(y0 / downsample))
        x_end = min(thumb_w, max(x_thumb + 1, x_thumb + patch_w))
        y_end = min(thumb_h, max(y_thumb + 1, y_thumb + patch_h))
        if x_thumb >= thumb_w or y_thumb >= thumb_h:
            continue
        heatmap[y_thumb:y_end, x_thumb:x_end] += float(score)
        count_map[y_thumb:y_end, x_thumb:x_end] += 1

    heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map != 0)
    cmap = get_attention_colormap(cmap_name)
    heatmap_rgb = (cmap(heatmap) * 255).astype(np.uint8)[:, :, :3]

    if heatmap_rgb.shape[:2] != background.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb, (thumb_w, thumb_h), interpolation=cv2.INTER_LINEAR)
    if count_map.shape[:2] != background.shape[:2]:
        count_map = cv2.resize(count_map, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)

    overlay = background.copy()
    valid_mask = count_map > 0
    if np.any(valid_mask):
        blended = cv2.addWeighted(background, 1 - alpha, heatmap_rgb, alpha, 0)
        overlay[valid_mask] = blended[valid_mask]

        hsv = cv2.cvtColor(overlay, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        sat[valid_mask] = np.clip(sat[valid_mask] * saturation_boost, 0, 255)
        val[valid_mask] = np.clip(val[valid_mask] * value_boost, 0, 255)
        hsv[:, :, 1] = sat
        hsv[:, :, 2] = val
        overlay = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return overlay, heatmap


def save_overlay_figure(overlay: np.ndarray, cmap_name: str, output_path: str, title: str, show_colorbar: bool = True) -> None:
    fig_w = max(8.0, overlay.shape[1] / 95.0)
    fig_h = max(7.0, overlay.shape[0] / 95.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.imshow(overlay)
    ax.set_axis_off()
    ax.set_title(title, fontsize=18, pad=14)

    if show_colorbar:
        sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=get_attention_colormap(cmap_name))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Normalized attention", rotation=270, labelpad=18)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def export_one_scale(slide_id: str, scale_name: str, attention_h5: str, patches_root: str, output_root: str,
                     top_k: int, export_bottom_k: int, copy_mode: str) -> list[dict[str, object]]:
    with h5py.File(attention_h5, "r") as f:
        attention = np.asarray(f["attention"]).reshape(-1)
        coords = np.asarray(f["coords"])

    if len(attention) != len(coords):
        raise ValueError(f"Length mismatch in {attention_h5}: attn={len(attention)} coords={len(coords)}")

    patch_dir = os.path.join(patches_root, slide_id)
    if not os.path.isdir(patch_dir):
        raise FileNotFoundError(f"Patch dir missing: {patch_dir}")

    sort_idx_desc = np.argsort(-attention)
    records: list[dict[str, object]] = []

    def export_indices(indices: np.ndarray, split_name: str) -> None:
        split_dir = os.path.join(output_root, scale_name, split_name, slide_id)
        os.makedirs(split_dir, exist_ok=True)
        for rank, idx in enumerate(indices, start=1):
            x, y = coords[idx]
            src_path = None
            src_name = None
            for candidate in patch_filename_candidates(slide_id, x, y):
                candidate_path = os.path.join(patch_dir, candidate)
                if os.path.isfile(candidate_path):
                    src_path = candidate_path
                    src_name = candidate
                    break
            if src_path is None or src_name is None:
                continue
            dst_name = f"rank_{rank:03d}_score_{attention[idx]:.6f}_{src_name}"
            dst_path = os.path.join(split_dir, dst_name)
            if copy_mode == "symlink":
                if os.path.lexists(dst_path):
                    os.unlink(dst_path)
                os.symlink(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            records.append({
                "slide_id": slide_id,
                "scale": scale_name,
                "split": split_name,
                "rank": rank,
                "attention": float(attention[idx]),
                "x": int(x),
                "y": int(y),
                "src_path": src_path,
                "dst_path": dst_path,
            })

    top_n = min(top_k, len(sort_idx_desc))
    export_indices(sort_idx_desc[:top_n], "top")

    if export_bottom_k > 0:
        bottom_n = min(export_bottom_k, len(sort_idx_desc))
        export_indices(sort_idx_desc[::-1][:bottom_n], "bottom")

    return records


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
    overlay_dir = os.path.join(args.results_dir, "overlay_heatmaps")
    overlay_dir_5x = os.path.join(overlay_dir, "5x")
    overlay_dir_20x = os.path.join(overlay_dir, "20x")
    attention_h5_dir = os.path.join(args.results_dir, "attention_h5")
    os.makedirs(attention_h5_dir, exist_ok=True)
    if not args.skip_heatmaps:
        os.makedirs(overlay_dir_5x, exist_ok=True)
        os.makedirs(overlay_dir_20x, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    prediction_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, str]] = []
    overlay_rows: list[dict[str, object]] = []

    with torch.no_grad():
        iterator = tqdm(loader, total=len(dataset), desc="Generating Jiangxi overlay heatmaps", ncols=100)
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
                failed_rows.append({"slide_id": str(slide_id), "reason": f"5x_length_mismatch:{len(attn_s_np)}!={len(coords_s_np)}"})
                continue
            if len(attn_l_np) != len(coords_l_np):
                failed_rows.append({"slide_id": str(slide_id), "reason": f"20x_length_mismatch:{len(attn_l_np)}!={len(coords_l_np)}"})
                continue

            wsi_path = resolve_wsi_path(args.wsi_root, str(slide_id), args.wsi_suffix)
            if wsi_path is None:
                failed_rows.append({"slide_id": str(slide_id), "reason": "wsi_missing"})
                continue

            save_attention_to_h5(attn_s_np, coords_s_np, os.path.join(attention_h5_dir, f"{slide_id}_5x_attention.h5"))
            save_attention_to_h5(attn_l_np, coords_l_np, os.path.join(attention_h5_dir, f"{slide_id}_20x_attention.h5"))

            if not args.skip_heatmaps:
                try:
                    slide = openslide.OpenSlide(wsi_path)
                    try:
                        overlay_5x, _ = build_attention_overlay(
                            slide=slide,
                            coords=coords_s_np,
                            scores=attn_s_np,
                            patch_extent=args.patch_size_5x_level0,
                            downsample=args.heatmap_downsample,
                            alpha=args.heatmap_alpha,
                            cmap_name=args.heatmap_cmap,
                            saturation_boost=args.saturation_boost,
                            value_boost=args.value_boost,
                        )
                        overlay_20x, _ = build_attention_overlay(
                            slide=slide,
                            coords=coords_l_np,
                            scores=attn_l_np,
                            patch_extent=args.patch_size_20x_level0,
                            downsample=args.heatmap_downsample,
                            alpha=args.heatmap_alpha,
                            cmap_name=args.heatmap_cmap,
                            saturation_boost=args.saturation_boost,
                            value_boost=args.value_boost,
                        )
                    finally:
                        slide.close()

                    out_path_5x = os.path.join(overlay_dir_5x, f"{slide_id}_5x_overlay_heatmap.png")
                    out_path_20x = os.path.join(overlay_dir_20x, f"{slide_id}_20x_overlay_heatmap.png")
                    save_overlay_figure(overlay_5x, args.heatmap_cmap, out_path_5x, f"{slide_id} (5x)", args.show_colorbar)
                    save_overlay_figure(overlay_20x, args.heatmap_cmap, out_path_20x, f"{slide_id} (20x)", args.show_colorbar)
                    overlay_rows.append({"slide_id": str(slide_id), "scale": "5x", "output_path": out_path_5x, "num_patches": int(len(coords_s_np))})
                    overlay_rows.append({"slide_id": str(slide_id), "scale": "20x", "output_path": out_path_20x, "num_patches": int(len(coords_l_np))})
                except Exception as exc:
                    failed_rows.append({"slide_id": str(slide_id), "reason": f"overlay_failed:{exc}"})

            probs = y_prob.cpu().numpy()[0]
            prediction_rows.append({
                "slide_id": str(slide_id),
                "true_label": int(label.item()),
                "pred_label": int(y_hat.item()),
                **{f"p_{c}": float(probs[c]) for c in range(n_classes)},
                "n_5x_patches": int(len(attn_s_np)),
                "n_20x_patches": int(len(attn_l_np)),
            })

    pd.DataFrame(prediction_rows).to_csv(os.path.join(args.results_dir, "predictions.csv"), index=False)
    if overlay_rows:
        pd.DataFrame(overlay_rows).to_csv(os.path.join(args.results_dir, "overlay_manifest.csv"), index=False)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(os.path.join(args.results_dir, "failed_slides.csv"), index=False)

    if args.export_top_k > 0:
        if not args.patches_5x_dir or not args.patches_20x_dir:
            raise ValueError("When --export_top_k > 0, both --patches_5x_dir and --patches_20x_dir are required")

        top_output_dir = os.path.join(args.results_dir, "top_patches")
        os.makedirs(top_output_dir, exist_ok=True)

        manifest_rows: list[dict[str, object]] = []
        export_failed_rows: list[dict[str, str]] = []
        slide_ids = pd.read_csv(args.csv_path)["slide_id"].dropna().astype(str).tolist()
        if args.max_slides is not None:
            slide_ids = slide_ids[: args.max_slides]

        for slide_id in slide_ids:
            for scale_name, suffix, patches_root in [
                ("5x", "_5x_attention.h5", args.patches_5x_dir),
                ("20x", "_20x_attention.h5", args.patches_20x_dir),
            ]:
                attention_h5 = os.path.join(attention_h5_dir, f"{slide_id}{suffix}")
                if not os.path.isfile(attention_h5):
                    export_failed_rows.append({"slide_id": slide_id, "scale": scale_name, "reason": "attention_h5_missing"})
                    continue
                try:
                    manifest_rows.extend(
                        export_one_scale(
                            slide_id=slide_id,
                            scale_name=scale_name,
                            attention_h5=attention_h5,
                            patches_root=patches_root,
                            output_root=top_output_dir,
                            top_k=args.export_top_k,
                            export_bottom_k=args.export_bottom_k,
                            copy_mode=args.copy_mode,
                        )
                    )
                except Exception as exc:
                    export_failed_rows.append({"slide_id": slide_id, "scale": scale_name, "reason": str(exc)})

        pd.DataFrame(manifest_rows).to_csv(os.path.join(top_output_dir, "top_patches_manifest.csv"), index=False)
        if export_failed_rows:
            pd.DataFrame(export_failed_rows).to_csv(os.path.join(top_output_dir, "failed_exports.csv"), index=False)

    print(f"Saved outputs to: {args.results_dir}")
    print(f"Successful slides: {len(prediction_rows)}")
    print(f"Failed rows: {len(failed_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
