#!/usr/bin/env python3
"""Step 2.5 BiomedCLIP feature and preprocessing audit.

The full feature scan is opt-in because it reads roughly 3.2 million vectors.
The preprocessing A/B comparison is also opt-in because it loads BiomedCLIP
and performs GPU inference on WSI patches. Both modes are read-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


MODEL_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
MODEL_PATH = f"hf-hub:{MODEL_REPO}"
CURRENT_MEAN = (0.485, 0.456, 0.406)
CURRENT_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--inventory", type=Path, default=Path("analysis/stage2_yiyuan_data_audit/00_inventory/dataset_inventory.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis/stage2_yiyuan_data_audit/05_biomedclip_feature_audit"))
    p.add_argument("--feature-dir-5x", type=Path, default=Path("data/yiyuan/features_biomedclip_5x"))
    p.add_argument("--feature-dir-20x", type=Path, default=Path("data/yiyuan/features_biomedclip_20x"))
    p.add_argument("--coord-dir-5x", type=Path, default=Path("data/yiyuan/patches_coords_5x/patches_256"))
    p.add_argument("--coord-dir-20x", type=Path, default=Path("data/yiyuan/patches_coords_20x/patches_256"))
    p.add_argument("--wsi-dir", type=Path, default=Path("data/yiyuan/wsi"))
    p.add_argument("--feature-scan", action="store_true", help="Read all feature values in chunks")
    p.add_argument("--ab-compare", action="store_true", help="Run current-vs-official preprocessing inference")
    p.add_argument("--limit-slides", type=int, default=None)
    p.add_argument("--ab-slides", nargs="*", default=None)
    p.add_argument("--ab-patches-per-slide", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cache-dir", type=Path, default=None)
    return p.parse_args()


def root_path(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    cwd = Path.cwd().resolve()
    return cwd if (cwd / "dataset_csv" / "all_data.csv").is_file() else Path(__file__).resolve().parents[3]


def rooted(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as h:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(h)]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def stats(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"count": 0, "min": "", "p01": "", "p25": "", "median": "", "p75": "", "p99": "", "max": "", "mean": "", "sd": ""}
    return {"count": len(a), "min": float(a.min()), "p01": float(np.percentile(a, 1)), "p25": float(np.percentile(a, 25)), "median": float(np.median(a)), "p75": float(np.percentile(a, 75)), "p99": float(np.percentile(a, 99)), "max": float(a.max()), "mean": float(a.mean()), "sd": float(a.std())}


def feature_scan(path: Path, scan_values: bool) -> dict[str, object]:
    row: dict[str, object] = {"feature_path": str(path), "read_success": False, "error": "", "shape": "", "dtype": "", "count": "", "dim": "", "nan_count": "", "inf_count": "", "zero_vector_count": "", "norm_min": "", "norm_p01": "", "norm_p25": "", "norm_median": "", "norm_p75": "", "norm_p99": "", "norm_max": "", "norm_mean": "", "norm_sd": ""}
    try:
        with h5py.File(path, "r") as h:
            if "features" not in h:
                raise KeyError("missing features dataset")
            d = h["features"]; row.update({"read_success": True, "shape": json.dumps(list(d.shape), separators=(",", ":")), "dtype": str(d.dtype), "count": int(d.shape[0]) if d.ndim else 0, "dim": int(d.shape[1]) if d.ndim == 2 else ""})
            if scan_values and d.ndim == 2:
                nan = inf = zero = 0; norms=[]
                for start in range(0, d.shape[0], 4096):
                    block = np.asarray(d[start:start + 4096], dtype=np.float32)
                    nan += int(np.isnan(block).sum()); inf += int(np.isinf(block).sum())
                    n = np.linalg.norm(block.astype(np.float64), axis=1); zero += int(np.count_nonzero(n == 0)); norms.extend(n.tolist())
                ns = stats(norms); row.update({"nan_count": nan, "inf_count": inf, "zero_vector_count": zero, **{f"norm_{k}": v for k, v in ns.items() if k != "count"}})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def transform_repr(transform: object) -> str:
    return repr(transform).replace("\n", " ")


def image_ab(args: argparse.Namespace, root: Path, inventory: list[dict[str, str]], out: Path) -> list[dict[str, object]]:
    import torch
    from PIL import Image
    from torchvision import transforms

    sys.path.insert(0, str(root))
    from feature_extraction.patch_extraction_utils_biomedclip import (  # noqa: PLC0415
        _load_biomedclip_model,
        get_biomedclip_transforms,
    )
    model, official, _ = _load_biomedclip_model(MODEL_PATH, cache_dir=str(args.cache_dir) if args.cache_dir else None, load_tokenizer=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A/B inference requires CUDA; refusing CPU inference")
    model = model.to(device).eval(); current = get_biomedclip_transforms()
    selected = args.ab_slides or ([r["slide_id"] for r in inventory[:4]] if args.limit_slides else ["25032146B2", "2476358-B2", "2485803-B2", "2486859-B2"])
    by_id = {r["slide_id"]: r for r in inventory}; rows=[]
    for sid in selected:
        if sid not in by_id: continue
        inv = by_id[sid]; coord_path = rooted(root, Path(inv["coord_5x_path"])); wsi_path = rooted(root, args.wsi_dir) / f"{sid}.svs"
        if not wsi_path.is_file():
            matches = list(rooted(root,args.wsi_dir).glob(f"{sid}.*"));
            if len(matches)!=1: raise FileNotFoundError(f"WSI not found for {sid}")
            wsi_path=matches[0]
        import openslide
        with h5py.File(coord_path,"r") as h:
            coords=h["coords"][:]
            ds=h["coords"].attrs
            level=int(ds["patch_level"]); size=int(ds["patch_size"])
        with openslide.OpenSlide(str(wsi_path)) as slide:
            n=min(args.ab_patches_per_slide,len(coords)); images=[slide.read_region(tuple(map(int,coords[j])),level,(size,size)).convert("RGB") for j in range(n)]
        a=torch.stack([current(im) for im in images]).to(device); b=torch.stack([official(im) for im in images]).to(device)
        with torch.inference_mode(): fa=model.encode_image(a).float(); fb=model.encode_image(b).float()
        na=fa.norm(dim=1); nb=fb.norm(dim=1); cos=torch.nn.functional.cosine_similarity(fa,fb,dim=1); dist=(fa-fb).norm(dim=1)
        for j in range(n): rows.append({"slide_id":sid,"scale":"5x","patch_index":j,"x":int(coords[j,0]),"y":int(coords[j,1]),"embedding_cosine":float(cos[j].cpu()),"embedding_l2_distance":float(dist[j].cpu()),"current_norm":float(na[j].cpu()),"official_norm":float(nb[j].cpu()),"current_transform":transform_repr(current),"official_transform":transform_repr(official)})
    return rows


def main() -> int:
    a=parse_args(); root=root_path(a.project_root); out=rooted(root,a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/"figures").mkdir(exist_ok=True)
    inv=read_csv(rooted(root,a.inventory)); inv=inv[:a.limit_slides] if a.limit_slides else inv
    stats_rows=[]
    for scale, key in (("5x","feature_5x_path"),("20x","feature_20x_path")):
        for r in inv:
            row=feature_scan(rooted(root,Path(r[key])),a.feature_scan); row.update({"slide_id":r["slide_id"],"scale":scale}); stats_rows.append(row)
    fields=list(stats_rows[0]); write_csv(out/"feature_statistics.csv",fields,stats_rows)
    anomalies=[]
    for r in stats_rows:
        if not r["read_success"]: anomalies.append({"slide_id":r["slide_id"],"scale":r["scale"],"issue_type":"read_failure","severity":"critical","details":r["error"]}); continue
        if r["shape"] != "[" + str(r["count"]) + ",512]": anomalies.append({"slide_id":r["slide_id"],"scale":r["scale"],"issue_type":"shape_not_Nx512","severity":"critical","details":r["shape"]})
        if a.feature_scan and (int(r["nan_count"] or 0) or int(r["inf_count"] or 0)): anomalies.append({"slide_id":r["slide_id"],"scale":r["scale"],"issue_type":"nan_or_inf","severity":"critical","details":f"nan={r['nan_count']}, inf={r['inf_count']}"})
        if a.feature_scan and int(r["zero_vector_count"] or 0): anomalies.append({"slide_id":r["slide_id"],"scale":r["scale"],"issue_type":"zero_vector","severity":"warning","details":str(r["zero_vector_count"])})
        if r["dtype"] != "float32": anomalies.append({"slide_id":r["slide_id"],"scale":r["scale"],"issue_type":"dtype_not_float32","severity":"warning","details":r["dtype"]})
    write_csv(out/"feature_anomalies.csv",["slide_id","scale","issue_type","severity","details"],anomalies)
    ab=[]
    if a.ab_compare: ab=image_ab(a,root,inv,out)
    write_csv(out/"preprocessing_comparison.csv",["slide_id","scale","patch_index","x","y","embedding_cosine","embedding_l2_distance","current_norm","official_norm","current_transform","official_transform"],ab)
    total=len(stats_rows); good=sum(bool(r["read_success"]) and r["shape"]==f"[{r['count']},512]" for r in stats_rows); summary=["# Step 2.5: BiomedCLIP Feature and Preprocessing Audit","","## Code-path conclusion",f"- Checkpoint: `{MODEL_PATH}` (`{MODEL_REPO}`).","- Image path: `model.encode_image(batch)`.","- Current project preprocessing: `Resize((224,224))`, `ToTensor()`, ImageNet mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)`.","- The official preprocess returned by `create_model_from_pretrained()` is intentionally discarded by the current extractor.","- Current extractor does not L2-normalize image features. The feature H5 stores raw encoder outputs.","","## Scan status",f"- Feature H5 pairs inspected: {total}; shape-valid records: {good}.",f"- Full value scan requested: **{a.feature_scan}**; A/B inference requested: **{a.ab_compare}**.",f"- Anomaly records: {len(anomalies)}.","","## Interpretation", "- Official-vs-current preprocessing must be judged from `preprocessing_comparison.csv`; no full feature regeneration is performed by this audit.","- Existing feature vectors are suitable as a baseline only if the controlled A/B cosine differences are accepted and downstream consumers preserve the current checkpoint/preprocessing convention.","- Step 2.6 should wait for the full scan and A/B sample result if either was not run.","","## Outputs", "- `feature_statistics.csv`, `feature_anomalies.csv`, `preprocessing_comparison.csv`, `summary.md`, and `figures/`."]
    (out/"summary.md").write_text("\n".join(summary)+"\n",encoding="utf-8"); print(f"wrote {out}; feature_pairs={total}; anomalies={len(anomalies)}; ab_rows={len(ab)}"); return 0


if __name__ == "__main__": raise SystemExit(main())
