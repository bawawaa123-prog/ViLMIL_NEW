#!/usr/bin/env python3
"""Step 2.4 Yiyuan low/high coordinate bbox mapping audit.

All coordinates are interpreted in level-0 pixels. The script never reads WSI
pixels or feature matrices. Mapping is based on actual continuous patch
footprints derived from each scale's level downsample and patch size.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--inventory", type=Path, default=Path("analysis/stage2_yiyuan_data_audit/00_inventory/dataset_inventory.csv"))
    p.add_argument("--physical-fov", type=Path, default=Path("analysis/stage2_yiyuan_data_audit/02_wsi_metadata/physical_fov.csv"))
    p.add_argument("--wsi-metadata", type=Path, default=Path("analysis/stage2_yiyuan_data_audit/02_wsi_metadata/wsi_metadata.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis/stage2_yiyuan_data_audit/04_cross_scale_mapping"))
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--figure-slides", nargs="*", default=None)
    p.add_argument("--max-figure-slides", type=int, default=6)
    p.add_argument("--limit-slides", type=int, default=None, help="Small test mode; do not use for final audit")
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


def f(value: object) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"non-finite value: {value}")
    return x


def i(value: object) -> int:
    return int(float(value))


def write_csv(path: Path, fields: list[str], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_coords(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h:
        if "coords" not in h:
            raise KeyError(f"missing coords: {path}")
        a = h["coords"][:]
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"bad coordinate shape {a.shape}: {path}")
    return a.astype(np.float64, copy=False)


def footprint(coords: np.ndarray, width: float, height: float) -> np.ndarray:
    out = np.empty((len(coords), 4), dtype=np.float64)
    out[:, 0:2] = coords
    out[:, 2] = coords[:, 0] + width
    out[:, 3] = coords[:, 1] + height
    return out


def overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0
    area = w * h
    return area, area / ((a[2] - a[0]) * (a[3] - a[1])), area / ((b[2] - b[0]) * (b[3] - b[1]))


def candidate_index(high: np.ndarray, bin_w: float, bin_h: float) -> dict[tuple[int, int], list[int]]:
    index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for j, box in enumerate(high):
        x0 = math.floor(box[0] / bin_w); x1 = math.floor(np.nextafter(box[2], -np.inf) / bin_w)
        y0 = math.floor(box[1] / bin_h); y1 = math.floor(np.nextafter(box[3], -np.inf) / bin_h)
        for bx in range(x0, x1 + 1):
            for by in range(y0, y1 + 1):
                index[(bx, by)].append(j)
    return index


def padding_ratio(box: np.ndarray, wsi_w: float, wsi_h: float) -> float:
    inside_w = max(0.0, min(box[2], wsi_w) - max(box[0], 0.0))
    inside_h = max(0.0, min(box[3], wsi_h) - max(box[1], 0.0))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return max(0.0, min(1.0, 1.0 - (inside_w * inside_h / area))) if area > 0 else 0.0


def bins(ratio: float) -> str:
    if ratio <= 0.05: return "<=5%"
    if ratio <= 0.10: return "5-10%"
    if ratio <= 0.25: return "10-25%"
    if ratio <= 0.50: return "25-50%"
    return ">50%"


def wsi_dimensions(meta: dict[str, str]) -> tuple[float, float]:
    raw = json.loads(meta["level_dimensions"])
    return float(raw[0][0]), float(raw[0][1])


def choose_figures(rows: list[dict[str, object]], requested: list[str] | None, limit: int) -> list[str]:
    ids = [str(x) for x in (requested or [])]
    if not ids:
        ids = ["2486859-B2"]
        ordered = sorted(rows, key=lambda x: int(x["low_count"]))
        ids += [str(ordered[0]["slide_id"]), str(ordered[len(ordered)//2]["slide_id"]), str(ordered[-1]["slide_id"])]
    seen = []
    for sid in ids:
        if sid not in seen: seen.append(sid)
    return seen[:limit]


def make_figure(path: Path, slide_id: str, low: np.ndarray, high: np.ndarray, mapping: list[tuple[int, int]], low_w: float, low_h: float, high_w: float, high_h: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    if len(high):
        ax.scatter((high[:, 0] + high[:, 2]) / 2, (high[:, 1] + high[:, 3]) / 2, s=3, c="#20a4d8", alpha=.45, label="high ~10x centers")
    if len(low):
        ax.scatter((low[:, 0] + low[:, 2]) / 2, (low[:, 1] + low[:, 3]) / 2, s=14, facecolors="none", edgecolors="#d94841", label="low ~2.5x centers")
    for li, hi in mapping[: min(len(mapping), 400)]:
        ax.plot([low[li, 0] + low_w / 2, high[hi, 0] + high_w / 2], [low[li, 1] + low_h / 2, high[hi, 1] + high_h / 2], color="#777", alpha=.06, linewidth=.3)
    ax.set_title(f"{slide_id}: low/high bbox mapping")
    ax.set_aspect("equal", adjustable="box"); ax.invert_yaxis(); ax.legend(loc="best", markerscale=2)
    ax.set_xlabel("level-0 x (px)"); ax.set_ylabel("level-0 y (px)"); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def main() -> int:
    a = args(); root = root_path(a.project_root); out = rooted(root, a.output_dir); out.mkdir(parents=True, exist_ok=True); figdir = out / "figures"; figdir.mkdir(exist_ok=True)
    inv = read_csv(rooted(root, a.inventory)); fov_rows = read_csv(rooted(root, a.physical_fov)); fov = {(r["slide_id"], r["scale"]): r for r in fov_rows}
    wsi_meta = {r["slide_id"]: r for r in read_csv(rooted(root, a.wsi_metadata))}
    if a.limit_slides: inv = inv[:a.limit_slides]
    low_stats=[]; high_stats=[]; relation_rows=[]; unmapped=[]; anomaly_rows=[]; padding_rows=[]; all_maps={}; all_arrays={}
    low_fields=["slide_id","low_index","low_x","low_y","low_x2","low_y2","high_child_count","high_center_inside_count","high_fully_inside_count","high_overlap_count","high_overlap_area_sum","high_child_indices"]
    high_fields=["slide_id","high_index","high_x","high_y","high_x2","high_y2","parent_count","parent_indices","best_parent_index","best_overlap_area","high_center_inside_parent_count","high_fully_inside_parent_count"]
    with (out/"low_to_high_mapping.csv").open("w",encoding="utf-8",newline="") as map_h, (out/"high_to_low_statistics.csv").open("w",encoding="utf-8",newline="") as high_h:
        map_w=csv.DictWriter(map_h,fieldnames=low_fields); map_w.writeheader(); high_wrt=csv.DictWriter(high_h,fieldnames=high_fields); high_wrt.writeheader()
        for n,row in enumerate(inv,1):
            sid=row["slide_id"]; lf=fov[(sid,"5x")]; hf=fov[(sid,"20x")]
            lw,lh=f(lf["level0_footprint_width_px"]),f(lf["level0_footprint_height_px"])
            hw,hh=f(hf["level0_footprint_width_px"]),f(hf["level0_footprint_height_px"])
            low=load_coords(rooted(root,Path(row["coord_5x_path"]))); high=load_coords(rooted(root,Path(row["coord_20x_path"])))
            lb=footprint(low,lw,lh); hb=footprint(high,hw,hh); idx=candidate_index(hb,lw,lh)
            high_parents=[[] for _ in range(len(high))]; low_map=[]; center_inside=np.zeros(len(low),int); full_inside=np.zeros(len(low),int); overlap_count=np.zeros(len(low),int); overlap_area=np.zeros(len(low),float)
            for li,box in enumerate(lb):
                bx0=math.floor(box[0]/lw); bx1=math.floor(np.nextafter(box[2],-np.inf)/lw); by0=math.floor(box[1]/lh); by1=math.floor(np.nextafter(box[3],-np.inf)/lh)
                cand=sorted({j for bx in range(bx0,bx1+1) for by in range(by0,by1+1) for j in idx.get((bx,by),[])})
                children=[]; centers=[]; fulls=[]
                for j in cand:
                    area,_,_=overlap(box,hb[j]);
                    if area<=0: continue
                    children.append(j); overlap_count[li]+=1; overlap_area[li]+=area; high_parents[j].append(li)
                    centers.append(box[0] <= (hb[j,0]+hb[j,2])/2 <= box[2] and box[1] <= (hb[j,1]+hb[j,3])/2 <= box[3])
                    fulls.append(hb[j,0]>=box[0] and hb[j,1]>=box[1] and hb[j,2]<=box[2] and hb[j,3]<=box[3])
                center_inside[li]=sum(centers); full_inside[li]=sum(fulls); low_map.append(children)
                map_w.writerow({"slide_id":sid,"low_index":li,"low_x":low[li,0],"low_y":low[li,1],"low_x2":box[2],"low_y2":box[3],"high_child_count":len(children),"high_center_inside_count":int(center_inside[li]),"high_fully_inside_count":int(full_inside[li]),"high_overlap_count":int(overlap_count[li]),"high_overlap_area_sum":overlap_area[li],"high_child_indices":";".join(map(str,children))})
            for hi,hbox in enumerate(hb):
                parents=high_parents[hi]; parent_overlap=[]
                for li in parents: parent_overlap.append(overlap(lb[li],hbox)[0])
                best=int(np.argmax(parent_overlap)) if parent_overlap else -1
                high_wrt.writerow({"slide_id":sid,"high_index":hi,"high_x":high[hi,0],"high_y":high[hi,1],"high_x2":hbox[2],"high_y2":hbox[3],"parent_count":len(parents),"parent_indices":";".join(map(str,parents)),"best_parent_index":best,"best_overlap_area":max(parent_overlap) if parent_overlap else 0.0,"high_center_inside_parent_count":sum(lb[li,0] <= (hbox[0]+hbox[2])/2 <= lb[li,2] and lb[li,1] <= (hbox[1]+hbox[3])/2 <= lb[li,3] for li in parents),"high_fully_inside_parent_count":sum(hbox[0]>=lb[li,0] and hbox[1]>=lb[li,1] and hbox[2]<=lb[li,2] and hbox[3]<=lb[li,3] for li in parents)})
            child_counts=np.array([len(x) for x in low_map]); parent_counts=np.array([len(x) for x in high_parents]); low_with=child_counts>0; high_with=parent_counts>0
            low_stats.append({"slide_id":sid,"low_count":len(low),"high_count":len(high),"low_with_child_count":int(low_with.sum()),"low_with_child_ratio":float(low_with.mean()) if len(low) else 0,"low_no_child_count":int((~low_with).sum()),"low_child_count_mean":float(child_counts.mean()) if len(low) else 0,"low_child_count_median":float(np.median(child_counts)) if len(low) else 0,"low_child_count_min":int(child_counts.min()) if len(low) else 0,"low_child_count_max":int(child_counts.max()) if len(low) else 0,"low_16_child_count":int((child_counts==16).sum()),"low_16_child_ratio":float((child_counts==16).mean()) if len(low) else 0,"high_with_parent_count":int(high_with.sum()),"high_with_parent_ratio":float(high_with.mean()) if len(high) else 0,"high_no_parent_count":int((~high_with).sum()),"high_multi_parent_count":int((parent_counts>1).sum()),"high_parent_count_max":int(parent_counts.max()) if len(high) else 0,"high_center_inside_total":int(center_inside.sum()),"high_fully_inside_total":int(full_inside.sum()),"high_overlap_total":int(overlap_count.sum()),"low_w":lw,"low_h":lh,"high_w":hw,"high_h":hh,"fov_ratio_x":lw/hw,"fov_ratio_y":lh/hh})
            all_maps[sid]=[(li,hi) for li,cs in enumerate(low_map) for hi in cs]; all_arrays[sid]=(low,hb,lb,high)
            for scale,boxes in (("5x",lb),("20x",hb)):
                wsi_w, wsi_h = wsi_dimensions(wsi_meta[sid])
                counts=Counter(bins(padding_ratio(box, wsi_w, wsi_h)) for box in boxes)
                for bucket,count in counts.items(): padding_rows.append({"slide_id":sid,"scale":scale,"patch_count":len(boxes),"padding_bucket":bucket,"count":count,"fraction":count/len(boxes) if len(boxes) else 0})
            if len(low)==0 or len(high)==0: anomaly_rows.append({"slide_id":sid,"issue_type":"empty_scale","severity":"critical","details":f"low={len(low)}, high={len(high)}"})
            if len(low) and (child_counts==16).mean()<0.25: anomaly_rows.append({"slide_id":sid,"issue_type":"low_16_child_ratio_low","severity":"info","details":f"ratio={(child_counts==16).mean():.4f}"})
            if len(high) and high_with.mean()<0.9: anomaly_rows.append({"slide_id":sid,"issue_type":"high_parent_coverage_low","severity":"warning","details":f"ratio={high_with.mean():.4f}"})
            if n==1 or n%a.progress_every==0 or n==len(inv): print(f"audited {n}/{len(inv)} slides",flush=True)
    stats_fields=list(low_stats[0]); write_csv(out/"mapping_statistics.csv",stats_fields,low_stats); write_csv(out/"padding_severity.csv",["slide_id","scale","patch_count","padding_bucket","count","fraction"],padding_rows); write_csv(out/"mapping_anomalies.csv",["slide_id","issue_type","severity","details"],anomaly_rows)
    # Unmapped regions are a compact per-slide record, while detailed relation files carry indices.
    for r in low_stats: unmapped.append({"slide_id":r["slide_id"],"low_no_child_count":r["low_no_child_count"],"high_no_parent_count":r["high_no_parent_count"],"low_count":r["low_count"],"high_count":r["high_count"]})
    write_csv(out/"unmapped_regions.csv",["slide_id","low_no_child_count","high_no_parent_count","low_count","high_count"],unmapped)
    for sid in choose_figures(low_stats,a.figure_slides,a.max_figure_slides):
        if sid in all_arrays:
            low, hb, lb, high = all_arrays[sid]; r=next(x for x in low_stats if x["slide_id"]==sid); make_figure(figdir/f"mapping_{sid}.png",sid,lb,hb,all_maps[sid],r["low_w"],r["low_h"],r["high_w"],r["high_h"])
    critical=sum(x["severity"]=="critical" for x in anomaly_rows); warning=sum(x["severity"]=="warning" for x in anomaly_rows); info=sum(x["severity"]=="info" for x in anomaly_rows)
    child=np.array([int(r["low_16_child_ratio"]>0) for r in low_stats]); summary=["# Step 2.4: Yiyuan Cross-scale Spatial Mapping Audit","","## Scope and method",f"- Population: {len(inv)} slides; mapping uses level-0 continuous bboxes from actual downsample and patch size.","- Nominal 5x is treated as actual ~2.5x; nominal 20x as actual ~10x.","- Mapping is coordinate-value based; no H5 row-order assumption is used.","- A relation is an overlap with positive area. Child/parent counts additionally report high-center-inside and high-fully-inside conditions.","- No WSI pixels or feature matrices are read.","","## Results"]
    for key,label in (("low_with_child_ratio","low patches with >=1 high overlap"),("high_with_parent_ratio","high patches with >=1 low overlap"),("low_16_child_ratio","low patches with exactly 16 high overlaps")):
        vals=np.array([float(r[key]) for r in low_stats]); summary.append(f"- {label}: median={np.median(vals):.4f}, min={np.min(vals):.4f}, max={np.max(vals):.4f}.")
    child_counts=np.array([int(r["low_child_count_median"]) for r in low_stats]); total_low=sum(int(r["low_count"]) for r in low_stats); total_high=sum(int(r["high_count"]) for r in low_stats)
    total_low_child=sum(int(r["low_with_child_count"]) for r in low_stats); total_high_parent=sum(int(r["high_with_parent_count"]) for r in low_stats); total_no_low=sum(int(r["low_no_child_count"]) for r in low_stats); total_no_high=sum(int(r["high_no_parent_count"]) for r in low_stats)
    total_exact16=sum(int(r["low_16_child_count"]) for r in low_stats); total_overlap=sum(int(r["high_overlap_total"]) for r in low_stats); total_center=sum(int(r["high_center_inside_total"]) for r in low_stats); total_full=sum(int(r["high_fully_inside_total"]) for r in low_stats)
    summary += [f"- Mean per-slide median low child count: {np.mean(child_counts):.3f}; exact-16 overlap ratio across all low patches: {total_exact16/total_low:.4f}.",f"- Low patches with >=1 overlapping high: {total_low_child:,}/{total_low:,} ({total_low_child/total_low:.4f}); high patches with >=1 low parent: {total_high_parent:,}/{total_high:,} ({total_high_parent/total_high:.4f}).",f"- High-center-inside relations: {total_center:,}; high-fully-inside relations: {total_full:,}; positive-area overlap relations: {total_overlap:,}.",f"- Unmapped: low without child={total_no_low:,}; high without parent={total_no_high:,}.",f"- Mapping anomaly records: critical={critical}, warning={warning}, info={info}.","- The full relationship CSV contains one row per low patch and its exact overlapping high indices; the reverse CSV contains one row per high patch and parent indices.","","## Padding severity"]
    for scale in ("5x","20x"):
        vals=[r for r in padding_rows if r["scale"]==scale]; total=sum(int(r["count"]) for r in vals); summary.append(f"- {scale}: total patches={total:,}; buckets are in `padding_severity.csv`.")
    stable = total_low_child / total_low >= 0.95 and total_high_parent / total_high >= 0.90
    passed = critical == 0 and stable
    summary += ["","## Required answers",f"","- Stable low/high spatial correspondence: **{'yes' if stable else 'no'}** under the operational thresholds low-with-child >=95% and high-with-parent >=90%.",f"- Rule-like 16-child structure: **{'yes' if total_exact16/total_low >= 0.50 else 'no'}** for strict positive-overlap count=16; exact-16 is not required for valid coordinate mapping because bbox boundary offsets can create neighboring overlaps.","- Spatial shift: no global shift is inferred unless per-slide anomalies identify one; coordinate bboxes are compared directly.","- Coordinate/patch regeneration: **not indicated by mapping alone; use offline mapping unless critical anomalies are present**.",f"- Step 2.4 pass: **{'True' if passed else 'False'}** (critical anomalies={critical}; stable correspondence={stable}).","","## Runtime and outputs",f"- Runtime: Python {platform.python_version()}, NumPy {np.__version__}.","- Outputs: `mapping_statistics.csv`, `low_to_high_mapping.csv`, `high_to_low_statistics.csv`, `unmapped_regions.csv`, `padding_severity.csv`, `mapping_anomalies.csv`, and `figures/`."]
    (out/"summary.md").write_text("\n".join(summary)+"\n",encoding="utf-8"); print(f"wrote {out}; critical={critical} warning={warning} info={info}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
