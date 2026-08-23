"""Aggregate E1 routing replication against existing E0/Dual controls."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PILOT = ROOT / "analysis/stage3_model_design/03_low_context_routing/03_pilot_validation"
E0 = PILOT / "results/E0/stage332_pilot_E0_high_only_s1/fold_summary.csv"
E1_0 = PILOT / "results/E1/stage332_pilot_E1_low_context_routing_s1/fold_summary.csv"
STAGE2_E0 = ROOT / "analysis/stage2_yiyuan_data_audit/06_scale_ablation/high_only/adenocarcinoma_biomedclip_high_only_strict5_s1/fold_summary.csv"
DUAL = ROOT / "trained_models/adenocarcinoma_strict5_new/adenocarcinoma_biomedclip_dual_strict5_new_s1/fold_summary.csv"


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def locate_fold_summary(base: Path, fold: int):
    matches = sorted(base.glob(f"**/stage332_multifold_E1_fold{fold}_s1/fold_summary.csv"))
    return matches[0] if matches else None


def fold_dir_from_summary(path: Path | None):
    return path.parent if path else None


def training_metadata(summary_path: Path | None, routing: bool):
    result = {"stop_epoch": None, "runtime_seconds": None,
              "routing_diagnostics_finite": None, "routing_nonzero_gradient": None}
    folder = fold_dir_from_summary(summary_path)
    if not folder:
        return result
    epochs = read_rows(folder / "epoch_details.csv")
    if epochs:
        result["stop_epoch"] = max(int(float(row["epoch"])) for row in epochs)
        result["runtime_seconds"] = sum(float(row["duration_seconds"]) for row in epochs)
    if routing:
        diag_path = folder / "routing_diagnostics.jsonl"
        if diag_path.exists():
            rows = [json.loads(line) for line in diag_path.read_text().splitlines() if line.strip()]
            numeric = [v for row in rows for v in row.values()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)]
            result["routing_diagnostics_finite"] = bool(rows) and all(math.isfinite(float(v)) for v in numeric)
            result["routing_nonzero_gradient"] = any(
                row.get("context_projection_grad_norm", 0) > 0 and
                row.get("route_score_grad_norm", 0) > 0 for row in rows)
    return result


def num(row, key):
    return float(row[key])


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(var)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "aggregate"))
    parser.add_argument("--e1-root", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    e1 = {}
    for fold, path in [(0, E1_0)]:
        rows = read_rows(path)
        if rows:
            e1[fold] = rows[0]
    for fold in range(1, 5):
        path = locate_fold_summary(Path(args.e1_root), fold)
        rows = read_rows(path) if path else []
        if rows:
            e1[fold] = rows[0]

    controls = {"E0_Stage2_HighOnly": read_rows(STAGE2_E0),
                "Dual_Stage1": read_rows(DUAL)}
    # Prefer the already completed pilot E0 for fold 0; assert it matches the
    # Stage 2 high-only row numerically before paired comparison.
    pilot_e0 = read_rows(E0)
    if pilot_e0 and controls["E0_Stage2_HighOnly"]:
        for key in ("test_auc", "test_acc", "test_f1", "val_auc", "val_acc"):
            if abs(num(pilot_e0[0], key) - num(controls["E0_Stage2_HighOnly"][0], key)) > 1e-7:
                raise AssertionError(f"fold 0 E0 mismatch for {key}")

    rows_out = []
    for fold in range(5):
        if fold not in e1:
            continue
        e1r = e1[fold]
        e0r = controls["E0_Stage2_HighOnly"][fold] if fold < len(controls["E0_Stage2_HighOnly"]) else {}
        dualr = controls["Dual_Stage1"][fold] if fold < len(controls["Dual_Stage1"]) else {}
        e1_summary_path = E1_0 if fold == 0 else locate_fold_summary(Path(args.e1_root), fold)
        metadata = training_metadata(e1_summary_path, routing=True)
        row = {"fold": fold, "e1_source": "pilot" if fold == 0 else "multifold", **metadata}
        for key in ("test_auc", "test_acc", "test_f1", "val_auc", "val_acc"):
            row[f"e1_{key}"] = num(e1r, key)
            if e0r:
                row[f"e0_{key}"] = num(e0r, key)
                row[f"e1_minus_e0_{key}"] = row[f"e1_{key}"] - row[f"e0_{key}"]
            if dualr:
                row[f"dual_{key}"] = num(dualr, key)
        rows_out.append(row)

    with (out / "paired_fold_results.csv").open("w", newline="") as f:
        if rows_out:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            writer.writeheader(); writer.writerows(rows_out)

    report = {"available_e1_folds": [r["fold"] for r in rows_out],
              "missing_e1_folds": [f for f in range(5) if f not in e1],
              "controls": {k: str(v) for k, v in controls.items()}, "metrics": {}}
    for metric in ("test_auc", "test_acc", "test_f1"):
        vals = [r[f"e1_{metric}"] for r in rows_out]
        diffs = [r[f"e1_minus_e0_{metric}"] for r in rows_out if f"e1_minus_e0_{metric}" in r]
        e0vals = [num(controls["E0_Stage2_HighOnly"][i], metric) for i in range(min(5, len(controls["E0_Stage2_HighOnly"]))) ]
        em, es = mean_std(vals); bm, bs = mean_std(e0vals); dm, ds = mean_std(diffs)
        report["metrics"][metric] = {
            "e1_mean": em, "e1_std": es, "e0_mean": bm, "e0_std": bs,
            "mean_delta": dm, "delta_std": ds,
            "improved_folds": sum(d > 0 for d in diffs),
            "max_degradation": min(diffs) if diffs else None,
        }
    (out / "paired_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
