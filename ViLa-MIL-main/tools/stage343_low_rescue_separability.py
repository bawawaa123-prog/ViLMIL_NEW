#!/usr/bin/env python3
"""Stage 3.4.3 validation-only Low-rescue separability diagnostic.

Consumes Stage 3.4.2 prediction CSVs only. No model, checkpoint, or feature
inference is performed. Rules are one-dimensional margin thresholds selected
on one fold and applied unchanged to the other fold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


EXPECTED = {
    0: {"n": 194, "disagreement": 25, "rescue": 4, "harmful": 21},
    1: {"n": 194, "disagreement": 24, "rescue": 5, "harmful": 19},
}
FEATURES = ("high_margin", "low_margin", "margin_delta", "log_margin_ratio")
EPS = 1e-8


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return {
        "auc": float(roc_auc_score(labels, probs[:, 1])) if len(np.unique(labels)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def _load_fold(path: Path, fold: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "slide_id", "label", "prediction_low", "prediction_high",
        "low_logit_0", "low_logit_1", "high_logit_0", "high_logit_1",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AssertionError(f"fold {fold}: missing required columns: {missing}")
    expected = EXPECTED[fold]
    if len(frame) != expected["n"]:
        raise AssertionError(f"fold {fold}: expected {expected['n']} rows, got {len(frame)}")
    if frame["slide_id"].duplicated().any():
        raise AssertionError(f"fold {fold}: slide_id is not unique")
    if frame["label"].isna().any():
        raise AssertionError(f"fold {fold}: label contains missing values")
    if not frame["label"].isin([0, 1]).all():
        raise AssertionError(f"fold {fold}: labels are not binary 0/1")
    if not np.array_equal(frame["prediction_low"].to_numpy(), frame[["low_logit_0", "low_logit_1"]].to_numpy().argmax(axis=1)):
        raise AssertionError(f"fold {fold}: prediction_low does not match logits")
    if not np.array_equal(frame["prediction_high"].to_numpy(), frame[["high_logit_0", "high_logit_1"]].to_numpy().argmax(axis=1)):
        raise AssertionError(f"fold {fold}: prediction_high does not match logits")

    frame = frame.sort_values("slide_id").reset_index(drop=True).copy()
    disagreement = frame.prediction_low != frame.prediction_high
    rescue = disagreement & (frame.prediction_high != frame.label) & (frame.prediction_low == frame.label)
    harmful = disagreement & (frame.prediction_high == frame.label) & (frame.prediction_low != frame.label)
    if int(disagreement.sum()) != expected["disagreement"]:
        raise AssertionError(f"fold {fold}: expected {expected['disagreement']} disagreements, got {int(disagreement.sum())}")
    if int(rescue.sum()) != expected["rescue"] or int(harmful.sum()) != expected["harmful"]:
        raise AssertionError(
            f"fold {fold}: expected rescue/harmful={expected['rescue']}/{expected['harmful']}, "
            f"got {int(rescue.sum())}/{int(harmful.sum())}"
        )

    high_margin = (frame.high_logit_1 - frame.high_logit_0).abs()
    low_margin = (frame.low_logit_1 - frame.low_logit_0).abs()
    frame["high_margin"] = high_margin
    frame["low_margin"] = low_margin
    frame["margin_delta"] = low_margin - high_margin
    frame["log_margin_ratio"] = np.log((low_margin + EPS) / (high_margin + EPS))
    frame["disagreement"] = disagreement
    frame["case_type"] = np.where(rescue, "rescue", np.where(harmful, "harmful", "non_disagreement"))
    frame["true_class"] = frame["label"].astype(int)
    return frame


def _class_counts(frame: pd.DataFrame, prefix: str) -> dict[str, int]:
    counts = frame.true_class.value_counts().to_dict()
    return {
        f"{prefix}_n": int(len(frame)),
        f"{prefix}_class_0": int(counts.get(0, 0)),
        f"{prefix}_class_1": int(counts.get(1, 0)),
    }


def _disagreement_cases(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fold, frame in frames.items():
        subset = frame[frame.disagreement].copy()
        keep = [
            "slide_id", "true_class", "label", "prediction_low", "prediction_high",
            "low_logit_0", "low_logit_1", "high_logit_0", "high_logit_1",
            "high_margin", "low_margin", "margin_delta", "log_margin_ratio", "case_type",
        ]
        subset.insert(0, "fold", fold)
        rows.append(subset[["fold"] + keep])
    return pd.concat(rows, ignore_index=True)


def _feature_summary(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fold, frame in frames.items():
        for subset_name, subset in (
            ("all_disagreement", frame[frame.disagreement]),
            ("rescue", frame[frame.case_type == "rescue"]),
            ("harmful", frame[frame.case_type == "harmful"]),
        ):
            for feature in FEATURES:
                values = subset[feature].to_numpy(dtype=float)
                rows.append({
                    "fold": fold, "subset": subset_name, "feature": feature, "n": len(values),
                    "mean": float(values.mean()) if len(values) else float("nan"),
                    "std": float(values.std()) if len(values) else float("nan"),
                    "median": float(np.median(values)) if len(values) else float("nan"),
                    "min": float(values.min()) if len(values) else float("nan"),
                    "max": float(values.max()) if len(values) else float("nan"),
                })
    return pd.DataFrame(rows)


def _feature_separability(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for fold, frame in frames.items():
        subset = frame[frame.disagreement]
        rescue = subset.case_type.to_numpy() == "rescue"
        harmful = subset.case_type.to_numpy() == "harmful"
        for feature in FEATURES:
            values = subset[feature].to_numpy(dtype=float)
            rescue_values, harmful_values = values[rescue], values[harmful]
            # AUC is a descriptive separability statistic on disagreement cases,
            # not a deployed classifier or a threshold selected for transfer.
            raw_auc = float(roc_auc_score(rescue.astype(int), values))
            orientation = "ge" if raw_auc >= 0.5 else "le"
            rows.append({
                "fold": fold, "feature": feature,
                "rescue_n": int(rescue.sum()), "harmful_n": int(harmful.sum()),
                "rescue_class_0": int((subset.loc[rescue, "true_class"] == 0).sum()),
                "rescue_class_1": int((subset.loc[rescue, "true_class"] == 1).sum()),
                "harmful_class_0": int((subset.loc[harmful, "true_class"] == 0).sum()),
                "harmful_class_1": int((subset.loc[harmful, "true_class"] == 1).sum()),
                "rescue_mean": float(rescue_values.mean()), "harmful_mean": float(harmful_values.mean()),
                "rescue_median": float(np.median(rescue_values)), "harmful_median": float(np.median(harmful_values)),
                "mean_difference_rescue_minus_harmful": float(rescue_values.mean() - harmful_values.mean()),
                "raw_rescue_auc": raw_auc,
                "best_orientation": orientation,
                "separability_auc": max(raw_auc, 1.0 - raw_auc),
            })
    return pd.DataFrame(rows)


def _thresholds(values: np.ndarray) -> list[float]:
    unique = np.unique(values[np.isfinite(values)])
    if len(unique) == 0:
        return []
    mids = (unique[:-1] + unique[1:]) / 2.0
    candidates = np.concatenate(([np.nextafter(unique[0], -np.inf)], unique, mids, [np.nextafter(unique[-1], np.inf)]))
    return [float(value) for value in np.unique(candidates)]


def _apply_rule(
    frame: pd.DataFrame,
    feature: str,
    orientation: str,
    threshold: float,
    rule_type: str = "threshold",
) -> np.ndarray:
    if rule_type == "always_high":
        return np.zeros(len(frame), dtype=bool)
    if rule_type != "threshold":
        raise ValueError(f"unsupported rule_type={rule_type!r}")
    disagreement = frame.disagreement.to_numpy(dtype=bool)
    values = frame[feature].to_numpy(dtype=float)
    trigger = values >= threshold if orientation == "ge" else values <= threshold
    return disagreement & trigger


def _rule_stats(frame: pd.DataFrame, trigger: np.ndarray) -> dict[str, float | int]:
    labels = frame.label.to_numpy(dtype=int)
    low_pred = frame.prediction_low.to_numpy(dtype=int)
    high_pred = frame.prediction_high.to_numpy(dtype=int)
    high_correct = high_pred == labels
    low_correct = low_pred == labels
    rescue = trigger & ~high_correct & low_correct
    harmful = trigger & high_correct & ~low_correct
    switched = int(trigger.sum())
    rescue_count, harmful_count = int(rescue.sum()), int(harmful.sum())
    return {
        "switched_count": switched,
        "rescue_count": rescue_count,
        "harm_count": harmful_count,
        "net_gain": rescue_count - harmful_count,
        "switch_precision": float(rescue_count / switched) if switched else 0.0,
    }


def _select_rule(frame: pd.DataFrame) -> dict:
    candidates = []
    feature_priority = {feature: index for index, feature in enumerate(FEATURES)}
    for feature in FEATURES:
        for orientation in ("ge", "le"):
            for threshold in _thresholds(frame.loc[frame.disagreement, feature].to_numpy(dtype=float)):
                stats = _rule_stats(frame, _apply_rule(frame, feature, orientation, threshold))
                candidates.append({
                    "rule_type": "threshold", "feature": feature, "orientation": orientation, "threshold": threshold,
                    **stats,
                })
    if not candidates:
        raise AssertionError("no threshold candidates generated")
    # Primary objective is explicitly net_gain. Remaining terms only make ties
    # deterministic and favor a precise, sparse switch rule.
    best = max(candidates, key=lambda row: (
        row["net_gain"], row["rescue_count"], -row["harm_count"],
        row["switch_precision"], -row["switched_count"],
        -feature_priority[row["feature"]], 1 if row["orientation"] == "ge" else 0,
    ))
    if best["switched_count"] == 0:
        # A zero-switch source optimum is a policy, not an extrapolatable
        # threshold outside the observed source-fold distribution.
        return {
            "rule_type": "always_high", "feature": "", "orientation": "none", "threshold": float("nan"),
            "switched_count": 0, "rescue_count": 0, "harm_count": 0,
            "net_gain": 0, "switch_precision": 0.0,
        }
    return best


def _evaluate_rule(frame: pd.DataFrame, rule: dict) -> dict:
    trigger = _apply_rule(
        frame, rule["feature"], rule["orientation"], rule["threshold"], rule.get("rule_type", "threshold")
    )
    current_stats = _rule_stats(frame, trigger)
    labels = frame.label.to_numpy(dtype=int)
    high_logits = frame[["high_logit_0", "high_logit_1"]].to_numpy(dtype=float)
    low_logits = frame[["low_logit_0", "low_logit_1"]].to_numpy(dtype=float)
    final_logits = high_logits.copy()
    final_logits[trigger] = low_logits[trigger]
    high_metrics = _metrics(labels, high_logits)
    final_metrics = _metrics(labels, final_logits)
    return {
        **rule,
        **current_stats,
        "high_auc": high_metrics["auc"], "high_accuracy": high_metrics["accuracy"], "high_macro_f1": high_metrics["macro_f1"],
        "selective_auc": final_metrics["auc"], "selective_accuracy": final_metrics["accuracy"], "selective_macro_f1": final_metrics["macro_f1"],
        "delta_auc": final_metrics["auc"] - high_metrics["auc"],
        "delta_accuracy": final_metrics["accuracy"] - high_metrics["accuracy"],
        "delta_macro_f1": final_metrics["macro_f1"] - high_metrics["macro_f1"],
    }


def _class_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, current in frame.groupby("fold", sort=True):
        for subset_name, subset in (
            ("all_disagreement", current),
            ("rescue", current[current.case_type == "rescue"]),
            ("harmful", current[current.case_type == "harmful"]),
        ):
            row = {"fold": int(fold), "subset": subset_name}
            row.update(_class_counts(subset, "samples"))
            rows.append(row)
    return pd.DataFrame(rows)


def _write_summary(output: Path, frames: dict[int, pd.DataFrame], transfers: pd.DataFrame, within: pd.DataFrame):
    lines = [
        "# Stage 3.4.3 Low-rescue separability / confidence-aware late-fusion feasibility",
        "",
        "This is a validation-only diagnostic over the existing Stage 3.4.2 prediction CSVs. No model inference, retraining, learned gate, calibration, or test evaluation was performed.",
        "",
        "## Input checks",
        "",
        "Both folds passed the required 194-slide, unique-ID, non-missing-label checks and matched the Stage 3.4.2 disagreement counts (Fold 0: 25; Fold 1: 24). Rescue/harmful counts were Fold 0: 4/21 and Fold 1: 5/19.",
        "",
        "## Margin separability",
        "",
    ]
    sep = pd.read_csv(output / "feature_separability.csv")
    for feature in FEATURES:
        rows = sep[sep.feature == feature]
        lines.append(
            f"- `{feature}` separability AUC: Fold 0 `{rows.iloc[0].separability_auc:.3f}`, Fold 1 `{rows.iloc[1].separability_auc:.3f}`; "
            f"best orientations: `{rows.iloc[0].best_orientation}`, `{rows.iloc[1].best_orientation}`."
        )
    lines += [
        "",
        "A separability AUC near 0.5 indicates little rescue-versus-harmful separation. Fold disagreement counts are small, so these statistics are descriptive and high variance.",
        "",
        "## Cross-fold transfer",
        "",
    ]
    for _, row in transfers.iterrows():
        if row.rule_type == "always_high":
            rule_text = "`always_high` / `no_switch` (threshold `n/a`)"
        else:
            rule_text = f"`{row.rule_type}` `{row.selected_feature}` `{row.orientation}` threshold `{row.threshold:.6g}`"
        lines.append(
            f"- Fold {int(row.source_fold)} -> Fold {int(row.target_fold)}: {rule_text}; "
            f"target switched/rescue/harm/net = {int(row.switched_count)}/{int(row.rescue_count)}/{int(row.harm_count)}/{int(row.net_gain)}; "
            f"accuracy delta `{row.delta_accuracy:+.4f}`, AUC delta `{row.delta_auc:+.4f}`, macro-F1 delta `{row.delta_macro_f1:+.4f}`."
        )
    lines += [
        "",
        "The within-fold rules are exploratory upper bounds only; they are not cross-fold generalization results. Rule selection maximized source-fold net_gain over the fixed one-dimensional candidate space, then froze the rule for the target fold.",
        "",
        "## Artifacts",
        "",
        "See `disagreement_cases.csv`, `feature_summary.csv`, `feature_separability.csv`, `cross_fold_transfer.csv`, `selective_fusion_metrics.csv`, `within_fold_optimal_rules.csv`, and `class_distribution.csv`.",
    ]
    (output / "stage343_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_dir", type=Path, default=Path("analysis/stage3_model_design/04_scale_complementarity/stage342_independent_complementarity"))
    parser.add_argument("--output_dir", type=Path, default=Path("analysis/stage3_model_design/04_scale_complementarity/stage343_low_rescue_separability"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        fold: _load_fold(args.predictions_dir / f"fold_{fold}_branch_predictions.csv", fold)
        for fold in (0, 1)
    }
    disagreements = _disagreement_cases(frames)
    disagreements.to_csv(args.output_dir / "disagreement_cases.csv", index=False)
    _feature_summary(frames).to_csv(args.output_dir / "feature_summary.csv", index=False)
    _feature_separability(frames).to_csv(args.output_dir / "feature_separability.csv", index=False)
    _class_distribution(disagreements).to_csv(args.output_dir / "class_distribution.csv", index=False)

    transfers = []
    within_rows = []
    selective_rows = []
    for source_fold, target_fold in ((0, 1), (1, 0)):
        source_rule = _select_rule(frames[source_fold][frames[source_fold].disagreement])
        # The source rule is selected using only source disagreements and then
        # applied unchanged to the target fold below.
        source_eval = _evaluate_rule(frames[source_fold], source_rule)
        target_eval = _evaluate_rule(frames[target_fold], source_rule)
        transfers.append({
            "source_fold": source_fold, "target_fold": target_fold,
            "rule_type": source_rule["rule_type"], "selected_feature": source_rule["feature"], "orientation": source_rule["orientation"], "threshold": source_rule["threshold"],
            "source_switched_count": source_eval["switched_count"], "source_rescue_count": source_eval["rescue_count"], "source_harm_count": source_eval["harm_count"], "source_net_gain": source_eval["net_gain"], "source_switch_precision": source_eval["switch_precision"],
            "switched_count": target_eval["switched_count"], "rescue_count": target_eval["rescue_count"], "harm_count": target_eval["harm_count"], "net_gain": target_eval["net_gain"], "switch_precision": target_eval["switch_precision"],
            "high_auc": target_eval["high_auc"], "high_accuracy": target_eval["high_accuracy"], "high_macro_f1": target_eval["high_macro_f1"],
            "selective_auc": target_eval["selective_auc"], "selective_accuracy": target_eval["selective_accuracy"], "selective_macro_f1": target_eval["selective_macro_f1"],
            "delta_auc": target_eval["delta_auc"], "delta_accuracy": target_eval["delta_accuracy"], "delta_macro_f1": target_eval["delta_macro_f1"],
        })
        selective_rows.append({"evaluation": f"transfer_{source_fold}_to_{target_fold}", **transfers[-1]})

        within_rule = _select_rule(frames[target_fold])
        within_eval = _evaluate_rule(frames[target_fold], within_rule)
        within_rows.append({"fold": target_fold, **within_eval, "exploratory_upper_bound": True})
        selective_rows.append({"evaluation": f"within_fold_{target_fold}_exploratory", "source_fold": target_fold, "target_fold": target_fold, **within_eval})

    transfers_frame = pd.DataFrame(transfers)
    transfers_frame.to_csv(args.output_dir / "cross_fold_transfer.csv", index=False)
    pd.DataFrame(within_rows).to_csv(args.output_dir / "within_fold_optimal_rules.csv", index=False)
    pd.DataFrame(selective_rows).to_csv(args.output_dir / "selective_fusion_metrics.csv", index=False)
    _write_summary(args.output_dir, frames, transfers_frame, pd.DataFrame(within_rows))
    print(transfers_frame.to_string(index=False))


if __name__ == "__main__":
    main()
