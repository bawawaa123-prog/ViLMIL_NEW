"""Inference-only routing residual-scale intervention on selected folds."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import h5py
import ml_collections
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
from utils.cross_scale_mapping import CrossScaleMapping

SPLIT_DIR = ROOT / "splits/strict/task_adenocarcinoma_100_k5_s1"
FEATURE_S = ROOT / "data/yiyuan/features_biomedclip_5x"
FEATURE_L = ROOT / "data/yiyuan/features_biomedclip_20x"
MAPPING_DIR = ROOT / "analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings"
CHECKPOINTS = {
    1: ROOT / "analysis/stage3_model_design/03_low_context_routing/03_pilot_validation/results/E1/stage332_pilot_E1_low_context_routing_s1/s_0_checkpoint.pt",
    2: ROOT / "analysis/stage3_model_design/03_low_context_routing/04_multifold_replication/results/E1_fold1/stage332_multifold_E1_fold1_s1/s_1_checkpoint.pt",
}


def prompts(path):
    df = pd.read_csv(path)
    cols = [str(c).strip().lower() for c in df.columns]
    if 'low_resolution_description' in cols and 'high_resolution_description' in cols:
        low = df.iloc[:, cols.index('low_resolution_description')].astype(str).tolist()
        high = df.iloc[:, cols.index('high_resolution_description')].astype(str).tolist()
        return low + high
    return [str(x) for x in df.to_numpy().reshape(-1).tolist()]


def read_test_slides(fold):
    rows = pd.read_csv(SPLIT_DIR / f"splits_{fold - 1}.csv")
    return rows['test'].dropna().astype(str).tolist()


def load_slide(slide):
    with h5py.File(FEATURE_S / f"{slide}.h5", 'r') as f:
        low = torch.from_numpy(f['features'][:]).float().unsqueeze(0)
        low_coord = torch.from_numpy(f['coords'][:]).unsqueeze(0)
    with h5py.File(FEATURE_L / f"{slide}.h5", 'r') as f:
        high = torch.from_numpy(f['features'][:]).float().unsqueeze(0)
        high_coord = torch.from_numpy(f['coords'][:]).unsqueeze(0)
    mapping = CrossScaleMapping.load_npz(MAPPING_DIR / f"{slide}.npz").to_dict()
    return low, low_coord, high, high_coord, mapping


def evaluate_fold(fold, lambdas, output_dir, prompt_path):
    slides = read_test_slides(fold)
    ckpt = CHECKPOINTS[fold]
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    labels_df = pd.read_csv(ROOT / 'dataset_csv/all_data.csv').set_index('slide_id')
    label_map = {'Adenocarcinoma': 0, 'NonAdenocarcinoma': 1}
    rows = []
    for lam in lambdas:
        cfg = ml_collections.ConfigDict()
        cfg.hidden_size = 192; cfg.text_prompt = prompts(prompt_path)
        cfg.prototype_number = 16; cfg.scale_mode = 'high'
        cfg.use_low_context_routing = True; cfg.routing_scale = float(lam)
        cfg.finetune_text_encoder = False; cfg.text_finetune_mode = 'proj'; cfg.text_unfreeze_last_n = 2
        torch.manual_seed(1)
        model = ViLa_MIL_BiomedCLIP(cfg, num_classes=2)
        model.load_state_dict(state, strict=True)
        model.eval()
        probs, labels, diagnostics = [], [], []
        with torch.no_grad():
            for slide in slides:
                low, low_coord, high, high_coord, mapping = load_slide(slide)
                raw_label = labels_df.loc[slide, 'label']
                if str(raw_label) in label_map:
                    label = label_map[str(raw_label)]
                elif str(raw_label).isdigit():
                    label = int(raw_label)
                else:
                    label = None
                if label is None:
                    raise ValueError(f'Unsupported label for slide {slide}: {raw_label!r}')
                prob, _, _ = model(low, low_coord, high, high_coord, torch.tensor([label]), mapping=mapping)
                d = dict(model.last_routing_diagnostics or {})
                context = model._last_mapping_context['high_parent_context']
                high_norm = float(high.squeeze(0).norm().cpu())
                context_norm = float(context.norm().cpu())
                d.update({'slide_id': slide, 'lambda': float(lam), 'high_norm': high_norm,
                          'context_norm': context_norm,
                          'residual_high_ratio': d.get('routing_residual_norm', 0.0) / max(high_norm, 1e-12),
                          'parent_count_mean': float((mapping['high_parent_ptr'][1:] - mapping['high_parent_ptr'][:-1]).mean()),
                          'parent_count_max': int((mapping['high_parent_ptr'][1:] - mapping['high_parent_ptr'][:-1]).max()),
                          'padding_ratio_mean': float(np.asarray(mapping['high_padding_ratio']).mean()),
                          'padding_ratio_max': float(np.asarray(mapping['high_padding_ratio']).max())})
                diagnostics.append(d); probs.append(float(prob[0, 1])); labels.append(label)
        pred = [int(p >= 0.5) for p in probs]
        summary = {'fold': fold, 'lambda': float(lam),
                   'auc': float(roc_auc_score(labels, probs)),
                   'accuracy': float(accuracy_score(labels, pred)),
                   'macro_f1': float(f1_score(labels, pred, average='macro', zero_division=0)),
                   'slide_count': len(slides),
                   'residual_high_ratio_mean': float(np.mean([d['residual_high_ratio'] for d in diagnostics])),
                   'route_mean_mean': float(np.mean([d['route_mean'] for d in diagnostics])),
                   'route_std_mean': float(np.mean([d['route_std'] for d in diagnostics])),
                   'context_norm_mean': float(np.mean([d['context_norm'] for d in diagnostics])),
                   'high_norm_mean': float(np.mean([d['high_norm'] for d in diagnostics])),
                   'mapped_ratio_mean': float(np.mean([d['mapped_high_count']/d['high_count'] for d in diagnostics])),
                   'parent_count_mean': float(np.mean([d['parent_count_mean'] for d in diagnostics])),
                   'padding_ratio_mean': float(np.mean([d['padding_ratio_mean'] for d in diagnostics]))}
        (output_dir / f'fold{fold}_lambda{lam:g}_slides.jsonl').write_text('\n'.join(json.dumps(d) for d in diagnostics) + '\n')
        rows.append(summary)
        del model
    with (output_dir / f'fold{fold}_lambda_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', nargs='+', type=int, default=[1, 2], choices=[1, 2])
    ap.add_argument('--output-dir', default=str(Path(__file__).parent / 'lambda_results'))
    ap.add_argument('--prompt-path', default=str(ROOT / 'text_prompt/adenocarcinoma_dual_scale_prompt.csv'))
    ap.add_argument('--lambdas', nargs='+', type=float, default=[0, .1, .25, .5, 1.0])
    args = ap.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for fold in args.folds: evaluate_fold(fold, args.lambdas, out, Path(args.prompt_path))


if __name__ == '__main__': main()
