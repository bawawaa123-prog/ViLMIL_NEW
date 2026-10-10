"""Optional manual validation replay, strict checkpoint loading, no Test loader."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

from common import (ARMS, CSV, DATA, LABELS, MAPPING, PROMPT, ROOT, SPLITS,
                    check_inputs, collect, digest, environment, write_json)


def metrics_from_predictions(frame):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    labels = frame.label.to_numpy(dtype=int)
    pred = frame.prediction.to_numpy(dtype=int)
    prob = frame[['p_0', 'p_1']].to_numpy(dtype=float)
    if set(labels) != {0, 1} or not set(pred) <= {0, 1}:
        raise ValueError('Expected both validation classes and binary predictions')
    if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError('Invalid probabilities')
    if not np.allclose(prob.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Class probabilities do not sum to 1')
    if not np.array_equal(pred, prob.argmax(axis=1)):
        raise ValueError('Prediction/probability column mismatch')
    return dict(val_auc=float(roc_auc_score(labels, prob[:, 1])),
                val_acc=float(accuracy_score(labels, pred)),
                val_macro_f1=float(f1_score(labels, pred, average='macro', zero_division=0)))


def validation_identity(arm, fold, row):
    ckpt = Path(row['checkpoint'])
    stat = ckpt.stat()
    return {'arm': arm, 'fold': fold, 'split': 'val', 'checkpoint': str(ckpt),
            'checkpoint_size': stat.st_size, 'checkpoint_mtime_ns': stat.st_mtime_ns,
            'selected_epoch': int(row['selected_epoch']),
            'csv_sha256': digest(CSV), 'prompt_sha256': digest(PROMPT),
            'split_sha256': digest(SPLITS / f'splits_{fold}.csv'),
            'epoch_sha256': digest(Path(row['source']) / 'epoch_details.csv'),
            'label_mapping': LABELS}


def replay(arm, fold, row, output):
    import torch
    import ml_collections
    from datasets.dataset_generic import Generic_MIL_Dataset
    from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
    from utils.utils import get_split_loader
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = ml_collections.ConfigDict()
    cfg.input_size, cfg.hidden_size, cfg.prototype_number = 512, 192, 16
    prompt = pd.read_csv(PROMPT)
    cfg.text_prompt = prompt.low_resolution_description.astype(str).tolist() + prompt.high_resolution_description.astype(str).tolist()
    cfg.scale_mode, cfg.use_low_context_routing = ARMS[arm]
    cfg.use_routing_stabilization = ARMS[arm][1]
    cfg.use_global_proto_context = False
    cfg.routing_residual_ratio = 0.10
    cfg.finetune_text_encoder = False
    cfg.text_finetune_mode = 'proj'
    cfg.text_unfreeze_last_n = 2
    dataset = Generic_MIL_Dataset(csv_path=str(CSV), mode='transformer',
        data_dir_s=str(DATA / 'features_biomedclip_5x'), data_dir_l=str(DATA / 'features_biomedclip_20x'),
        shuffle=False, print_info=False, label_dict=LABELS, patient_strat=False, ignore=[],
        mapping_path=str(MAPPING), return_mapping=ARMS[arm][1])
    validation = dataset.return_splits(from_id=False, csv_path=str(SPLITS / f'splits_{fold}.csv'))[1]
    loader = get_split_loader(validation, mode='transformer')
    model = ViLa_MIL_BiomedCLIP(config=cfg, num_classes=2)
    state = torch.load(row['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            tensors = [x.to(device) for x in batch[:5]]
            mapping = batch[6][0] if len(batch) >= 7 else None
            prob, _, _ = model(*tensors, mapping=mapping)
            p = prob[0].detach().cpu().numpy()
            rows.append(dict(slide_id=str(validation.slide_data.iloc[index].slide_id),
                             label=int(tensors[4].item()), prediction=int(p.argmax()),
                             p_0=float(p[0]), p_1=float(p[1])))
    frame = pd.DataFrame(rows)
    metrics = metrics_from_predictions(frame)
    for key in ('val_auc', 'val_acc', 'val_macro_f1'):
        if np.isfinite(row[key]) and not np.isclose(metrics[key], row[key], atol=1e-8, rtol=0):
            raise ValueError(f'{arm}/{fold}: replay {key} disagrees with recorded selected epoch')
    frame.to_csv(output / 'validation_predictions.csv', index=False)
    write_json(output / 'validation_metrics.json', {**validation_identity(arm, fold, row),
        **metrics, 'strict_load': True, 'metric_source': 'validation checkpoint replay',
        'predictions_sha256': digest(output / 'validation_predictions.csv')})
    del model, dataset, loader
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(arm, fold, metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--folds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    parser.add_argument('--output-dir', type=Path, required=True, help='New destination; existing files are never replaced')
    parser.add_argument('--execute', action='store_true', help='Manual permission to perform validation inference')
    args = parser.parse_args()
    print('Python:', environment())
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    check_inputs()
    plan = []
    for arm in dict.fromkeys(args.arms):
        for fold in dict.fromkeys(args.folds):
            row = collect(arm, fold)
            if row['status'] not in ('Complete', 'MissingMetric'):
                raise RuntimeError(f'Cannot evaluate unavailable/invalid checkpoint: {row}')
            output = args.output_dir.resolve() / arm / f'fold{fold}'
            if output.exists():
                raise FileExistsError(f'Replay destination already exists: {output}')
            plan.append((arm, fold, row, output))
            print(f'Validation replay plan: {arm}/{fold}, {row["checkpoint"]} -> {output}')
    if not args.execute:
        print('Dry run only. No model was constructed or inference performed.')
        return
    for arm, fold, row, output in plan:
        output.mkdir(parents=True, exist_ok=False)
        replay(arm, fold, row, output)


if __name__ == '__main__':
    main()
