"""Frozen Stage 3.5.0 protocol and read-only artifact validation."""
from __future__ import annotations

import ast
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
FINAL = BASE.parent
SPLITS = ROOT / 'splits/strict/task_adenocarcinoma_100_k5_s1'
CSV = ROOT / 'dataset_csv/all_data.csv'
PROMPT = ROOT / 'text_prompt/adenocarcinoma_dual_scale_prompt.csv'
DATA = ROOT / 'data/yiyuan'
MAPPING = ROOT / 'analysis/stage3_model_design/01_cross_scale_mapping/full_output/mappings'
ARMS = {'E0': ('high', False), 'E1': ('high', True), 'E2': ('dual', False)}
METRICS = ('val_auc', 'val_acc', 'val_macro_f1')
LABELS = {'Adenocarcinoma': 0, 'NonAdenocarcinoma': 1}


def environment():
    if Path(sys.prefix).name != 'vila_mil_overlay_rt':
        raise RuntimeError(f'必须使用 vila_mil_overlay_rt，当前解释器：{sys.executable}')
    return sys.executable


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def destinations(arm, fold):
    """Only explicitly approved sources; never search historical Dual results."""
    arm_name = 'clean_high' if arm == 'E0' else 'routed_control'
    if arm != 'E2' and fold < 2:
        code = f'stage350_{arm_name}_folds01'
        parent = FINAL / '00_clean_high_baseline' / arm_name
        return [(parent / f'{code}_s1', parent / 'train_folds01.log')]
    new = BASE / 'runs' / arm / f'fold{fold}'
    code = f'stage351_matched_{arm}_fold{fold}'
    choices = [(new / f'{code}_s1', new / 'train.log')]
    if arm != 'E2':
        old = FINAL / '01_frozen_confirmation' / arm_name
        choices.insert(0, (old / f'fold{fold}' / f'stage351_{arm_name}_fold{fold}_s1',
                           old / f'fold{fold}_train.log'))
    return choices


def split_members(path):
    frame = pd.read_csv(path, dtype=str)
    return {name: frame[name].dropna().tolist() for name in ('train', 'val', 'test')}


def same_split(a, b):
    aa, bb = split_members(a), split_members(b)
    return all(len(aa[k]) == len(bb[k]) and set(aa[k]) == set(bb[k]) for k in aa)


def check_inputs():
    """Metadata/path checks only. No feature extraction, inference or split edits."""
    frame = pd.read_csv(CSV, dtype=str)
    if frame[['case_id', 'slide_id', 'label']].isna().any().any():
        raise ValueError('CSV contains missing identifiers/labels')
    if frame.slide_id.duplicated().any() or not set(frame.label) <= set(LABELS):
        raise ValueError('Duplicate slides or unsupported label values')
    prompt = pd.read_csv(PROMPT)
    if len(prompt) != 2 or not {'low_resolution_description', 'high_resolution_description'} <= set(prompt):
        raise ValueError('Expected two classes and both prompt columns')
    if prompt[['low_resolution_description', 'high_resolution_description']].isna().any().any():
        raise ValueError('Empty prompt')
    all_ids = set(frame.slide_id)
    cases = frame.set_index('slide_id').case_id.to_dict()
    candidate_file = ROOT / 'analysis/stage2_yiyuan_data_audit/01_patient_case_audit/possible_same_patient.csv'
    candidates = pd.read_csv(candidate_file, dtype=str) if candidate_file.exists() else None
    audit = []
    test_sets = []
    for fold in range(5):
        split = split_members(SPLITS / f'splits_{fold}.csv')
        sets = {k: set(v) for k, v in split.items()}
        if any(len(split[k]) != len(sets[k]) for k in split):
            raise ValueError(f'Fold {fold}: duplicate split members')
        if set.union(*sets.values()) != all_ids or sum(map(len, sets.values())) != len(all_ids):
            raise ValueError(f'Fold {fold}: overlap or incomplete CSV coverage')
        case_sets = {k: {cases[s] for s in v} for k, v in sets.items()}
        pairs = [('train', 'val'), ('train', 'test'), ('val', 'test')]
        overlap = sum(len(case_sets[a] & case_sets[b]) for a, b in pairs)
        if overlap:
            raise ValueError(f'Fold {fold}: literal case_id overlap ({overlap})')
        for name, ids in sets.items():
            if len(set(frame.loc[frame.slide_id.isin(ids), 'label'])) != 2:
                raise ValueError(f'Fold {fold}/{name}: both classes required')
        candidate_cross = None
        if candidates is not None:
            candidate_cross = sum(
                sum(bool(set(group.slide_id) & ids) for ids in sets.values()) > 1
                for _, group in candidates.groupby('candidate_group_id'))
        audit.append({'fold': fold, **{k: len(v) for k, v in sets.items()},
                      'literal_case_overlap': overlap, 'candidate_cross_split_groups': candidate_cross})
        test_sets.append(sets['test'])
    if set.union(*test_sets) != all_ids or sum(map(len, test_sets)) != len(all_ids):
        raise ValueError('Test folds are not a fivefold partition')
    missing = []
    for slide in frame.slide_id:
        for name in ('features_biomedclip_5x', 'features_biomedclip_20x'):
            path = DATA / name / f'{slide}.h5'
            fallback = DATA / name / f'{slide.split(".")[0]}.h5'
            if not any(p.is_file() and p.stat().st_size > 0 for p in (path, fallback)):
                missing.append(str(path))
        path = MAPPING / f'{slide}.npz'
        if not path.is_file() or not path.stat().st_size:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing/empty input files: {missing[:8]}')
    return audit


def command(arm, fold, result_parent):
    scale, routing = ARMS[arm]
    code = f'stage351_matched_{arm}_fold{fold}'
    args = [sys.executable, '-u', 'main.py', '--data_root_dir', 'data/yiyuan',
            '--data_folder_s', 'features_biomedclip_5x', '--data_folder_l', 'features_biomedclip_20x',
            '--task', 'task_adenocarcinoma', '--split_dir', str(SPLITS.relative_to(ROOT)),
            '--model_type', 'ViLa_MIL_BiomedCLIP', '--mode', 'transformer',
            '--text_prompt_path', str(PROMPT.relative_to(ROOT)), '--prototype_number', '16',
            '--k', '5', '--k_start', str(fold), '--k_end', str(fold), '--seed', '1',
            '--lr', '1e-4', '--reg', '1e-5', '--opt', 'adam', '--bag_loss', 'ce',
            '--drop_out', '--max_epochs', '80', '--patience', '10', '--early_stopping',
            '--scale_mode', scale, '--mapping_path', str(MAPPING.relative_to(ROOT)),
            '--skip_test_evaluation', '--exp_code', code, '--results_dir', str(result_parent)]
    if routing:
        args += ['--use_low_context_routing', '--use_routing_stabilization']
    return args


def checkpoint_schema(path, arm):
    import torch
    # mmap avoids reading gigabytes of frozen pretrained tensors just to inspect schema.
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    routed = any(k.startswith('high_router.') for k in state)
    if routed != ARMS[arm][1] or any(k.startswith('global_proto_context') for k in state):
        raise ValueError('Checkpoint routing/global-context schema mismatch')
    router_shapes = {k: tuple(v.shape) for k, v in state.items() if k.startswith('high_router.')}
    expected_router = {
        'high_router.context_projection.weight': (512, 512),
        'high_router.context_projection.bias': (512,),
        'high_router.route_score.0.weight': (128, 1026),
        'high_router.route_score.0.bias': (128,),
        'high_router.route_score.2.weight': (1, 128),
        'high_router.route_score.2.bias': (1,),
    }
    if routed and router_shapes != expected_router:
        raise ValueError('Missing/unexpected routing weights or incompatible router shapes')
    if tuple(state['learnable_image_center'].shape) != (16, 1, 512):
        raise ValueError('Checkpoint prototype shape mismatch')
    if tuple(state['attention_V.0.weight'].shape) != (192, 512):
        raise ValueError('Checkpoint hidden size mismatch')
    return {k: tuple(v.shape) for k, v in state.items()}


def inspect_result(path, log, arm, fold, check_schema=True):
    required = ['fold_summary.csv', 'epoch_details.csv', f's_{fold}_checkpoint.pt', f'splits_{fold}.csv']
    for name in required:
        p = path / name
        if not p.is_file() or not p.stat().st_size:
            raise FileNotFoundError(f'Missing/empty {p}')
    configs = list(path.glob('experiment_*.txt'))
    if len(configs) != 1:
        raise ValueError(f'Expected exactly one config in {path}')
    cfg = ast.literal_eval(configs[0].read_text().strip())
    expected = dict(num_splits=5, task='task_adenocarcinoma', max_epochs=80, lr=1e-4,
                    label_frac=1.0, seed=1, model_type='ViLa_MIL_BiomedCLIP', mode='transformer',
                    scale_mode=ARMS[arm][0], skip_test_evaluation=True,
                    use_low_context_routing=ARMS[arm][1], use_routing_stabilization=ARMS[arm][1],
                    use_global_proto_context=False, init_checkpoint_template=None,
                    use_drop_out=True, weighted_sample=False, opt='adam', patience=10,
                    split_dir=str(SPLITS.relative_to(ROOT)), mapping_path=str(MAPPING.relative_to(ROOT)))
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'{path}: {key}={cfg.get(key)!r}, expected {value!r}')
    start, end = (0, 1) if fold < 2 and path.parent.parent.name == '00_clean_high_baseline' else (fold, fold)
    if (cfg['k_start'], cfg['k_end']) != (start, end):
        raise ValueError('Config fold range mismatch')
    if not same_split(path / f'splits_{fold}.csv', SPLITS / f'splits_{fold}.csv'):
        raise ValueError(f'Fold {fold}: saved split differs from strict5')
    if not log.is_file() or 'end script' not in log.read_text(errors='replace')[-4096:]:
        raise ValueError(f'Incomplete/missing training log {log}')
    if ARMS[arm][1] and (not (path / 'routing_diagnostics.jsonl').is_file()
                         or not (path / 'routing_diagnostics.jsonl').stat().st_size):
        raise ValueError('Missing routing diagnostics')
    summary = pd.read_csv(path / 'fold_summary.csv')
    tests = [c for c in summary if c.startswith('test_')]
    if not tests or summary[tests].notna().any().any():
        raise ValueError('Expected empty test columns (validation-only)')
    epochs = pd.read_csv(path / 'epoch_details.csv')
    rows = epochs[epochs.fold == fold + 1].sort_values('epoch')
    final = summary[summary.fold == fold + 1]
    if len(final) != 1 or rows.empty:
        raise ValueError('Missing/duplicate fold summary or epochs')
    if rows.epoch.tolist() != list(range(1, len(rows) + 1)) or len(rows) > 80:
        raise ValueError('Epoch sequence is incomplete')
    cols = ['val_acc', 'val_auc', 'val_loss']
    if not np.isfinite(rows[cols].to_numpy()).all():
        raise ValueError('Non-finite validation metrics')
    selected = rows.loc[rows.val_acc.idxmax()]  # first strict maximum; ties never overwrite
    if len(rows) != 80 and len(rows) - int(selected.epoch) != 10:
        raise ValueError('No evidence of completed patience=10 training')
    for metric in ('val_acc', 'val_auc'):
        if not np.isclose(float(final.iloc[0][metric]), selected[metric], atol=1e-12, rtol=0):
            raise ValueError(f'{metric}: selected epoch and checkpoint summary disagree')
    f1 = float(selected.get('val_f1', np.nan))
    if not 0 <= selected.val_acc <= 1 or not 0 <= selected.val_auc <= 1:
        raise ValueError('Invalid AUC/ACC')
    if np.isfinite(f1) and not 0 <= f1 <= 1:
        raise ValueError('Invalid validation Macro-F1')
    if check_schema:
        checkpoint_schema(path / f's_{fold}_checkpoint.pt', arm)
    return dict(arm=arm, fold=fold, status='Complete' if np.isfinite(f1) else 'MissingMetric',
                val_auc=float(selected.val_auc), val_acc=float(selected.val_acc), val_macro_f1=f1,
                selected_epoch=int(selected.epoch), epochs_run=len(rows), source=str(path),
                checkpoint=str(path / f's_{fold}_checkpoint.pt'),
                metric_source='selected checkpoint epoch_details.csv', issue='')


def collect(arm, fold, check_schema=True):
    choices = destinations(arm, fold)
    existing = [(p, log) for p, log in choices if p.exists() or log.exists()
                or (p.parent / '.running.lock').exists()
                or (BASE / 'runs' in p.parents and p.parent.exists())]
    if len(existing) > 1:
        return dict(arm=arm, fold=fold, status='Invalid/Partial', source=str(existing),
                    issue='Ambiguous sources; resolve explicitly, never overwrite',
                    **{k: np.nan for k in METRICS})
    if not existing:
        return dict(arm=arm, fold=fold, status='Pending', source=str(choices[-1][0]),
                    issue='No training artifacts', **{k: np.nan for k in METRICS})
    path, log = existing[0]
    try:
        if BASE / 'runs' in path.parents:
            if (path.parent / '.running.lock').exists() or not (path.parent / 'completion.json').is_file():
                raise ValueError('New run is active/partial or has no completion marker; never auto-resume')
            manifest = json.loads((path.parent / 'run_manifest.json').read_text())
            for item in (CSV, PROMPT, SPLITS / f'splits_{fold}.csv'):
                if manifest['sha256'].get(str(item)) != digest(item):
                    raise ValueError(f'Input changed since training: {item}')
        return inspect_result(path, log, arm, fold, check_schema)
    except Exception as exc:
        return dict(arm=arm, fold=fold, status='Invalid/Partial', source=str(path),
                    issue=f'{type(exc).__name__}: {exc}', **{k: np.nan for k in METRICS})


def provenance(args):
    paths = [CSV, PROMPT, *(SPLITS / f'splits_{f}.csv' for f in range(5)),
             ROOT / 'main.py', ROOT / 'utils/core_utils.py', ROOT / 'utils/utils.py',
             ROOT / 'datasets/dataset_generic.py', ROOT / 'models/model_ViLa_MIL_BiomedCLIP.py',
             ROOT / 'models/cross_scale_modules.py', BASE / 'README.md',
             FINAL / '00_clean_high_baseline/run_stage350_matched_clean_high.sh',
             FINAL / '01_frozen_confirmation/run_stage351_folds234.sh',
             ROOT / 'analysis/stage2_yiyuan_data_audit/01_patient_case_audit/possible_same_patient.csv',
             *BASE.glob('*.py'), *BASE.glob('*.sh')]
    return {'python': sys.executable, 'argv': args, 'sha256': {str(p): digest(p) for p in paths},
            'package_versions': {name: metadata.version(name) for name in
                                 ('torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'open_clip_torch', 'h5py')},
            'protocol': {'reg': 1e-5, 'prototype_number': 16, 'prompt_lr': 1e-3,
                         'text_encoder_frozen': True, 'rho': 0.10,
                         'checkpoint_selection': 'first maximum validation accuracy'},
            'feature_paths': [str(DATA / n) for n in ('features_biomedclip_5x', 'features_biomedclip_20x')],
            'note': 'Feature contents are reused, not rehashed; old runs have no retrospective source/input hashes.'}


def write_json(path, payload):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
