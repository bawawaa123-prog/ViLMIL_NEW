"""Frozen independent-scale protocol; reuse Stage351/352 source validators."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.append(str(BASE.parent / '02_routing_diagnostics'))
import bridge as routing
matched = routing.matched
from evaluate_validation import metrics_from_predictions
from analysis_utils import align, class_metrics

DEFAULT_TRAIN = BASE / 'runs/low_matched_v1'
DEFAULT_352_RUN = BASE.parent / '02_routing_diagnostics/runs/stage352_full_v1'
DEFAULT_352_REPORT = BASE.parent / '02_routing_diagnostics/reports/20261010_130747_580789'
GROUPS = ('G1', 'G2', 'G3', 'G4')
MARGIN_BINS = (0., .10, .50, .90, 1.)


def write_json(path, value):
    matched.write_json(path, value)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_folds(folds):
    if not folds or len(set(folds)) != len(folds) or not set(folds) <= set(range(5)):
        raise ValueError('Expected unique folds in 0..4')
    return sorted(folds)


def reserve(path, private=False):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    if private:
        path.chmod(0o700)
    return path


def low_command(fold, parent):
    # Exact E0 training argv; only scale, experiment label and destination differ.
    argv = matched.command('E0', fold, parent)
    argv[argv.index('--scale_mode') + 1] = 'low'
    argv[argv.index('--exp_code') + 1] = f'stage353_low_fold{fold}'
    return argv


def low_destination(root, fold):
    parent = Path(root).resolve() / f'fold{fold}'
    return parent, parent / f'stage353_low_fold{fold}_s1'


def input_hashes(fold):
    paths = [matched.CSV, matched.PROMPT, matched.SPLITS / f'splits_{fold}.csv']
    return {str(p): matched.digest(p) for p in paths}


def code_hashes():
    paths = [ROOT / n for n in ('main.py', 'utils/core_utils.py', 'utils/utils.py',
             'models/model_ViLa_MIL_BiomedCLIP.py', 'models/cross_scale_modules.py',
             'datasets/dataset_generic.py')]
    return {str(p): matched.digest(p) for p in paths}


def verify_hashes(root, hashes):
    for name, value in hashes.items():
        if matched.digest(Path(root) / name) != value:
            raise ValueError(f'Changed/missing artifact: {name}')


def historical_audit():
    import torch
    sources = [
        ('Stage2 Low', ROOT / 'analysis/stage2_yiyuan_data_audit/06_scale_ablation/low_only/adenocarcinoma_biomedclip_low_only_strict5_s1'),
        ('Stage342 Low', ROOT / 'analysis/stage3_model_design/04_scale_complementarity/stage342_matched_low_only/stage342_matched_low_only_folds01_s1'),
        ('Stage341 Dual-trained', ROOT / 'analysis/stage3_model_design/04_scale_complementarity/stage341_matched_dual_baseline/stage341_matched_dual_baseline_folds01_s1')]
    records = []
    for name, path in sources:
        configs = list(path.glob('experiment_*.txt'))
        cfg = ast.literal_eval(configs[0].read_text()) if len(configs) == 1 else {}
        folds = [f for f in range(5) if (path / f's_{f}_checkpoint.pt').is_file()]
        issues = []
        required = dict(scale_mode='low', skip_test_evaluation=True, use_low_context_routing=False,
                        use_routing_stabilization=False, use_global_proto_context=False,
                        init_checkpoint_template=None, num_splits=5, task='task_adenocarcinoma',
                        max_epochs=80, lr=1e-4, label_frac=1., seed=1,
                        model_type='ViLa_MIL_BiomedCLIP', mode='transformer', use_drop_out=True,
                        weighted_sample=False, opt='adam', patience=10,
                        split_dir=str(matched.SPLITS.relative_to(ROOT)))
        for key, value in required.items():
            if key not in cfg or cfg[key] != value:
                issues.append(f'{key}: recorded {cfg.get(key, "UNRECORDED")!r}, required {value!r}')
        checkpoints=[]
        for fold in folds:
            state=torch.load(path/f's_{fold}_checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
            summary=pd.read_csv(path/'fold_summary.csv')
            epochs=pd.read_csv(path/'epoch_details.csv')
            part=epochs[epochs.fold==fold+1].sort_values('epoch')
            selected=part.loc[part.val_acc.idxmax()] if len(part) else None
            final=summary[summary.fold==fold+1]
            selected_matches=bool(selected is not None and len(final)==1 and all(
                np.isclose(final.iloc[0][k],selected[k],atol=1e-12,rtol=0) for k in ('val_acc','val_auc')))
            checkpoints.append(dict(fold=fold,identity=routing.checkpoint_identity(path/f's_{fold}_checkpoint.pt'),
                exact_strict5=matched.same_split(path/f'splits_{fold}.csv',matched.SPLITS/f'splits_{fold}.csv'),
                router_keys=[k for k in state if k.startswith('high_router.')],
                global_context_keys=[k for k in state if k.startswith('global_proto_context')],
                prototype_shape=list(state['learnable_image_center'].shape),
                selected_epoch=int(selected.epoch) if selected is not None else None,
                first_max_val_acc_matches_summary=selected_matches,
                test_metrics_present=bool(final[[k for k in final if k.startswith('test_')]].notna().any().any())))
            del state
        script=ROOT/'analysis/stage3_model_design/04_scale_complementarity/run_stage342_matched_low_only.sh'
        records.append(dict(name=name, source=str(path), checkpoint_folds=folds,checkpoint_audit=checkpoints,
            status='Rejected for matched baseline', issues=issues,
            recorded_config=cfg,
            protocol_evidence=dict(dataset='Existing all_data.csv by historical task convention; no historical CSV hash',
                features='Existing 5x/20x BiomedCLIP folders; no historical content hash',
                prompt='Stage342 retained launcher names adenocarcinoma_dual_scale_prompt.csv; Stage2 lacks contemporaneous argv/prompt hash',
                reg='Stage342 retained launcher explicitly 1e-5; Stage2 config omits weight decay',
                bag_loss='Stage342 launcher uses historical default; no contemporaneous argv for Stage2',
                text_encoder='Stage342 training log confirms frozen text and prompt LR=.001; Stage2 lacks retained log here',
                prototype='Checkpoint shape audited per fold',early_stopping='Current implementation selects first max validation ACC; historical summary checked for consistency',
                independent='Low scale config; Stage342 init_checkpoint_template=None; Stage2 warm-start field unrecorded',
                retained_stage342_launcher_sha256=matched.digest(script)),
            note='Low loss may bypass Router, but historical flag/schema and initialization provenance are not matched; no claim that Router caused Low errors.'))
    return records


def inspect_low(root, fold, require_completion=True):
    parent, path = low_destination(root, fold)
    if not parent.exists():
        return dict(arm='L', fold=fold, status='Pending', source=str(path), issue='No matched independent Low run')
    try:
        manifest = read_json(parent / 'run_manifest.json')
        if manifest['fold'] != fold or manifest['argv'] != low_command(fold, parent):
            raise ValueError('Training command/fold differs from frozen E0-derived protocol')
        if manifest['inputs'] != input_hashes(fold) or manifest['core_code'] != code_hashes():
            raise ValueError('Inputs/training code changed since launch')
        configs = list(path.glob('experiment_*.txt'))
        if len(configs) != 1:
            raise ValueError('Expected one training config')
        cfg = ast.literal_eval(configs[0].read_text())
        expected = dict(num_splits=5, k_start=fold, k_end=fold, task='task_adenocarcinoma',
            max_epochs=80, lr=1e-4, label_frac=1., seed=1, model_type='ViLa_MIL_BiomedCLIP',
            mode='transformer', scale_mode='low', skip_test_evaluation=True,
            use_low_context_routing=False, use_routing_stabilization=False,
            use_global_proto_context=False, init_checkpoint_template=None, use_drop_out=True,
            weighted_sample=False, opt='adam', patience=10,
            split_dir=str(matched.SPLITS.relative_to(ROOT)), mapping_path=str(matched.MAPPING.relative_to(ROOT)))
        if any(cfg.get(k) != v for k, v in expected.items()):
            raise ValueError('Saved config does not match protocol')
        if not matched.same_split(path / f'splits_{fold}.csv', matched.SPLITS / f'splits_{fold}.csv'):
            raise ValueError('Saved strict5 split differs')
        log = parent / 'train.log'
        if 'end script' not in log.read_text(errors='replace')[-4096:]:
            raise ValueError('Training did not finish')
        epochs = pd.read_csv(path / 'epoch_details.csv')
        epochs = epochs[epochs.fold == fold + 1].sort_values('epoch')
        summary = pd.read_csv(path / 'fold_summary.csv')
        final = summary[summary.fold == fold + 1]
        test_cols = [c for c in summary if c.startswith('test_')]
        if not test_cols or summary[test_cols].notna().any().any():
            raise ValueError('Test evaluation is forbidden')
        if len(final) != 1 or epochs.empty or epochs.epoch.tolist() != list(range(1, len(epochs) + 1)):
            raise ValueError('Incomplete/duplicate epoch sequence')
        if not np.isfinite(epochs[['val_acc','val_auc','val_loss','val_f1']].to_numpy()).all():
            raise ValueError('Nonfinite training metrics')
        selected = epochs.loc[epochs.val_acc.idxmax()]
        if len(epochs) > 80 or (len(epochs) != 80 and len(epochs) - int(selected.epoch) != 10):
            raise ValueError('Training completion/patience mismatch')
        for k in ('val_auc', 'val_acc'):
            if not np.isclose(final.iloc[0][k], selected[k], atol=1e-12, rtol=0):
                raise ValueError('Selected checkpoint summary mismatch')
        ckpt = path / f's_{fold}_checkpoint.pt'
        matched.checkpoint_schema(ckpt, 'E0')
        row = dict(arm='L', fold=fold, status='Complete', source=str(path), checkpoint=str(ckpt),
                   selected_epoch=int(selected.epoch), val_auc=float(selected.val_auc),
                   val_acc=float(selected.val_acc), val_macro_f1=float(selected.val_f1), issue='')
        if require_completion:
            if (parent / '.running.lock').exists():
                raise ValueError('Active/interrupted run')
            completed = read_json(parent / 'completion.json')
            if completed['result'] != row or completed['manifest_sha256'] != matched.digest(parent / 'run_manifest.json'):
                raise ValueError('Completion/source mismatch')
            if completed['checkpoint_identity'] != routing.checkpoint_identity(ckpt):
                raise ValueError('Checkpoint identity changed')
            verify_hashes(parent, completed['sha256'])
        return row
    except Exception as exc:
        return dict(arm='L', fold=fold, status='Invalid', source=str(path), issue=f'{type(exc).__name__}: {exc}')


def validate_predictions(frame, fold, arm=None, checkpoint=None):
    if frame.slide_id.isna().any() or frame.slide_id.duplicated().any():
        raise ValueError('Missing/duplicate Slide ID')
    expected = set(matched.split_members(matched.SPLITS / f'splits_{fold}.csv')['val'])
    if set(frame.slide_id) != expected:
        raise ValueError('Validation coverage mismatch')
    labels = pd.read_csv(matched.CSV, dtype=str).set_index('slide_id').label.map(matched.LABELS)
    if frame.label.tolist() != labels.loc[frame.slide_id].tolist():
        raise ValueError('Label mismatch')
    for key, value in dict(fold=fold, split='val', arm=arm, checkpoint=checkpoint).items():
        if value is not None and (key not in frame or not frame[key].eq(value).all()):
            raise ValueError(f'Prediction source mismatch: {key}')
    return metrics_from_predictions(frame)


def assert_metrics(metrics, source):
    for key in matched.METRICS:
        if not np.isclose(metrics[key], source[key], atol=1e-7, rtol=0):
            raise ValueError(f'Replay differs from selected checkpoint: {key}')


def load_stage352(run_dir=DEFAULT_352_RUN, report_dir=DEFAULT_352_REPORT):
    from aggregate_diagnostics import inspect_run
    manifest, state, frames = inspect_run(Path(run_dir).resolve())
    if len(state) != 35 or not state.status.eq('Complete').all() or not (Path(run_dir) / 'completion.json').is_file() or (Path(run_dir) / '.running.lock').exists():
        raise ValueError('Requires authenticated complete default35 Stage352 jobs')
    report_manifest = read_json(Path(report_dir) / 'experiment_manifest.json')
    if Path(report_manifest['source_run']).resolve() != Path(run_dir).resolve() or report_manifest['source_manifest_sha256'] != matched.digest(Path(run_dir) / 'experiment_manifest.json'):
        raise ValueError('Stage352 report/run provenance mismatch')
    verify_hashes(report_dir, report_manifest['output_sha256'])
    return manifest, frames


def build_clean_model(source, prompts, device):
    import torch
    import ml_collections
    from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
    cfg = ml_collections.ConfigDict(dict(input_size=512, hidden_size=192, prototype_number=16,
        text_prompt=prompts, scale_mode='low' if source['arm']=='L' else 'high',
        use_low_context_routing=False, use_routing_stabilization=False,
        use_global_proto_context=False, finetune_text_encoder=False))
    model = ViLa_MIL_BiomedCLIP(config=cfg, num_classes=2)
    model.load_state_dict(torch.load(source['checkpoint'], map_location='cpu', weights_only=True, mmap=True), strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def case_token(salt, slide):
    import hmac
    return 'C' + hmac.new(salt, str(slide).encode(), hashlib.sha256).hexdigest()[:24]
