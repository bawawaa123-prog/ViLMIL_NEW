"""Reuse frozen Stage351 source mapping and Stage335/336/337 diagnostics."""
from pathlib import Path
import hashlib
import json
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
MATCHED = BASE.parent / '01_matched_fivefold'
sys.path.insert(0, str(ROOT))
sys.path.append(str(MATCHED))
import common as matched

MODES = ('normal', 'context_shuffle', 'context_mean', 'residual_off')
SEEDS = (352, 353, 354)
GATE_LOW, GATE_HIGH = .05, .95
NEAR_MARGIN = .10  # |P1-P0|, diagnostic stratum only; never a prediction threshold
EPS = 1e-8


def slide_seed(seed, fold, slide):
    payload = f'{seed}/{fold}/{slide}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def jobs(folds, modes, seeds, include_e2=False):
    rows = []
    for fold in folds:
        rows.append(dict(fold=fold, arm='E0', mode='normal', seed=0))
        if include_e2:
            rows.append(dict(fold=fold, arm='E2', mode='normal', seed=0))
        for mode in modes:
            for seed in seeds if mode == 'context_shuffle' else (0,):
                rows.append(dict(fold=fold, arm='E1', mode=mode, seed=seed))
    return rows


def audit(folds, include_e2=False):
    matched.environment()
    split = matched.check_inputs()
    rows = [matched.collect(arm, fold) for fold in folds
            for arm in (('E0','E1','E2') if include_e2 else ('E0','E1'))]
    for row in rows:
        if row['status'] != 'Complete':
            raise RuntimeError(f'Unavailable/incompatible Stage351 source: {row}')
        schema = matched.checkpoint_schema(Path(row['checkpoint']), row['arm'])
        if any('.module' in k or 'instance_loss_fn' in k for k in schema):
            raise ValueError('Legacy key sanitization would alter checkpoint; exact keys required')
    shared = [ {k:v for k,v in matched.checkpoint_schema(Path(r['checkpoint']),r['arm']).items()
                if not k.startswith('high_router.')} for r in rows]
    if any(s != shared[0] for s in shared):
        raise ValueError('Shared checkpoint schemas differ')
    return rows, split


def checkpoint_identity(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def write_json(path, payload):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write('\n')


def build_model(row, prompts, device):
    import torch
    from tools import stage335_routing_diagnostics as s335
    from types import SimpleNamespace
    if row['arm'] == 'E1':
        args = SimpleNamespace(checkpoint=row['checkpoint'], scale_mode='high',
                               prototype_number=16, seed=1)
        model = s335._model(args, 2, prompts, 'normal')  # audited exact schema; strict load
        model.to(device)
    else:
        import ml_collections
        from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
        cfg = ml_collections.ConfigDict(dict(input_size=512, hidden_size=192,
            prototype_number=16, text_prompt=prompts, scale_mode=matched.ARMS[row['arm']][0],
            use_low_context_routing=False, use_routing_stabilization=False,
            use_global_proto_context=False, finetune_text_encoder=False))
        model = ViLa_MIL_BiomedCLIP(config=cfg, num_classes=2)
        model.load_state_dict(torch.load(row['checkpoint'], map_location='cpu',
                              weights_only=True, mmap=True), strict=True)
        model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
