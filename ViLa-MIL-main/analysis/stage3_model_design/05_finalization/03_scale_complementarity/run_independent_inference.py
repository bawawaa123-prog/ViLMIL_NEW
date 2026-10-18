"""Manual frozen Low Validation replay; reuse authenticated Stage352 High CSVs."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import shutil
import sys
import numpy as np
import pandas as pd
from stage353_common import *


def infer_low(source):
    import torch
    from tools import stage335_routing_diagnostics as s335
    from tools import stage336_context_shuffle_stability as s336
    from utils.utils import get_simple_loader
    s336._seed_everything(1)
    config = SimpleNamespace(task='task_adenocarcinoma', csv_path=str(matched.CSV),
        data_root_dir=str(matched.DATA), data_folder_s='features_biomedclip_5x',
        data_folder_l='features_biomedclip_20x', mapping_path=str(matched.MAPPING),
        text_prompt=None, text_prompt_path=str(matched.PROMPT))
    prompts = s335._prompts(config)
    dataset, _ = s335._dataset(config, prompts)
    validation = dataset.return_splits(from_id=False, csv_path=str(matched.SPLITS / f'splits_{source["fold"]}.csv'))[1]
    loader = get_simple_loader(validation, batch_size=1, num_workers=0, mode='transformer')
    model = build_clean_model(source, prompts, s335.DEVICE)
    rows = []
    with torch.no_grad():
        for batch in loader:
            tensors = [t.to(s335.DEVICE) for t in batch[:5]]
            probabilities, _, _ = model(*tensors, mapping=batch[6][0])
            p = probabilities[0].detach().cpu().numpy()
            rows.append(dict(fold=source['fold'], arm='L', split='val', status='Complete',
                checkpoint=source['checkpoint'], slide_id=str(batch[5][0]), label=int(tensors[4].item()),
                prediction=int(p.argmax()), p_0=float(p[0]), p_1=float(p[1])))
    del model, dataset, loader
    if s335.DEVICE.type=='cuda':
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def inspect_inference(root):
    root = Path(root).resolve()
    manifest = read_json(root / 'experiment_manifest.json')
    if manifest['split'] != 'val' or validate_folds(manifest['folds']) != manifest['folds']:
        raise ValueError('Invalid inference plan')
    if manifest['core_code'] != code_hashes():
        raise ValueError('Inference model code changed')
    _, high_frames = load_stage352(manifest['stage352_run'], manifest['stage352_report'])
    if manifest['stage352_manifest_sha256'] != matched.digest(Path(manifest['stage352_run'])/'experiment_manifest.json'):
        raise ValueError('Changed Stage352 source')
    if (root/'completion.json').is_file():
        done = read_json(root/'completion.json')
        if done['manifest_sha256'] != matched.digest(root/'experiment_manifest.json') or done['status']!='Complete':
            raise ValueError('Root completion mismatch')
        required = {f'fold{f}/completion.json' for f in manifest['folds']}
        if set(done['sha256']) != required:
            raise ValueError('Incomplete root completion hashes')
        verify_hashes(root, done['sha256'])
    records, frames = [], {}
    for f in manifest['folds']:
        directory = root / f'fold{f}'
        item = dict(fold=f, status='Pending', issue='Missing completed inference')
        if directory.exists() and not (directory/'completion.json').exists():
            item.update(status='Invalid', issue='Partial inference; never auto-resume')
        if (directory/'completion.json').is_file():
            try:
                row = inspect_low(manifest['low_run'], f)
                if row['status']!='Complete' or row!=manifest['low_sources'][str(f)]:
                    raise ValueError('Low training source changed/unavailable')
                if manifest['inputs'][str(f)] != input_hashes(f):
                    raise ValueError('Changed inputs')
                done = read_json(directory/'completion.json')
                if done['fold']!=f or done['status']!='Complete' or set(done['sha256'])!={'high_predictions.csv','low_predictions.csv'}:
                    raise ValueError('Invalid job completion metadata')
                if done['checkpoint_identity']!=routing.checkpoint_identity(row['checkpoint']):
                    raise ValueError('Low checkpoint identity changed')
                verify_hashes(directory, done['sha256'])
                high = pd.read_csv(directory/'high_predictions.csv', dtype={'slide_id':str})
                low = pd.read_csv(directory/'low_predictions.csv', dtype={'slide_id':str})
                hsource = matched.collect('E0', f)
                if hsource['status']!='Complete':
                    raise ValueError('High source unavailable')
                assert_metrics(validate_predictions(high, f, 'E0', hsource['checkpoint']), hsource)
                assert_metrics(validate_predictions(low, f, 'L', row['checkpoint']), row)
                a, b = align(high, high_frames[(f,'E0','normal',0)])
                if not np.array_equal(a[['p_0','p_1','prediction']].to_numpy(), b[['p_0','p_1','prediction']].to_numpy()):
                    raise ValueError('Copied High predictions differ from Stage352')
                align(high, low)
                frames[f] = (high, low)
                item.update(status='Complete', issue='')
            except Exception as exc:
                item.update(status='Invalid', issue=f'{type(exc).__name__}: {exc}')
        records.append(item)
    return manifest, pd.DataFrame(records), frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--low-run-dir', type=Path, default=DEFAULT_TRAIN)
    p.add_argument('--stage352-run-dir', type=Path, default=DEFAULT_352_RUN)
    p.add_argument('--stage352-report-dir', type=Path, default=DEFAULT_352_REPORT)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    print('Python:', matched.environment())
    folds = validate_folds(args.folds)
    matched.check_inputs()
    _, high_frames = load_stage352(args.stage352_run_dir, args.stage352_report_dir)
    sources = {str(f):inspect_low(args.low_run_dir, f) for f in folds}
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for f in folds:
        hsource = matched.collect('E0', f)
        if hsource['status']!='Complete' or sources[str(f)]['status']=='Invalid':
            raise ValueError('Invalid source')
        assert_metrics(validate_predictions(high_frames[(f,'E0','normal',0)], f, 'E0', hsource['checkpoint']), hsource)
        print(f'Fold {f}: H=authenticated Stage352 E0, L={sources[str(f)]["status"]}')
    if not args.execute:
        print('Dry-run only; Pending Low sources must finish before --execute. No inference/output.'); return
    if any(r['status']!='Complete' for r in sources.values()):
        raise ValueError('Pending Low training; no inference started')
    root = reserve(args.output_dir.resolve(), private=True)
    lock = root / '.running.lock'; lock.touch(exist_ok=False)
    write_json(root/'experiment_manifest.json', dict(folds=folds, split='val',
        low_run=str(args.low_run_dir.resolve()), low_sources=sources,
        stage352_run=str(args.stage352_run_dir.resolve()), stage352_report=str(args.stage352_report_dir.resolve()),
        stage352_manifest_sha256=matched.digest(args.stage352_run_dir/'experiment_manifest.json'),
        inputs={str(f):input_hashes(f) for f in folds}, core_code=code_hashes(),
        analysis_code={str(x):matched.digest(x) for x in BASE.glob('*.py')},
        note='High copied from matching Stage352 E0; only independent Low is re-evaluated; all raw IDs local under mode700 directory.'))
    from run_diagnostics import Tee
    with (root/'run.log').open('x') as handle:
        oldout, olderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(oldout, handle), Tee(olderr, handle)
        try:
            for f in folds:
                out = reserve(root/f'fold{f}')
                low = infer_low(sources[str(f)])
                assert_metrics(validate_predictions(low, f, 'L', sources[str(f)]['checkpoint']), sources[str(f)])
                # Preserve exact authenticated CSV bytes, avoiding a second float round-trip.
                shutil.copyfile(args.stage352_run_dir/f'fold{f}/E0/normal_seed0/predictions.csv',out/'high_predictions.csv')
                low.to_csv(out/'low_predictions.csv', index=False)
                write_json(out/'completion.json', dict(fold=f, status='Complete',
                    checkpoint_identity=routing.checkpoint_identity(sources[str(f)]['checkpoint']),
                    sha256={x.name:matched.digest(x) for x in out.glob('*.csv')}))
                print(f'Completed Low Validation fold {f}', flush=True)
        finally:
            sys.stdout, sys.stderr = oldout, olderr
    write_json(root/'completion.json', dict(status='Complete',manifest_sha256=matched.digest(root/'experiment_manifest.json'),
        sha256={f'fold{f}/completion.json':matched.digest(root/f'fold{f}/completion.json') for f in folds}))
    lock.unlink()
    print('Completed independent inference:', root)


if __name__=='__main__':
    main()
